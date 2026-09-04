#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path

import chess

TELEMETRY_RE = re.compile(r"event=persona_decision .*?mode_after=(?P<mode>[A-Z]+)")
OBSERVATION_RE = re.compile(r"event=opponent_observation .*?estimate_elo=(?P<elo>-?\d+) confidence_cp=(?P<conf>-?\d+)")

class UCI:
    def __init__(self, command, name):
        self.name = name
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.q = queue.Queue()
        self.lines = []
        self.send("uci")
        self._read_until("uciok", 30)
        self.send("isready")
        self._read_until("readyok", 60)

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _read_until(self, token, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} exited while waiting for {token}")
            line = line.rstrip("\n")
            self.lines.append(line)
            if token in line:
                return line
        raise TimeoutError(f"{self.name} timed out waiting for {token}")

    def new_game(self, options):
        self.send("ucinewgame")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self._read_until("readyok", 120)

    def move(self, board, clocks, inc_ms):
        hist = " ".join(m.uci() for m in board.move_stack)
        self.send("position startpos" + (" moves " + hist if hist else ""))
        self.send(f"go wtime {clocks[chess.WHITE]} btime {clocks[chess.BLACK]} winc {inc_ms} binc {inc_ms}")
        deadline = time.monotonic() + max(30.0, clocks[board.turn] / 1000.0 + inc_ms / 1000.0 + 10.0)
        telemetry = []
        best = None
        while time.monotonic() < deadline:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} exited during search")
            line = line.rstrip("\n")
            self.lines.append(line)
            m = TELEMETRY_RE.search(line)
            if m:
                telemetry.append(m.group("mode"))
            obs = OBSERVATION_RE.search(line)
            if obs:
                elos.append({"elo": int(obs.group("elo")), "confidence": int(obs.group("conf"))})
            if line.startswith("bestmove "):
                best = line.split()[1]
                break
        if best is None:
            raise TimeoutError(f"{self.name} timed out without bestmove")
        return best, telemetry

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def parse_engine(spec):
    return spec.split(" ")


def play_game(white_spec, black_spec, game_id, base_options, inc_ms, initial_ms, max_plies):
    white = UCI(parse_engine(white_spec["command"]), white_spec["name"])
    black = UCI(parse_engine(black_spec["command"]), black_spec["name"])
    engines = {chess.WHITE: white, chess.BLACK: black}
    try:
        white.new_game(base_options.get(white.name, {}))
        black.new_game(base_options.get(black.name, {}))
        board = chess.Board()
        clocks = {chess.WHITE: initial_ms, chess.BLACK: initial_ms}
        modes = []
        elos = []
        started = time.time()
        termination = "unknown"
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                termination = board.outcome(claim_draw=True).termination.name
                break
            eng = engines[board.turn]
            before = time.monotonic()
            move_uci, telemetry = eng.move(board, clocks, inc_ms)
            elapsed = int((time.monotonic() - before) * 1000)
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                termination = "illegal_move"
                break
            if move not in board.legal_moves:
                termination = "illegal_move"
                break
            clocks[board.turn] = clocks[board.turn] - elapsed + inc_ms
            if clocks[board.turn] <= 0:
                termination = "flag"
                break
            for mode in telemetry:
                modes.append(mode)
            board.push(move)
        else:
            termination = "max_plies"
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        return {
            "game": game_id,
            "white": white.name,
            "black": black.name,
            "result": result,
            "termination": termination,
            "plies": len(board.move_stack),
            "elapsed_s": round(time.time() - started, 3),
            "persona_decisions": len(modes),
            "mode_switches": sum(a != b for a, b in zip(modes, modes[1:])),
            "modes": {m: modes.count(m) for m in sorted(set(modes))},
            "elo_first": elos[0]["elo"] if elos else None,
            "elo_last": elos[-1]["elo"] if elos else None,
            "confidence_first": elos[0]["confidence"] if elos else None,
            "confidence_last": elos[-1]["confidence"] if elos else None,
            "pgn_moves": " ".join(m.uci() for m in board.move_stack),
        }
    finally:
        white.close()
        black.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unchessed", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--maia", required=True)
    ap.add_argument("--eval-file", required=True, help="explicit NNUE file; refuses silent HCE fallback")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--initial-ms", type=int, default=180000)
    ap.add_argument("--inc-ms", type=int, default=2000)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eval_file = Path(args.eval_file).expanduser().resolve()
    if not eval_file.is_file() or eval_file.stat().st_size == 0:
        ap.error(f"NNUE file is missing or empty: {eval_file}")
    # Disable opening-book shortcuts so Elo/persona evidence and move quality are
    # measured from the same start position under the requested clock.
    options = {
        "Unchessed Game Adapter": {"EvalFile": str(eval_file), "OwnBook": "false", "AdapterTelemetry": "true", "PersonaSmooth": "true"},
        "Stockfish": {"UCI_AnalyseMode": "false", "Threads": "1", "Hash": "32"},
        "Maia-3": {"Threads": "1"},
    }
    specs = {
        "Unchessed": {"name": "Unchessed Game Adapter", "command": args.unchessed},
        "Stockfish": {"name": "Stockfish", "command": args.stockfish},
        "Maia-3": {"name": "Maia-3", "command": args.maia},
    }
    pairings = [("Unchessed", "Stockfish"), ("Unchessed", "Maia-3")]
    results = []
    for pair_index, (a, b) in enumerate(pairings):
        for i in range(args.games):
            white, black = ((a, b) if i % 2 == 0 else (b, a))
            print(f"game {pair_index * args.games + i + 1}: {white} vs {black}", flush=True)
            results.append(play_game(specs[white], specs[black], pair_index * args.games + i + 1,
                                     options, args.inc_ms, args.initial_ms, args.max_plies))
    Path(args.out).write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
