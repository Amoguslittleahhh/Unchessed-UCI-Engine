#!/usr/bin/env python3
"""Small headless UCI benchmark runner for environments where cutechess-cli cannot start games."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import chess


class UciEngine:
    def __init__(self, command: str, options: dict[str, object], name: str):
        self.name = name
        self.proc = subprocess.Popen(
            [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert self.proc.stdin and self.proc.stdout
        self.inp = self.proc.stdin
        self.out = self.proc.stdout
        self._send("uci")
        self._wait_for("uciok", 15)
        for key, value in options.items():
            self._send(f"setoption name {key} value {value}")
        self._send("isready")
        self._wait_for("readyok", 15)

    def _send(self, line: str) -> None:
        self.inp.write(line + "\n")
        self.inp.flush()

    def _wait_for(self, token: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        lines = []
        while time.monotonic() < deadline:
            line = self.out.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)
            if line.startswith(token) or token in line:
                return "\n".join(lines)
        raise RuntimeError(f"{self.name}: timed out waiting for {token}; output={lines[-10:]}")

    def move(self, board: chess.Board, movetime_ms: int) -> tuple[chess.Move, str]:
        self._send("position fen " + board.fen())
        self._send(f"go movetime {movetime_ms}")
        output = self._wait_for("bestmove", max(5.0, movetime_ms / 1000.0 * 20))
        line = next(x for x in output.splitlines() if x.startswith("bestmove "))
        uci = line.split()[1]
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"{self.name}: illegal move {uci} in {board.fen()}")
        return move, output

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def play_game(unchessed: str, stockfish: str, sf_elo: int, game_id: int, movetime: int, max_plies: int):
    white_is_unchessed = game_id % 2 == 0
    u = UciEngine(unchessed, {"Threads": 1, "Hash": 16, "Adaptive": "true", "PersonaSmooth": "false", "EngineDetectV2": "false"}, "Unchessed")
    s = UciEngine(stockfish, {"Threads": 1, "Hash": 16, "UCI_LimitStrength": "true", "UCI_Elo": sf_elo}, f"Stockfish-{sf_elo}")
    board = chess.Board()
    moves = []
    telemetry = []
    try:
        for ply in range(max_plies):
            engine = u if ((board.turn == chess.WHITE) == white_is_unchessed) else s
            move, output = engine.move(board, movetime)
            moves.append(move.uci())
            telemetry.append({"ply": ply + 1, "side": "white" if board.turn else "black", "engine": engine.name, "move": move.uci(), "info_tail": output.splitlines()[-3:]})
            board.push(move)
            if board.outcome(claim_draw=True):
                break
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "1/2-1/2"
        if result == "1/2-1/2":
            score = 0.5
        elif (result == "1-0") == white_is_unchessed:
            score = 1.0
        else:
            score = 0.0
        return {"game": game_id + 1, "sf_elo": sf_elo, "white": "Unchessed" if white_is_unchessed else f"Stockfish {sf_elo}", "result": result, "unchessed_score": score, "plies": len(moves), "moves": moves, "telemetry": telemetry}
    finally:
        u.close()
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unchessed", required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--elos", nargs="+", type=int, default=[1320, 1800, 2400])
    parser.add_argument("--games-per-elo", type=int, default=2)
    parser.add_argument("--movetime-ms", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = []
    for elo in args.elos:
        for i in range(args.games_per_elo):
            game_id = len(results)
            started = time.time()
            result = play_game(args.unchessed, args.stockfish, elo, game_id, args.movetime_ms, args.max_plies)
            result["elapsed_seconds"] = round(time.time() - started, 3)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("game", "sf_elo", "white", "result", "unchessed_score", "plies", "elapsed_seconds")}), flush=True)
    Path(args.output).write_text(json.dumps({"configuration": vars(args), "games": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
