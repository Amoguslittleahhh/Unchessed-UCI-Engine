#!/usr/bin/env python3
"""Fixed-control candidate-vs-Stockfish calibration tournament.

This is a research harness, not an Elo claim. Each pairing is played twice with
colors reversed. The candidate can vary existing HCE UCI weights or enable the
original HomemadeNonlinear mode. Results are written as JSONL and summarized
by variant and time control.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import chess


class Engine:
    def __init__(self, path: str, name: str, options: dict[str, str]):
        self.name = name
        self.process = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        self.wait(lambda line: line == "uciok", 20)
        for key, value in options.items():
            self.send(f"setoption name {key} value {value}")
        self.send("isready")
        self.wait(lambda line: line == "readyok", 20)

    def _reader(self):
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))

    def send(self, command: str):
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait(self, predicate, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=max(0.02, deadline - time.time()))
            except queue.Empty as exc:
                raise RuntimeError(f"{self.name}: timeout") from exc
            if predicate(line):
                return line
        raise RuntimeError(f"{self.name}: timeout")

    def move(self, board: chess.Board, movetime: int) -> str:
        self.send(f"position fen {board.fen()}")
        self.send(f"go movetime {movetime}")
        line = self.wait(lambda value: value.startswith("bestmove"), movetime / 1000 + 15)
        return line.split()[1]

    def close(self):
        try:
            self.send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


@dataclass(frozen=True)
class Variant:
    name: str
    options: dict[str, str]


VARIANTS = [
    Variant("hce-default", {"HomemadeNonlinear": "false"}),
    Variant("hce-mobility-110", {"HomemadeNonlinear": "false", "MobilityPct": "110"}),
    Variant("hce-passed-110", {"HomemadeNonlinear": "false", "PassedPawnMgPct": "110", "PassedPawnEgPct": "110"}),
    Variant("homemade-nonlinear", {"HomemadeNonlinear": "true"}),
]


def play(candidate_path: str, stockfish_path: str, variant: Variant, movetime: int, candidate_white: bool, max_plies: int):
    candidate = Engine(candidate_path, f"candidate-{variant.name}", {
        "Adaptive": "false", "OwnBook": "false", "Threads": "1", "Hash": "16", **variant.options,
    })
    reference = Engine(stockfish_path, "stockfish19", {"Threads": "1", "Hash": "16", "OwnBook": "false"})
    engines = (candidate, reference) if candidate_white else (reference, candidate)
    board = chess.Board()
    moves = []
    started = time.time()
    try:
        for ply in range(max_plies):
            side_engine = engines[ply % 2]
            move = side_engine.move(board, movetime)
            if move == "0000":
                break
            try:
                board.push_uci(move)
            except ValueError as exc:
                raise RuntimeError(f"{side_engine.name} emitted illegal move {move} at {board.fen()}") from exc
            moves.append(move)
            if board.outcome(claim_draw=True) is not None:
                break
        outcome = board.outcome(claim_draw=True)
        result = "1/2-1/2" if outcome is None or outcome.winner is None else ("1-0" if outcome.winner else "0-1")
        candidate_score = 1.0 if (result == "1-0" and candidate_white) or (result == "0-1" and not candidate_white) else 0.0 if result != "1/2-1/2" else 0.5
        return {
            "variant": variant.name, "movetime_ms": movetime, "candidate_white": candidate_white,
            "result": result, "candidate_score": candidate_score, "plies": len(moves),
            "seconds": round(time.time() - started, 3), "final_fen": board.fen(),
        }
    finally:
        candidate.close()
        reference.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--movetimes", default="30,100,300")
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=160)
    args = parser.parse_args()
    movetimes = [int(value) for value in args.movetimes.split(",")]
    out = Path(args.out)
    records = []
    for variant in VARIANTS:
        for movetime in movetimes:
            for game in range(args.games_per_pair):
                candidate_white = game % 2 == 0
                record = play(args.candidate, args.stockfish, variant, movetime, candidate_white, args.max_plies)
                records.append(record)
                print(json.dumps(record), flush=True)
    out.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    print("summary")
    for variant in VARIANTS:
        for movetime in movetimes:
            rows = [r for r in records if r["variant"] == variant.name and r["movetime_ms"] == movetime]
            score = sum(r["candidate_score"] for r in rows) / len(rows)
            print(f"{variant.name} movetime={movetime} games={len(rows)} score={score:.3f}")


if __name__ == "__main__":
    main()
