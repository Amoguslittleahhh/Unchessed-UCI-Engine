#!/usr/bin/env python3
"""Fixed-depth reviewer benchmark with self-consistent UCI settings.

Usage:
  python tools/bench_research.py CURRENT [--baseline BASELINE] [--depth 12]

The harness refuses the adaptive adapter, pins MultiPV=1/Threads=1/Hash=128,
reads through bestmove, and compares throughput rather than raw wall time when
trees differ. Output is JSON for archival and independent analysis.
"""

import argparse
import json
import math
import re
import subprocess
import time

POSITIONS = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "position3": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "position4": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "position5": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "position6": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "closed": "r1bq1rk1/pp2bppp/2np1n2/2p1p3/4P3/2PP1N2/PP1N1PPP/R1BQR1K1 w - - 4 9",
    "open_tactical": "3r2k1/pp3ppp/2p1b3/4P3/1q1P1Q2/2R5/PP3PPP/4R1K1 w - - 0 22",
    "rook_endgame": "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/4R3 w - - 0 40",
    "pawn_endgame": "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/8 w - - 0 40",
    "queenless": "r3k2r/ppp2ppp/2npbn2/3pp3/8/2N1PN2/PPPPBPPP/R3K2R w KQkq - 6 9",
    "sharp": "r1bq1rk1/ppp2ppp/2np1n2/4p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 9",
}

FIELD = re.compile(r"\b(depth|nodes|nps|time) (\d+)")


class Reviewer:
    def __init__(self, path, options=()):
        self.process = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.send("uci")
        handshake = []
        while True:
            line = self.process.stdout.readline().rstrip("\n")
            handshake.append(line)
            if line == "uciok":
                break
        identity = next((line for line in handshake if line.startswith("id name ")), "")
        if "Reviewer" not in identity:
            self.close()
            raise SystemExit(f"benchmark requires unchessed-reviewer, got: {identity}")
        for command in (
            "setoption name MultiPV value 1",
            "setoption name Threads value 1",
            "setoption name Hash value 128",
        ):
            self.send(command)
        for option in options:
            name, value = option.split("=", 1)
            self.send(f"setoption name {name} value {value}")

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def search(self, fen, depth):
        self.send("ucinewgame")
        self.send("position fen " + fen)
        started = time.monotonic()
        self.send(f"go depth {depth}")
        last_info = ""
        bestmove = ""
        score = ""
        while True:
            line = self.process.stdout.readline().rstrip("\n")
            if line.startswith("info depth "):
                last_info = line
                match = re.search(r"\bscore (cp|mate) (-?\d+)", line)
                if match:
                    score = f"{match.group(1)} {match.group(2)}"
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                break
        fields = {key: int(value) for key, value in FIELD.findall(last_info)}
        return {
            **fields,
            "wall_ms": round((time.monotonic() - started) * 1000, 3),
            "bestmove": bestmove,
            "score": score,
        }

    def close(self):
        if self.process.poll() is None:
            self.send("quit")
            self.process.wait(timeout=5)


def run(path, depth, options=()):
    engine = Reviewer(path, options)
    try:
        return {name: engine.search(fen, depth) for name, fen in POSITIONS.items()}
    finally:
        engine.close()


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current")
    parser.add_argument("--baseline")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        help="current-engine UCI option as NAME=VALUE (repeatable)",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    result = {
        "depth": args.depth,
        "options": args.option,
        "current": run(args.current, args.depth, args.option),
    }
    if args.baseline:
        baseline = run(args.baseline, args.depth)
        result["baseline"] = baseline
        speedups = []
        for name in POSITIONS:
            old = baseline[name]["nps"]
            new = result["current"][name]["nps"]
            speedups.append(new / old)
        result["nps_geomean_speedup"] = geometric_mean(speedups)
        result["bestmove_agreement"] = sum(
            result["current"][name]["bestmove"] == baseline[name]["bestmove"]
            for name in POSITIONS
        )
        result["positions"] = len(POSITIONS)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")


if __name__ == "__main__":
    main()
