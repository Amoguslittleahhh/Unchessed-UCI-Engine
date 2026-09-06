#!/usr/bin/env python3
"""Benchmark scalar fallback and runtime-dispatched engine builds.

The harness records completed UCI searches only. It never treats a partial
"info" line as a completed measurement and labels cross-compiled targets as
compile-only because this host cannot execute them natively.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

POSITIONS = [
    ("startpos", "position startpos"),
    ("kiwipete", "position fen r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1"),
    ("middlegame", "position fen 4rrk1/pp1b1ppp/2n1p3/2qpP3/3N4/2P1B3/PPQ2PPP/R4RK1 w - - 0 16"),
    ("endgame", "position fen 8/5pk1/6p1/3p4/3P1P2/5KP1/8/8 w - - 0 40"),
]

INFO_RE = re.compile(r"^info .*?nodes (\d+) .*?nps (\d+).*?time (\d+)")


def run_once(binary: Path, position: str, hash_mb: int, nodes: int, scalar: bool) -> dict[str, str]:
    env = os.environ.copy()
    if scalar:
        env["UNCHESSED_DISABLE_SIMD"] = "1"
    started = time.monotonic()
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write("\n".join([
        "uci",
        "setoption name Threads value 1",
        f"setoption name Hash value {hash_mb}",
        "setoption name EvalFile value " + str(Path("benchmarks/artifacts/benchmark-v1.unchnnue").resolve()),
        "setoption name Adaptive value false",
        "setoption name OwnBook value false",
        "isready",
        position,
        f"go nodes {nodes}",
        "",
    ]))
    proc.stdin.flush()
    lines: list[str] = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip("\n"))
        if line.startswith("bestmove "):
            break
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)
    elapsed = time.monotonic() - started
    completed = [INFO_RE.search(line) for line in lines if line.startswith("info ")]
    completed = [match for match in completed if match]
    if not completed:
        raise RuntimeError(f"no completed info line from {binary}:\n{lines[-20:]}")
    match = completed[-1]
    return {
        "nodes": match.group(1),
        "time_ms": match.group(3),
        "nps": match.group(2),
        "wall_ms": str(round(elapsed * 1000)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hash", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--label", required=True)
    parser.add_argument("--scalar", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for repeat in range(args.repeats):
        for hash_mb in args.hash:
            for name, position in POSITIONS:
                measurement = run_once(args.binary, position, hash_mb, args.nodes, args.scalar)
                rows.append(
                    {
                        "label": args.label,
                        "mode": "scalar" if args.scalar else "dispatch",
                        "repeat": str(repeat),
                        "hash_mb": str(hash_mb),
                        "position": name,
                        **measurement,
                    }
                )
                print(rows[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
