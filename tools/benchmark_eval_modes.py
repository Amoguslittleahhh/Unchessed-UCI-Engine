#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from pathlib import Path

INFO = re.compile(r"info depth (\d+).*?nodes (\d+).*?nps (\d+).*?time (\d+)")


def run_mode(binary: str, mode: str, fens: list[str], depth: int):
    process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdin and process.stdout
    process.stdin.write("uci\nsetoption name Adaptive value false\nsetoption name OwnBook value false\nsetoption name Threads value 1\n")
    if mode == "nonlinear":
        process.stdin.write("setoption name HomemadeNonlinear value true\n")
    process.stdin.write("isready\n")
    process.stdin.flush()
    for line in process.stdout:
        if line.rstrip() == "readyok":
            break
    rows = []
    for fen in fens:
        process.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
        process.stdin.flush()
        last = None
        for line in process.stdout:
            line = line.rstrip()
            match = INFO.search(line)
            if match and int(match.group(1)) == depth:
                last = tuple(int(value) for value in match.groups())
            if line.startswith("bestmove"):
                break
        if last is None:
            raise RuntimeError(f"missing final info for {mode}: {fen}")
        rows.append(last)
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=10)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--epd", required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fens = [" ".join(line.split()[:4]) for line in Path(args.epd).read_text().splitlines() if line.strip() and not line.startswith("#")]
    base = run_mode(args.binary, "base", fens, args.depth)
    nonlinear = run_mode(args.binary, "nonlinear", fens, args.depth)
    base_nps = [row[2] for row in base]
    nonlinear_nps = [row[2] for row in nonlinear]
    base_ms = [row[3] for row in base]
    nonlinear_ms = [row[3] for row in nonlinear]
    with Path(args.out).open("w") as handle:
        handle.write("# Nonlinear evaluator overhead benchmark\n\n")
        handle.write(f"corpus={args.epd}\npositions={len(fens)}\ndepth={args.depth}\n")
        handle.write(f"base_median_nps={statistics.median(base_nps)}\nnonlinear_median_nps={statistics.median(nonlinear_nps)}\n")
        handle.write(f"base_median_time_ms={statistics.median(base_ms)}\nnonlinear_median_time_ms={statistics.median(nonlinear_ms)}\n")
        handle.write(f"nps_ratio_nonlinear_over_base={statistics.median(nonlinear_nps) / statistics.median(base_nps):.4f}\n")
        handle.write(f"time_ratio_nonlinear_over_base={statistics.median(nonlinear_ms) / statistics.median(base_ms):.4f}\n")
    print(f"base_median_nps={statistics.median(base_nps)}")
    print(f"nonlinear_median_nps={statistics.median(nonlinear_nps)}")
    print(f"nps_ratio_nonlinear_over_base={statistics.median(nonlinear_nps) / statistics.median(base_nps):.4f}")
    print(f"base_median_time_ms={statistics.median(base_ms)}")
    print(f"nonlinear_median_time_ms={statistics.median(nonlinear_ms)}")
    print(f"time_ratio_nonlinear_over_base={statistics.median(nonlinear_ms) / statistics.median(base_ms):.4f}")


if __name__ == "__main__":
    main()
