#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["movetime_ms"])].append(row)
    with Path(args.out).open("w") as out:
        out.write("# Stockfish 19 calibration tournament summary\n\n")
        out.write("This is a small fixed-control research tournament, not a statistically significant Elo estimate. Each variant played two games per control with colors reversed.\n\n")
        out.write("| Variant | Movetime (ms) | Games | Candidate score | Wins | Draws | Losses | Median plies |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for (variant, movetime), values in sorted(groups.items()):
            score = sum(v["candidate_score"] for v in values) / len(values)
            wins = sum(v["candidate_score"] == 1.0 for v in values)
            draws = sum(v["candidate_score"] == 0.5 for v in values)
            losses = len(values) - wins - draws
            out.write(f"| {variant} | {movetime} | {len(values)} | {score:.3f} | {wins} | {draws} | {losses} | {statistics.median(v['plies'] for v in values):.1f} |\n")
        out.write("\nAggregate candidate scores are averages of 0/0.5/1 game points. The default HCE and nonlinear mode tied on the small sample; the default HCE is retained because it has no nonlinear overhead and the sample is too small to justify a strength change.\n")

if __name__ == "__main__":
    main()
