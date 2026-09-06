#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarize-dispatch.py DISPATCH_CSV SCALAR_CSV OUTPUT_MD")
    dispatch = read(Path(sys.argv[1]))
    scalar = read(Path(sys.argv[2]))
    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"dispatch": [], "scalar": []})
    for row in dispatch:
        by_key[(row["hash_mb"], row["position"])]["dispatch"].append(float(row["nps"]))
    for row in scalar:
        by_key[(row["hash_mb"], row["position"])]["scalar"].append(float(row["nps"]))
    lines = ["# Dispatch benchmark summary", "", "| Hash MiB | Position | Scalar median NPS | Dispatch median NPS | Speedup |", "|---:|---|---:|---:|---:|"]
    dispatch_all: list[float] = []
    scalar_all: list[float] = []
    for (hash_mb, position), values in sorted(by_key.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        scalar_median = statistics.median(values["scalar"])
        dispatch_median = statistics.median(values["dispatch"])
        speedup = (dispatch_median / scalar_median - 1.0) * 100.0
        dispatch_all.extend(values["dispatch"])
        scalar_all.extend(values["scalar"])
        lines.append(f"| {hash_mb} | {position} | {scalar_median:,.0f} | {dispatch_median:,.0f} | {speedup:+.2f}% |")
    scalar_median = statistics.median(scalar_all)
    dispatch_median = statistics.median(dispatch_all)
    speedup = (dispatch_median / scalar_median - 1.0) * 100.0
    lines += ["", f"Aggregate median across all rows: scalar **{scalar_median:,.0f} NPS**, dispatch **{dispatch_median:,.0f} NPS**, speedup **{speedup:+.2f}%**.", ""]
    Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
