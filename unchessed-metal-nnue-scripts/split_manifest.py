#!/usr/bin/env python3
"""Split JSONL positions by game_id, never by individual position."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=.90)
    ap.add_argument("--valid", type=float, default=.05)
    args = ap.parse_args()
    groups = defaultdict(list)
    for line in Path(args.input).read_text().splitlines():
        if line.strip():
            row = json.loads(line); groups[row["game_id"]].append(row)
    games = list(groups); random.Random(args.seed).shuffle(games)
    n_train = int(len(games) * args.train); n_valid = int(len(games) * args.valid)
    names = {"train": games[:n_train], "valid": games[n_train:n_train+n_valid], "test": games[n_train+n_valid:]}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, ids in names.items():
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for gid in ids:
                for row in groups[gid]: f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"{name}_games={len(ids)} {name}_positions={sum(len(groups[g]) for g in ids)}")

if __name__ == "__main__":
    main()
