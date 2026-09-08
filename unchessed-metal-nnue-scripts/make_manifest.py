#!/usr/bin/env python3
"""Create a JSONL position manifest from FEN/EPD-like lines."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def fen(line: str) -> str:
    line = line.split(" bm ", 1)[0].strip()
    fields = line.split()
    if len(fields) == 4:
        fields += ["0", "1"]
    if len(fields) != 6:
        raise ValueError(f"expected FEN or 4-field EPD: {line}")
    return " ".join(fields)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--game-field", default="none", help="column containing game id, or none")
    args = ap.parse_args()
    game_col = None if args.game_field == "none" else int(args.game_field)
    rows = []
    for n, raw in enumerate(Path(args.input).read_text().splitlines()):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        board = fen(parts[0])
        game_id = parts[game_col] if game_col is not None and len(parts) > game_col else f"line-{n}"
        pid = hashlib.sha256(f"{args.source}\0{game_id}\0{board}".encode()).hexdigest()[:24]
        rows.append({"position_id": pid, "game_id": game_id, "fen": board, "source": args.source, "line": n})
    with Path(args.out).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"positions={len(rows)} source={args.source} out={args.out}")

if __name__ == "__main__":
    main()
