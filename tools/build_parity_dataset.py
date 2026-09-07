#!/usr/bin/env python3
"""Build a leakage-safe real-game corpus for homemade evaluation research.

Games, not positions, are assigned to splits. The output contains only legal
FENs and provenance metadata; no Stockfish code, weights, or labels are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import chess
import chess.pgn


def split_for(key: str) -> str:
    value = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=2).digest(), "big") % 100
    return "train" if value < 80 else "valid" if value < 90 else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("unchessed-eval-bar/data/parity_manifest.jsonl"))
    ap.add_argument("--max-positions", type=int, default=300_000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--max-ply", type=int, default=180)
    args = ap.parse_args()

    candidates = [
        args.root / "data/training/lichess-2022-10-05/elo-2300-2600.pgn",
        args.root / "data/training/lichess-2022-10-05/elo-2000-2300.pgn",
        args.root / "data/training/twic/twic1649.pgn",
        args.root / "data/archive/1946-1970/nostalgia-1961-1970.pgn",
        args.root / "data/archive/1946-1970/nostalgia-1941-1960.pgn",
        args.root / "data/archive/1900-1945/nostalgia-1921-1940.pgn",
    ]
    sources = [p for p in candidates if p.exists()]
    if not sources:
        raise SystemExit("no configured real PGN archives were found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "valid": 0, "test": 0}
    games = {"train": 0, "valid": 0, "test": 0}
    total = 0
    with args.out.open("w", encoding="utf-8") as out:
        for source in sources:
            with source.open(encoding="utf-8", errors="replace") as stream:
                game_index = 0
                while total < args.max_positions:
                    game = chess.pgn.read_game(stream)
                    if game is None:
                        break
                    game_key = f"{source.relative_to(args.root)}:{game_index}"
                    split = split_for(game_key)
                    games[split] += 1
                    board = game.board()
                    for ply, move in enumerate(game.mainline_moves(), start=1):
                        board.push(move)
                        if ply < args.min_ply or ply > args.max_ply or ply % args.stride:
                            continue
                        if board.is_check() or board.is_game_over():
                            continue
                        if total >= args.max_positions:
                            break
                        record = {
                            "split": split,
                            "source": str(source.relative_to(args.root)),
                            "game": game_index,
                            "ply": ply,
                            "fen": board.fen(),
                        }
                        out.write(json.dumps(record, separators=(",", ":")) + "\n")
                        counts[split] += 1
                        total += 1
                    game_index += 1
                if total >= args.max_positions:
                    break
    summary = {"sources": [str(p.relative_to(args.root)) for p in sources], "positions": counts, "games": games, "total": total, "split_rule": "blake2b(game provenance) modulo 100: 0-79 train, 80-89 valid, 90-99 test"}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
