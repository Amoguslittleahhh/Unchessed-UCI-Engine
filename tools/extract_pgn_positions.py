#!/usr/bin/env python3
from __future__ import annotations

import argparse
import chess
import chess.pgn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn")
    parser.add_argument("out")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--source-tag", default="g", help="prefix for the emitted game_id column, so ids stay unique across multiple PGN files feeding one manifest")
    args = parser.parse_args()
    selected = []
    game_idx = 0
    with open(args.pgn, encoding="utf-8", errors="replace") as stream:
        while len(selected) < args.limit:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            game_idx += 1
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply >= 16 and ply % 8 == 0 and not board.is_check() and not board.is_game_over():
                    # game_id is tab-separated so make_manifest.py's own
                    # --game-field can group positions by real originating
                    # game for genuine game-disjoint train/valid/test splits
                    # (see split_manifest.py) instead of every position
                    # silently getting its own single-position "game".
                    selected.append((board.fen(), f"{args.source_tag}-{game_idx}"))
                    if len(selected) >= args.limit:
                        break
    with open(args.out, "w", encoding="utf-8") as output:
        output.write("# Legal real-game FEN positions extracted by tools/extract_pgn_positions.py\n")
        output.write("# No engine scores or weights are included in this corpus.\n")
        output.write("# Format: FEN\\tgame_id -- pass --game-field 1 to make_manifest.py.\n")
        for fen, game_id in selected:
            output.write(f"{fen}\t{game_id}\n")
    print(f"positions={len(selected)} games={game_idx}")


if __name__ == "__main__":
    main()
