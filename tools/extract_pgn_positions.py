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
    args = parser.parse_args()
    selected = []
    with open(args.pgn, encoding="utf-8", errors="replace") as stream:
        while len(selected) < args.limit:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply >= 16 and ply % 8 == 0 and not board.is_check() and not board.is_game_over():
                    selected.append(board.fen())
                    if len(selected) >= args.limit:
                        break
    with open(args.out, "w", encoding="utf-8") as output:
        output.write("# Legal real-game FEN positions extracted by tools/extract_pgn_positions.py\n")
        output.write("# No engine scores or weights are included in this corpus.\n")
        for fen in selected:
            output.write(fen + "\n")
    print(f"positions={len(selected)}")


if __name__ == "__main__":
    main()
