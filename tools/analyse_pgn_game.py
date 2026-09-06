#!/usr/bin/env python3
"""Move-by-move engine evaluation trace for a single PGN game.

Drives any UCI engine (Stockfish or otherwise) via python-chess to print
the position eval (White's POV, centipawns; mate scores clamped to
+/-100000) before each move, and optionally flags moves whose eval swing
exceeds a threshold as candidate mistakes.

Large swings late in an already-decided position (deep king/pawn races,
forced mates) are typically engine-eval noise rather than real additional
errors -- read the full trace, not just the flagged list, before calling
something a blunder. Used to find the actual turning point in
2026.09.06_Unchessed Game Adapter - RubiChess.pgn (main lost 0-1): not the
tactic around move 22 that looked significant on a manual read, but 9.a4,
verified by tracing the full eval curve rather than trusting a single
move-by-move loss threshold.

Usage:
    python analyse_pgn_game.py game.pgn /path/to/stockfish [--depth 18]
        [--from PLY] [--to PLY] [--min-loss CP] [--trace]

--trace prints every move's eval instead of only flagged losses -- use it
to sanity-check a flagged spike against the surrounding trend, or to find
a gradual drift a single-move threshold would miss.
"""
from __future__ import annotations

import argparse
import chess
import chess.engine
import chess.pgn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pgn_path")
    parser.add_argument("engine_path")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--from", dest="ply_from", type=int, default=1)
    parser.add_argument("--to", dest="ply_to", type=int, default=10_000)
    parser.add_argument("--min-loss", type=int, default=100, help="cp threshold for flagging a move (ignored with --trace)")
    parser.add_argument("--trace", action="store_true", help="print every move's eval instead of only flagged losses")
    args = parser.parse_args()

    with open(args.pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        raise SystemExit(f"no game found in {args.pgn_path}")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    engine.configure({"Threads": 1, "Hash": 64})

    board = game.board()
    rows: list[tuple[int, str, str, int | None]] = []
    try:
        for move in game.mainline_moves():
            ply = board.fullmove_number
            mover = "White" if board.turn == chess.WHITE else "Black"
            score = None
            if args.ply_from <= ply <= args.ply_to:
                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                score = info["score"].white().score(mate_score=100_000)
            san = board.san(move)
            board.push(move)
            rows.append((ply, mover, san, score))
    finally:
        engine.quit()

    if args.trace:
        for ply, mover, san, score in rows:
            if score is None:
                continue
            sep = "." if mover == "White" else "..."
            print(f"before {ply}{sep}{san}: eval(white pov)={score}")
        return 0

    for i in range(len(rows) - 1):
        ply, mover, san, score_before = rows[i]
        _, _, _, score_after = rows[i + 1]
        if score_before is None or score_after is None:
            continue
        loss = (score_before - score_after) if mover == "White" else (score_after - score_before)
        if loss >= args.min_loss:
            sep = "." if mover == "White" else "..."
            print(f"move {ply}{sep} {san} ({mover}): eval {score_before} -> {score_after}, loss {loss}cp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
