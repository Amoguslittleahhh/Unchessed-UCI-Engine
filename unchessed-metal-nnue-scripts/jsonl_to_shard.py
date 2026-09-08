#!/usr/bin/env python3
"""Convert label_uci.py JSONL output into the 104-byte binary shard format
tools/train_nnue.py expects (see its own module docstring for the exact
record layout, mirrored here):

  bytes 0-95   12 x u64 LE bitboards, side-to-move normalized (planes 0-5 =
               mover P,N,B,R,Q,K; 6-11 = opponent; board vertically flipped
               (sq -> sq^56) when Black is to move, matching this project's
               existing mover-up-frame convention used throughout nnue.rs
               and unchessed-datagen)
  96-97        i16 LE search score in centipawns, from the STM perspective
  98           u8 WDL from the STM perspective (2=win, 1=draw, 0=loss)
  99-103       padding (0)

Records with score_cp is None (timed_out, terminal, or label_failed) are
skipped -- train_nnue.py has no encoding for "no ordinary score", and
silently writing a fabricated score for those would corrupt training the
same way the earlier evaluate_gates.py bug corrupted an evaluation gate.
"""
from __future__ import annotations
import argparse, json, struct, sys
from pathlib import Path

import chess


def bitboards_stm_normalized(board: chess.Board) -> list[int]:
    """12 x u64, planes 0-5 = mover P,N,B,R,Q,K, 6-11 = opponent,
    vertically flipped (sq -> sq^56) when Black is to move."""
    flip = board.turn == chess.BLACK
    mover = board.turn
    opponent = not mover
    bbs = [0] * 12
    piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for plane, (color, piece_type) in enumerate(
        [(mover, pt) for pt in piece_order] + [(opponent, pt) for pt in piece_order]
    ):
        bb = 0
        for sq in board.pieces(piece_type, color):
            out_sq = sq ^ 56 if flip else sq
            bb |= 1 << out_sq
        bbs[plane] = bb
    return bbs


def wdl_byte(wdl_stm) -> int | None:
    if not wdl_stm or len(wdl_stm) != 3:
        return None
    w, d, l = wdl_stm
    best = max(range(3), key=lambda i: (w, d, l)[i])
    return {0: 2, 1: 1, 2: 0}[best]  # index 0=win -> byte 2, 1=draw -> 1, 2=loss -> 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+", help="one or more label_uci.py JSONL outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    written = 0
    skipped_no_score = 0
    skipped_bad_fen = 0
    skipped_no_wdl = 0

    with open(args.out, "wb") as out:
        for path in args.labels:
            for raw in Path(path).read_text().splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("label_failed") or row.get("timed_out") or row.get("terminal"):
                    skipped_no_score += 1
                    continue
                cp = row.get("score_cp")
                if cp is None:
                    skipped_no_score += 1
                    continue
                if not (-32768 <= cp <= 32767):
                    skipped_no_score += 1
                    continue
                try:
                    board = chess.Board(row["fen"])
                except Exception:
                    skipped_bad_fen += 1
                    continue
                wb = wdl_byte(row.get("wdl"))
                if wb is None:
                    skipped_no_wdl += 1
                    continue
                bbs = bitboards_stm_normalized(board)
                out.write(struct.pack("<12QhB5x", *bbs, cp, wb))
                written += 1

    print(
        f"written={written} skipped_no_score={skipped_no_score} "
        f"skipped_bad_fen={skipped_bad_fen} skipped_no_wdl={skipped_no_wdl}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
