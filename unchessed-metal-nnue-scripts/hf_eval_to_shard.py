#!/usr/bin/env python3
"""Convert the real Lichess/chess-position-evaluations dataset
(huggingface.co, CC0, 394.7M positions/957.8M rows, real Stockfish evals
computed by Lichess's own analysis board) directly into the 104-byte
binary shard format tools/train_nnue.py expects -- no Stockfish
relabeling needed, since this dataset already carries real engine evals.

Record layout (matches jsonl_to_shard.py exactly):
  bytes 0-95   12 x u64 LE bitboards, side-to-move normalized (planes 0-5 =
               mover P,N,B,R,Q,K; 6-11 = opponent; board vertically flipped
               (sq -> sq^56) when Black is to move)
  96-97        i16 LE search score in centipawns, from the STM perspective
  98           u8 WDL from the STM perspective (2=win, 1=draw, 0=loss) --
               approximated from cp via a standard win-probability curve,
               since this dataset has no WDL field
  99-103       padding (0)

IMPORTANT: this dataset's cp/mate fields are documented (and empirically
verified against known-good/known-bad opening moves) as being from
WHITE's point of view always, unlike the UCI/STM-relative convention
used elsewhere in this project's own label_uci.py pipeline. Every score
here is explicitly converted: STM-relative = cp if White to move else -cp.

Mate scores use the same graduated-magnitude convention as
evaluate_gates.py's mate_signed_cp() (sign * (30000 - min(plies, 100))),
NOT a flat +-30000. A first run (hf-eval-284m, SPRT-rejected at -178 Elo
vs the shipped net) used a flat +-30000 for every mate regardless of
distance; real post-training eval through the actual Rust inference path
showed the ~14% mate-labeled slice hitting 28,391.7cp MAE vs 175.6cp on
ordinary positions, i.e. those positions carried no usable distance
signal at all. Graduating the magnitude by ply-distance at least gives
the trainer something to fit even though sigmoid(cp/400) is already
saturated well below 29900cp either way.
"""
from __future__ import annotations
import argparse, glob, hashlib, math, struct, sys

import chess
import pyarrow.parquet as pq


def split_for(fen: str, valid_frac: float, test_frac: float) -> str:
    """Deterministic position-level split by hashing the FEN. This dataset
    has no originating-game id (it's aggregated per-position analyses, not
    sequential game moves), so the leakage risk to guard against is the
    same exact position landing in two splits, not same-game contamination
    -- a stable per-FEN hash split handles that in a single streaming pass."""
    h = int(hashlib.sha256(fen.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < valid_frac:
        return "valid"
    if h < valid_frac + test_frac:
        return "test"
    return "train"


def bitboards_stm_normalized(board: chess.Board) -> list[int]:
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


def cp_to_wdl_byte(cp_stm: int) -> int:
    """Approximate WDL from a centipawn score via a standard logistic
    win-probability curve (matches the shape Stockfish's own WDL model
    uses), since this dataset has no WDL field of its own."""
    win_prob = 1.0 / (1.0 + math.exp(-cp_stm / 400.0))
    if win_prob > 0.55:
        return 2
    if win_prob < 0.45:
        return 0
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-glob", required=True, nargs="+",
                     help="one or more globs matching the downloaded data_*.parquet files")
    ap.add_argument("--out-prefix", required=True,
                     help="writes <prefix>-train.bin, <prefix>-valid.bin, <prefix>-test.bin")
    ap.add_argument("--limit", type=int, default=None,
                     help="stop after writing this many TRAIN records (default: all)")
    ap.add_argument("--min-depth", type=int, default=20,
                     help="skip evals below this search depth (shallow "
                          "browser-triggered analyses are noisier)")
    ap.add_argument("--valid-frac", type=float, default=0.02)
    ap.add_argument("--test-frac", type=float, default=0.02)
    args = ap.parse_args()

    files = sorted({f for pattern in args.parquet_glob for f in glob.glob(pattern)})
    if not files:
        print(f"no files matched {args.parquet_glob}", file=sys.stderr)
        sys.exit(1)

    written = {"train": 0, "valid": 0, "test": 0}
    skipped_shallow = 0
    skipped_bad_fen = 0
    skipped_overflow = 0

    outs = {
        split: open(f"{args.out_prefix}-{split}.bin", "wb")
        for split in ("train", "valid", "test")
    }
    with outs["train"], outs["valid"], outs["test"]:
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(
                batch_size=200_000, columns=["fen", "cp", "mate", "depth"]
            ):
                df = batch.to_pandas()
                for row in df.itertuples(index=False):
                    if args.limit is not None and written["train"] >= args.limit:
                        print(f"written={written} skipped_shallow={skipped_shallow} "
                              f"skipped_bad_fen={skipped_bad_fen} "
                              f"skipped_overflow={skipped_overflow}", file=sys.stderr)
                        return
                    if row.depth < args.min_depth:
                        skipped_shallow += 1
                        continue
                    try:
                        board = chess.Board(row.fen)
                    except Exception:
                        skipped_bad_fen += 1
                        continue
                    white_to_move = board.turn == chess.WHITE
                    if row.mate is not None and not (isinstance(row.mate, float) and math.isnan(row.mate)):
                        mate_white = int(row.mate)
                        mate_plies_stm = mate_white if white_to_move else -mate_white
                        sign = 1 if mate_plies_stm > 0 else -1
                        cp_stm = sign * (30000 - min(abs(mate_plies_stm), 100))
                    elif row.cp is not None and not (isinstance(row.cp, float) and math.isnan(row.cp)):
                        cp_white = int(row.cp)
                        cp_stm = cp_white if white_to_move else -cp_white
                    else:
                        skipped_bad_fen += 1
                        continue
                    if not (-32768 <= cp_stm <= 32767):
                        skipped_overflow += 1
                        continue
                    split = split_for(row.fen, args.valid_frac, args.test_frac)
                    bbs = bitboards_stm_normalized(board)
                    wb = cp_to_wdl_byte(cp_stm)
                    outs[split].write(struct.pack("<12QhB5x", *bbs, cp_stm, wb))
                    written[split] += 1

    print(f"written={written} skipped_shallow={skipped_shallow} "
          f"skipped_bad_fen={skipped_bad_fen} skipped_overflow={skipped_overflow}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
