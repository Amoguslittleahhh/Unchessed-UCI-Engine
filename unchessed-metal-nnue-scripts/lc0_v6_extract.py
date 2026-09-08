#!/usr/bin/env python3
"""Extract legal FEN positions from real LCZero (LC0) V6 self-play training
chunks (https://storage.lczero.org/files/training_data/), for feeding into
the same make_manifest.py -> label_uci.py (Stockfish) -> jsonl_to_shard.py
pipeline already used for Lichess-sourced positions.

V6 record layout and field semantics are ported directly from the official
LC0 sources (src/trainingdata/trainingdata_v6.h, src/neural/decoder.cc,
src/neural/encoder.cc) and the reference Python decoder
(lczero-training/tf/chunkparser.py) -- not reverse-engineered from scratch.
Only input_format == 1 (INPUT_CLASSICAL_112_PLANE) is handled; other/older
formats (e.g. V5 chunks from older runs) are skipped rather than guessed at,
since getting this silently wrong would corrupt every extracted position the
same way the earlier evaluate_gates.py mate-score bug corrupted a gate.

No engine scores, policy targets, or LC0 weights are extracted or emitted --
only the legal board position (FEN) and rule50, exactly mirroring what
tools/extract_pgn_positions.py exports for real Lichess games.
"""
from __future__ import annotations
import argparse, glob, gzip, os, struct, sys

import chess

V6_SIZE = 8356
V6_FMT = "<II7432s832sBBBBBBBBfffffffffffffffIHHQ"
# version, input_format, probs(7432s), planes(832s), us_ooo, us_oo,
# them_ooo, them_oo, stm, rule50_count, invariance_info, dep_result,
# root_q, best_q, root_d, best_d, root_m, best_m, plies_left, result_q,
# result_d, played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
# played_idx, best_idx, reserved(policy_kld+q_st as 8 raw bytes)


def mirror_bb(bb: int) -> int:
    """Reverse byte order of a 64-bit int == BitBoard::Mirror() in lc0
    (vertical flip, rank r -> 7-r), equivalent to sq -> sq^56 per bit."""
    b = bb.to_bytes(8, "little")
    return int.from_bytes(b[::-1], "little")


# LC0's plane order is P, N, B, R, K, Q (king before queen) -- verified
# empirically against a real game-start record (e1/d1 landed in planes
# 4/5 respectively), not the naive P,N,B,R,Q,K guess.
PLANE_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
                      chess.KING, chess.QUEEN]


def record_to_fen(rec: bytes) -> tuple[str, int] | None:
    """Returns (fen, rule50) for a real V6 record, or None if unsupported
    (non-classical input_format) or structurally invalid."""
    (version, input_format, probs, planes_raw, us_ooo, us_oo, them_ooo,
     them_oo, stm, rule50_count, invariance_info, dep_result, root_q,
     best_q, root_d, best_d, root_m, best_m, plies_left, result_q,
     result_d, played_q, played_d, played_m, orig_q, orig_d, orig_m,
     visits, played_idx, best_idx, reserved) = struct.unpack(V6_FMT, rec)

    if version != 6 or input_format != 1:
        return None

    planes = struct.unpack("<104Q", planes_raw)
    mover = planes[0:6]
    opponent = planes[6:12]
    black_to_move = stm == 1

    if black_to_move:
        black_bbs = [mirror_bb(b) for b in mover]
        white_bbs = [mirror_bb(b) for b in opponent]
    else:
        white_bbs = list(mover)
        black_bbs = list(opponent)

    board = chess.Board(None)
    board.turn = chess.BLACK if black_to_move else chess.WHITE
    for i, pt in enumerate(PLANE_PIECE_TYPES):
        bb = white_bbs[i]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
            bb &= bb - 1
        bb = black_bbs[i]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
            bb &= bb - 1

    # us/them castling booleans -> absolute white/black per side-to-move.
    # Real self-play chunks have been observed with a right still flagged
    # after the corresponding king/rook has already left its home square
    # (e.g. a king shuffle move, rights lost, flag stays stale for the
    # position) -- gate each claimed right on the actual piece placement so
    # we never hand Stockfish a geometrically impossible castling claim.
    if black_to_move:
        w_oo, w_ooo, b_oo, b_ooo = them_oo, them_ooo, us_oo, us_ooo
    else:
        w_oo, w_ooo, b_oo, b_ooo = us_oo, us_ooo, them_oo, them_ooo
    rights = ""
    if w_oo and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) \
            and board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
        rights += "K"
    if w_ooo and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) \
            and board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
        rights += "Q"
    if b_oo and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) \
            and board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
        rights += "k"
    if b_ooo and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) \
            and board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
        rights += "q"
    board.set_castling_fen(rights if rights else "-")
    board.halfmove_clock = rule50_count
    board.fullmove_number = 1
    board.ep_square = None

    try:
        if not board.is_valid():
            return None
    except Exception:
        return None
    return board.fen(), rule50_count


def iter_records(gz_path: str):
    try:
        data = gzip.open(gz_path, "rb").read()
    except Exception:
        return
    if len(data) % V6_SIZE != 0:
        return
    for off in range(0, len(data), V6_SIZE):
        yield data[off:off + V6_SIZE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", nargs="+", required=True,
                     help="one or more extracted-tar directories, each "
                          "containing per-game training.<id>.gz chunks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1_000_000)
    ap.add_argument("--ply-stride", type=int, default=8,
                     help="sample every Nth ply within a game, mirroring "
                          "extract_pgn_positions.py's stride")
    ap.add_argument("--source-tag", default="lc0")
    args = ap.parse_args()

    selected = []
    n_games = 0
    n_skipped_format = 0
    n_skipped_invalid = 0

    for d in args.input_dirs:
        if len(selected) >= args.limit:
            break
        for gz_path in sorted(glob.glob(os.path.join(d, "*.gz"))):
            if len(selected) >= args.limit:
                break
            n_games += 1
            game_id = f"{args.source_tag}-{os.path.basename(d)}-{n_games}"
            recs = list(iter_records(gz_path))
            for ply, rec in enumerate(recs):
                if ply < 16 or ply % args.ply_stride != 0:
                    continue
                result = record_to_fen(rec)
                if result is None:
                    if len(rec) == V6_SIZE:
                        n_skipped_invalid += 1
                    else:
                        n_skipped_format += 1
                    continue
                fen, _ = result
                selected.append((fen, game_id))
                if len(selected) >= args.limit:
                    break

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# Legal real-game FEN positions extracted from real LC0 "
                   "V6 self-play training data by lc0_v6_extract.py\n")
        out.write("# No LC0 weights, policy targets, or engine scores are "
                   "included in this corpus.\n")
        out.write("# Format: FEN\\tgame_id -- pass --game-field 1 to "
                   "make_manifest.py.\n")
        for fen, game_id in selected:
            out.write(f"{fen}\t{game_id}\n")

    print(f"positions={len(selected)} games_seen={n_games} "
          f"skipped_invalid={n_skipped_invalid} "
          f"skipped_unsupported_format={n_skipped_format}", file=sys.stderr)


if __name__ == "__main__":
    main()
