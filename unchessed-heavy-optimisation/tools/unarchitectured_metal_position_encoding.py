#!/usr/bin/env python3
"""Shared Unarchitectured Metal position encoding for Python-side evaluation tools.

This mirrors `position_to_input` in `unchessed-core/src/unarchitectured_metal_runtime.rs`
exactly: the mover-perspective vertical flip, the piece-value convention, the
mover-relative castling bits, the en-passant file, the halfmove bucket, and the
`from | to << 6 | promotion << 12` action encoding.

It is deliberately dependency-light (only `python-chess`) so that the
calibration harness and the Rust runtime agree on model input without
duplicating the transform in more than one place.

Self-check (reproduces the frozen start-position parity fixture):

  python3 tools/unarchitectured_metal_position_encoding.py --self-check
"""
from __future__ import annotations

import sys

if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import chess

# Piece indices follow the Rust runtime: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3,
# QUEEN=4, KING=5. Encoded piece values are `index + 1` for the mover and
# `6 + index + 1` for the opponent, leaving 0 for "empty".
_PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Promotion codes in the action encoding.
_PROMOTION_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

NO_EP_FILE = 8
MAX_ACTIONS = 218

POLICY_HUMAN = 0
POLICY_GUIDE = 1


def encode_action(board: chess.Board, move: chess.Move) -> int:
    """Encode one legal move the way the Rust runtime does."""
    flip = board.turn == chess.BLACK
    from_square = move.from_square ^ 56 if flip else move.from_square
    to_square = move.to_square ^ 56 if flip else move.to_square
    promotion = _PROMOTION_CODE[move.promotion]
    return from_square | (to_square << 6) | (promotion << 12)


def encode_position(board: chess.Board) -> dict:
    """Return the mover-perspective model input fields for `board`.

    The returned `legal_moves` list is index-aligned with `legal_actions`, so
    callers can map model outputs back to real moves without re-deriving order.
    """
    flip = board.turn == chess.BLACK
    mover = board.turn
    pieces = [0] * 64
    for square, piece in board.piece_map().items():
        index = square ^ 56 if flip else square
        value = _PIECE_INDEX[piece.piece_type] + 1
        if piece.color != mover:
            value += 6
        pieces[index] = value

    if mover == chess.WHITE:
        mover_king, mover_queen = chess.BB_H1, chess.BB_A1
        opponent_king, opponent_queen = chess.BB_H8, chess.BB_A8
    else:
        mover_king, mover_queen = chess.BB_H8, chess.BB_A8
        opponent_king, opponent_queen = chess.BB_H1, chess.BB_A1

    castling = 0
    rights = board.castling_rights
    if rights & mover_king:
        castling |= 1
    if rights & mover_queen:
        castling |= 2
    if rights & opponent_king:
        castling |= 4
    if rights & opponent_queen:
        castling |= 8

    ep_file = NO_EP_FILE if board.ep_square is None else chess.square_file(board.ep_square)
    halfmove_clock = min(board.halfmove_clock, 255)

    legal_moves = list(board.legal_moves)
    legal_actions = [encode_action(board, move) for move in legal_moves]

    return {
        "pieces": pieces,
        "castling": castling,
        "ep_file": ep_file,
        "halfmove_clock": halfmove_clock,
        "halfmove_bucket": min(halfmove_clock // 8, 15),
        "legal_moves": legal_moves,
        "legal_actions": legal_actions,
    }


def _self_check() -> int:
    """Confirm the encoder reproduces the frozen start-position fixture."""
    board = chess.Board()
    encoded = encode_position(board)

    expected_pieces = [0] * 64
    back = [3 + 1, 1 + 1, 2 + 1, 4 + 1, 5 + 1, 2 + 1, 1 + 1, 3 + 1]
    for file_index in range(8):
        expected_pieces[file_index] = back[file_index]
        expected_pieces[8 + file_index] = 1
        expected_pieces[48 + file_index] = 7
        expected_pieces[56 + file_index] = 6 + back[file_index]

    failures = []
    if encoded["pieces"] != expected_pieces:
        failures.append("start-position piece plane mismatch")
    if encoded["castling"] != 15:
        failures.append(f"castling {encoded['castling']} != 15")
    if encoded["ep_file"] != NO_EP_FILE:
        failures.append(f"ep_file {encoded['ep_file']} != {NO_EP_FILE}")
    if encoded["halfmove_bucket"] != 0:
        failures.append("halfmove bucket != 0")
    if len(encoded["legal_actions"]) != 20:
        failures.append(f"legal action count {len(encoded['legal_actions'])} != 20")

    expected_actions = []
    for file_index in range(8):
        source = 8 + file_index
        expected_actions.append(source | ((source + 8) << 6))
        expected_actions.append(source | ((source + 16) << 6))
    expected_actions += [1 | (16 << 6), 1 | (18 << 6), 6 | (21 << 6), 6 | (23 << 6)]
    if sorted(encoded["legal_actions"]) != sorted(expected_actions):
        failures.append("start-position action set mismatch")

    # The frozen Rust/Python parity fixture reports action 1350 (g1f3) as the
    # best start-position action; confirm that encoding round-trips to that move.
    move = chess.Move.from_uci("g1f3")
    if encode_action(board, move) != 1350:
        failures.append(f"g1f3 encodes to {encode_action(board, move)}, expected 1350")

    # Black-to-move mirror: after 1. e4, black's e7e5 must encode identically to
    # white's e2e4 under the mover-perspective flip.
    mirrored = chess.Board()
    mirrored.push_uci("e2e4")
    white_action = encode_action(chess.Board(), chess.Move.from_uci("e2e4"))
    black_action = encode_action(mirrored, chess.Move.from_uci("e7e5"))
    if white_action != black_action:
        failures.append(f"mover-flip mismatch: {white_action} != {black_action}")

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("OK encoder matches the frozen start-position fixture and mover flip")
    return 0


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        return _self_check()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
