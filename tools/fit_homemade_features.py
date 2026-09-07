#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import chess
import numpy as np

CENTER = sum(1 << chess.parse_square(s) for s in ("d4", "e4", "d5", "e5"))
EXTENDED = sum(1 << chess.parse_square(s) for s in [f"{file}{rank}" for file in "cdef" for rank in "3456"])
FILE_MASKS = [sum(1 << chess.parse_square(f"{file}{rank}") for rank in range(1, 9)) for file in "abcdefgh"]


def structure_score(board: chess.Board, color: chess.Color) -> int:
    pawns = int(board.pieces(chess.PAWN, color).mask)
    score = 0
    for file, mask in enumerate(FILE_MASKS):
        count = (pawns & mask).bit_count()
        if count > 1:
            score -= count - 1
        if count:
            adjacent = 0
            if file:
                adjacent |= FILE_MASKS[file - 1]
            if file < 7:
                adjacent |= FILE_MASKS[file + 1]
            score += 1 if pawns & adjacent else -1
    return score


def feature_vector(fen: str) -> tuple[int, ...]:
    board = chess.Board(fen)
    colors = [chess.WHITE, chess.BLACK]
    occ = [int(board.occupied_co[c]) for c in colors]
    center = [(occ[i] & CENTER).bit_count() for i in range(2)]
    activity = [(occ[i] & EXTENDED).bit_count() for i in range(2)]
    coordination = [0, 0]
    space = [0, 0]
    passed = [0, 0]
    outposts = [0, 0]
    bishop_pair = [0, 0]
    rook_files = [0, 0]
    safety = [0, 0]
    for i, color in enumerate(colors):
        enemy = colors[1 - i]
        own_pawns = int(board.pieces(chess.PAWN, color).mask)
        enemy_pawns = int(board.pieces(chess.PAWN, enemy).mask)
        own_pawn_attacks = 0
        for sq in board.pieces(chess.PAWN, color):
            own_pawn_attacks |= int(chess.BB_PAWN_ATTACKS[color][sq])
        own_knight_attacks = 0
        enemy_pawn_attacks = 0
        for sq in board.pieces(chess.KNIGHT, color):
            own_knight_attacks |= int(chess.BB_KNIGHT_ATTACKS[sq])
        for sq in board.pieces(chess.PAWN, enemy):
            enemy_pawn_attacks |= int(chess.BB_PAWN_ATTACKS[enemy][sq])
        coordination[i] = ((own_pawn_attacks & occ[i]).bit_count() + (own_knight_attacks & occ[i]).bit_count())
        space[i] = (own_pawns & EXTENDED).bit_count() + (own_pawns.bit_count() // 2)
        for sq in board.pieces(chess.PAWN, color):
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            enemy_ahead = 0
            for f in range(max(0, file - 1), min(7, file + 1) + 1):
                mask = FILE_MASKS[f]
                if color == chess.WHITE:
                    ranks = 0 if rank == 7 else mask & (~0 << ((rank + 1) * 8))
                else:
                    ranks = 0 if rank == 0 else mask & ((1 << (rank * 8)) - 1)
                enemy_ahead |= enemy_pawns & ranks
            passed[i] += int(enemy_ahead == 0)
        for sq in board.pieces(chess.KNIGHT, color):
            if (1 << sq) & EXTENDED and not ((1 << sq) & enemy_pawn_attacks):
                outposts[i] += 1
        bishop_pair[i] = int(len(board.pieces(chess.BISHOP, color)) >= 2)
        for sq in board.pieces(chess.ROOK, color):
            if not ((int(board.pieces(chess.PAWN, chess.WHITE).mask) | int(board.pieces(chess.PAWN, chess.BLACK).mask)) & FILE_MASKS[chess.square_file(sq)]):
                rook_files[i] += 1
        king = board.king(color)
        safety[i] = (int(chess.BB_KING_ATTACKS[king]) & own_pawns).bit_count()
    phase = min(max(board.occupied.bit_count() - 2, 0), 30)
    return (
        center[0] - center[1],
        structure_score(board, chess.WHITE) - structure_score(board, chess.BLACK),
        safety[0] - safety[1],
        activity[0] - activity[1],
        coordination[0] - coordination[1],
        space[0] - space[1],
        passed[0] - passed[1],
        outposts[0] - outposts[1],
        bishop_pair[0] - bishop_pair[1],
        rook_files[0] - rook_files[1],
        1 if board.turn == chess.WHITE else -1,
        phase,
    )


def current_residual(fen: str) -> int:
    *features, phase = feature_vector(fen)
    raw = (5 * features[0] + 4 * features[1] + 3 * features[2] + 2 * features[3] +
           3 * features[4] + 2 * features[5] + 6 * features[6] + 4 * features[7] +
           10 * features[8] + 3 * features[9] + features[10])
    return max(-180, min(180, raw * (8 + phase) // 16))


def rows(path: str):
    pattern = re.compile(r"\| `([^`]+)` \| (-?\d+) \| (-?\d+) \| ([+-]?\d+) \|")
    for line in Path(path).read_text().splitlines():
        match = pattern.match(line)
        if match:
            fen, sf, homemade, _ = match.groups()
            residual = current_residual(fen)
            board = chess.Board(fen)
            hce = int(homemade) - (residual if board.turn == chess.WHITE else -residual)
            yield fen, int(sf), hce, feature_vector(fen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = list(rows(args.comparison))
    x = []
    y = []
    for _, sf, hce, features in data:
        *signals, phase = features
        phase_norm = phase / 30.0
        x.append([hce] + signals + [v * phase_norm for v in signals])
        y.append(sf)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    train = np.arange(len(data)) % 5 != 0
    test = ~train
    weights, *_ = np.linalg.lstsq(x[train], y[train], rcond=None)
    pred = x @ weights
    baseline = x[:, 0]
    train_mae = np.mean(np.abs(pred[train] - y[train]))
    test_mae = np.mean(np.abs(pred[test] - y[test]))
    baseline_test_mae = np.mean(np.abs(baseline[test] - y[test]))
    with Path(args.out).open("w") as out:
        out.write("# Homemade multi-breakthrough fit against black-box SFNNv16 reference\n\n")
        out.write("Only UCI score observations and original Unchessed bitboard signals are used. No Stockfish source, weights, feature rows, or formulas are imported. Positions with index divisible by five are held out.\n\n")
        out.write(f"positions={len(data)}\ntrain={train.sum()}\ntest={test.sum()}\n")
        out.write(f"baseline_test_mae_cp={baseline_test_mae:.3f}\nfit_train_mae_cp={train_mae:.3f}\nfit_test_mae_cp={test_mae:.3f}\n\n")
        names = ["hce", "center", "pawn_structure", "king_safety", "activity", "coordination", "space", "passed_pawns", "outposts", "bishop_pair", "rook_files", "tempo", "center_phase", "pawn_phase", "king_phase", "activity_phase", "coordination_phase", "space_phase", "passed_phase", "outposts_phase", "bishop_pair_phase", "rook_files_phase", "tempo_phase"]
        out.write("| Feature | Coefficient |\n|---|---:|\n")
        for name, value in zip(names, weights):
            out.write(f"| {name} | {value:.9f} |\n")
    print(f"positions={len(data)}")
    print(f"baseline_test_mae_cp={baseline_test_mae:.3f}")
    print(f"fit_train_mae_cp={train_mae:.3f}")
    print(f"fit_test_mae_cp={test_mae:.3f}")
    print("weights=" + ",".join(f"{v:.9f}" for v in weights))


if __name__ == "__main__":
    main()
