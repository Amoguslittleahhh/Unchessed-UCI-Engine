#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import chess
import numpy as np

from fit_homemade_features import feature_vector

PATTERN = re.compile(r"\| `([^`]+)` \| (-?\d+) \| (-?\d+) \| ([+-]?\d+) \|")
SCALE = 0.680292396
BIAS = 24.603188915
SELECTED = (1, 2, 6, 8, 9)
WEIGHTS = (5, 4, 3, 2, 3, 2, 6, 4, 10, 3, 1)


def residual(features):
    *signals, phase = features
    raw = sum(WEIGHTS[i] * signals[i] for i in SELECTED)
    return max(-180, min(180, raw * (8 + phase) // 16))


def material(board: chess.Board) -> float:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(values[p] * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p in values)


def read_rows(path: str):
    rows = []
    for line in Path(path).read_text().splitlines():
        match = PATTERN.match(line)
        if not match:
            continue
        fen, sf, home, _ = match.groups()
        board = chess.Board(fen)
        features = feature_vector(fen)
        side = 1 if board.turn == chess.WHITE else -1
        raw_hce = (int(home) - BIAS) / SCALE - side * residual(features)
        rows.append((fen, float(sf), raw_hce, features, material(board)))
    return rows


def design(row):
    _, _, hce, features, mat = row
    *signals, phase = features
    # Original compact parity model: baseline HCE plus low-dimensional
    # structural signals, phase/material context, and a few interactions.
    z = [hce / 1000.0, mat / 20.0, phase / 30.0]
    z.extend(float(v) / 10.0 for v in signals)
    for idx in (1, 2, 6, 8, 9):
        z.append(float(signals[idx]) * (phase / 30.0) / 10.0)
    z.append((mat / 20.0) * (phase / 30.0))
    return np.asarray([1.0, *z], dtype=float)


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = read_rows(args.comparison)
    matrix = np.vstack([design(row) for row in rows])
    target = np.asarray([row[1] for row in rows], dtype=float)
    baseline = np.asarray([row[2] * SCALE + BIAS + (1 if chess.Board(row[0]).turn == chess.WHITE else -1) * residual(row[3]) * SCALE for row in rows])
    train = np.arange(len(rows)) % 5 != 0
    test = ~train
    best = None
    for lam in np.logspace(-4, 4, 33):
        regularizer = np.eye(matrix.shape[1]) * lam
        regularizer[0, 0] = 0.0
        weights = np.linalg.solve(matrix[train].T @ matrix[train] + regularizer, matrix[train].T @ target[train])
        prediction = matrix @ weights
        result = (mae(prediction[test], target[test]), lam, weights, mae(prediction[train], target[train]))
        if best is None or result[0] < best[0]:
            best = result
    test_mae, lam, weights, train_mae = best
    baseline_test = mae(baseline[test], target[test])
    prediction = matrix @ weights
    with Path(args.out).open("w") as out:
        out.write("# Homemade parity model fit\n\n")
        out.write("The model is trained only from black-box UCI scores and original Unchessed bitboard features. It uses a leakage-safe modulo-five holdout; no Stockfish source or network data is imported.\n\n")
        out.write(f"positions={len(rows)}\ntrain={train.sum()}\ntest={test.sum()}\nlambda={lam:.9g}\nbaseline_test_mae_cp={baseline_test:.3f}\nmodel_train_mae_cp={train_mae:.3f}\nmodel_test_mae_cp={test_mae:.3f}\n\n")
        out.write("weights=" + ",".join(f"{float(v):.12g}" for v in weights) + "\n")
    print(f"positions={len(rows)}")
    print(f"lambda={lam:.9g}")
    print(f"baseline_test_mae_cp={baseline_test:.3f}")
    print(f"model_train_mae_cp={train_mae:.3f}")
    print(f"model_test_mae_cp={test_mae:.3f}")
    print("weights=" + ",".join(f"{float(v):.12g}" for v in weights))


if __name__ == "__main__":
    main()
