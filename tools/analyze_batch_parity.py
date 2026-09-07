#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import chess
import numpy as np
from fit_homemade_features import feature_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    y = np.asarray([r["stockfish_stm_cp"] if chess.Board(r["fen"]).turn == chess.WHITE else -r["stockfish_stm_cp"] for r in rows], dtype=float)
    base = np.asarray([r["unchessed_white_cp"] for r in rows], dtype=float)
    x = []
    for r in rows:
        f = feature_vector(r["fen"])
        *signals, phase = f
        side = 1.0 if chess.Board(r["fen"]).turn == chess.WHITE else -1.0
        z = [1.0, base[len(x)] / 1000.0, phase / 30.0]
        z.extend(float(v) / 10.0 for v in signals)
        for idx in (1, 2, 6, 8, 9):
            z.append(float(signals[idx]) * (phase / 30.0) / 10.0)
        z.append((signals[1] * phase) / 300.0)
        z.append((signals[2] * phase) / 300.0)
        z.append((signals[6] * phase) / 300.0)
        z.append(side * phase / 30.0)
        x.append(z)
    x = np.asarray(x, dtype=float)
    test = np.arange(len(rows)) % 5 == 0
    best = None
    for lam in np.logspace(-3, 5, 65):
        reg = np.eye(x.shape[1]) * lam
        reg[0, 0] = 0
        w = np.linalg.solve(x[~test].T @ x[~test] + reg, x[~test].T @ y[~test])
        pred = x @ w
        score = float(np.mean(np.abs(pred[test] - y[test])))
        if best is None or score < best[0]:
            best = (score, lam, w, float(np.mean(np.abs(pred[~test] - y[~test]))))
    score, lam, w, train_mae = best
    baseline = float(np.mean(np.abs(base[test] - y[test])))
    all_baseline = float(np.mean(np.abs(base - y)))
    all_model = float(np.mean(np.abs((x @ w) - y)));
    report = {
        "positions": len(rows), "test": int(test.sum()), "train": int((~test).sum()),
        "baseline_test_mae_cp": baseline, "baseline_all_mae_cp": all_baseline,
        "model_test_mae_cp": score, "model_train_mae_cp": train_mae,
        "model_all_mae_cp": all_model, "lambda": float(lam),
        "weights": [float(v) for v in w],
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
