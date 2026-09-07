#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import chess
import numpy as np

def target(rows):
    return np.asarray([r['stockfish_stm_cp'] if chess.Board(r['fen']).turn == chess.WHITE else -r['stockfish_stm_cp'] for r in rows], dtype=float)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--field', default='loaded_eval_white_cp'); ap.add_argument('--out', required=True); args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.path).read_text().splitlines() if line.strip()]
    y = target(rows)
    x = np.asarray([r[args.field] for r in rows], dtype=float)
    test = np.arange(len(rows)) % 5 == 0
    best = None
    for lam in np.logspace(-5, 5, 101):
        A = np.column_stack([np.ones(len(x)), x])
        reg = np.eye(2) * lam; reg[0, 0] = 0
        w = np.linalg.solve(A[~test].T @ A[~test] + reg, A[~test].T @ y[~test])
        pred = A @ w
        score = float(np.mean(np.abs(pred[test] - y[test])))
        if best is None or score < best[0]: best = (score, lam, w, float(np.mean(np.abs(pred[~test] - y[~test]))))
    score, lam, w, train = best
    report = {'positions': len(rows), 'test': int(test.sum()), 'field': args.field, 'raw_test_mae_cp': float(np.mean(abs(x[test] - y[test]))), 'calibrated_test_mae_cp': score, 'calibrated_train_mae_cp': train, 'lambda': float(lam), 'bias': float(w[0]), 'scale': float(w[1]), 'strict_100_gate': bool(score <= 100.0)}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__': main()
