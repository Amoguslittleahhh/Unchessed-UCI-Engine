#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
import chess
from fit_homemade_features import feature_vector

WEIGHTS = (5, 4, 3, 2, 3, 2, 6, 4, 10, 3, 1)
GROUPS = {
    "base": (),
    "center_pressure": (0,),
    "pawn_geometry": (1, 6),
    "king_shelter": (2,),
    "activity_space": (3, 5),
    "coordination": (4,),
    "outpost": (7,),
    "bishop_rook": (8, 9),
    "tempo": (10,),
    "full_stack": tuple(range(11)),
    "selected_stack": (1, 2, 6, 8, 9),
}


def residual(features, indices):
    *signals, phase = features
    raw = sum(WEIGHTS[i] * signals[i] for i in indices)
    return max(-180, min(180, raw * (8 + phase) // 16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pattern = re.compile(r"\| `([^`]+)` \| (-?\d+) \| (-?\d+) \| ([+-]?\d+) \|")
    rows = []
    for line in Path(args.comparison).read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        fen, sf, home, _ = match.groups()
        features = feature_vector(fen)
        full = residual(features, tuple(range(11)))
        board = chess.Board(fen)
        white_sign = 1 if board.turn == chess.WHITE else -1
        hce = int(home) - white_sign * full
        rows.append((int(sf), hce, features))
    with Path(args.out).open("w") as out:
        out.write("# Homemade breakthrough ablation\n\n")
        out.write("Scores are computed from the direct black-box Stockfish 19/SFNNv16 report. No Stockfish implementation or weights are used. `base` is the existing HCE score; every other row adds only the named original Unchessed feature family.\n\n")
        out.write("| Variant | MAE cp | Median absolute error cp |\n|---|---:|---:|\n")
        results=[]
        for name, indices in GROUPS.items():
            errors=[]
            for sf, hce, features in rows:
                score = hce + (residual(features, indices) if indices else 0)
                if name in {"full_stack", "selected_stack"}:
                    score = round(0.680292396 * score + 24.603188915)
                errors.append(abs(score - sf))
            errors.sort()
            mae=sum(errors)/len(errors)
            median=errors[len(errors)//2]
            results.append((name, mae, median))
            out.write(f"| {name} | {mae:.3f} | {median:.3f} |\n")
        out.write("\nThe final calibrated runtime applies the learned shrinkage and bias after the full stack; this table intentionally isolates feature-family contribution before that calibration.\n")
    for name, mae, median in results:
        print(f"{name}: mae_cp={mae:.3f} median_abs_error_cp={median:.3f}")

if __name__ == "__main__":
    main()
