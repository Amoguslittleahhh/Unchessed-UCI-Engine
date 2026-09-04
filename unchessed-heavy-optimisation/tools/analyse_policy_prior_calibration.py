#!/usr/bin/env python3
"""Is our policy usable as a PUCT-style probability prior, not just a ranking?

Motivation
----------

Lc0's PUCT rule (see the AlphaZero primer on lczero.org) selects edges by

    PUCT = Q + c_puct * P * sqrt(N_total) / (1 + N)

where `P` is the policy's **prior probability** for that move. Note what this
requires: `P` enters multiplicatively, so its *magnitude* matters, not merely
the order of the moves. A policy that ranks well but is systematically
over-confident will over-explore its favourite and starve the alternatives;
one that is under-confident will spread visits too thinly.

Our root hint uses the policy in the weakest possible way -- it sorts by logit
and throws the magnitudes away (`roots.sort_by(...policy_hint...)` in
`search.rs`). Every analysis in this repository has likewise measured only
ranking quality: top-1 accuracy, rank of the best move, first-move regret.

**Whether the magnitudes mean anything has never been tested.** That is worth
knowing independently of MCTS, because it decides which uses of the policy are
even available:

  - if `softmax(logits)` is a calibrated probability, the policy can weight
    things -- reduction amounts, futility margins, a PUCT prior -- and its
    confidence can be trusted as a signal;
  - if it is badly calibrated, the only sound use is the ordering we already
    have, and any confidence-weighted scheme would need a temperature fit
    first.

What it measures
----------------

1. **Reliability curve.** Bucket every legal move by its predicted
   `softmax(logit)` and compare against how often that move actually is the
   teacher's best. A calibrated prior sits on the diagonal.

2. **Expected Calibration Error (ECE)** -- the sample-weighted mean gap
   between predicted and actual, the standard single-number summary.

3. **Optimal temperature.** Fits `softmax(logits / T)` by minimising negative
   log-likelihood of the teacher's best move. `T > 1` means the raw policy is
   over-confident and should be softened; `T < 1` means under-confident.

4. **Top-1 confidence separation** -- mean predicted probability when the top
   move is right versus wrong. A prior whose confidence carries no information
   about correctness cannot be used to modulate anything.

Usage
-----
    python3 tools/analyse_policy_prior_calibration.py \\
        artifacts/unarchitectured-v1-final.unarchv1 \\
        artifacts/unarchitectured-v1-calibration-corpus.jsonl \\
        artifacts/unarchitectured-v1-calibration-labels.json

Requires torch (install separately; see tools/requirements-dev.txt).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import chess  # noqa: E402

from unarchitectured_v1_position_encoding import encode_position  # noqa: E402

CONFIG = {"d_model": 256, "heads": 8, "history_width": 32, "policy_adapter_rank": 16}
N_BINS = 10


def softmax(values, temperature=1.0):
    scaled = [v / temperature for v in values]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def collect(weights, rows, labels, rating, time_class, policy_kind):
    """Return per-position (logits, index_of_teacher_best)."""
    import torch

    from reference_forward_unarchitectured_v1 import forward

    out = []
    for rec in rows:
        label = labels.get(rec["fen"])
        if not label:
            continue
        board = chess.Board(rec["fen"])
        enc = encode_position(board)
        actions = list(enc["legal_actions"])
        n = len(actions)
        moves = enc["legal_moves"]
        best = max(label["scores"], key=lambda k: label["scores"][k])
        try:
            best_index = [m.uci() for m in moves].index(best)
        except ValueError:
            continue
        batch = {
            "pieces": torch.tensor([enc["pieces"]], dtype=torch.long),
            "castling": torch.tensor([enc["castling"]], dtype=torch.long),
            "ep_file": torch.tensor([enc["ep_file"]], dtype=torch.long),
            "halfmove_bucket": torch.tensor([enc["halfmove_bucket"]], dtype=torch.long),
            "rating": torch.tensor([rating], dtype=torch.long),
            "time_class": torch.tensor([time_class], dtype=torch.long),
            "policy_kind": torch.tensor([policy_kind], dtype=torch.long),
            "safe_actions": torch.tensor(
                [actions + [0xFFFF] * (218 - n)], dtype=torch.long
            ),
            "legal_mask": torch.tensor([[i < n for i in range(218)]]),
        }
        logits = forward(weights, batch, CONFIG, layers=8, width=256)["logits"][0][
            :n
        ].tolist()
        out.append((logits, best_index))
    return out


def reliability(samples, temperature=1.0):
    bins = [[0, 0, 0.0] for _ in range(N_BINS)]
    for logits, best_index in samples:
        probs = softmax(logits, temperature)
        for i, p in enumerate(probs):
            b = min(int(p * N_BINS), N_BINS - 1)
            bins[b][0] += int(i == best_index)
            bins[b][1] += 1
            bins[b][2] += p
    total = sum(b[1] for b in bins)
    ece = 0.0
    rows = []
    for idx, (hits, n, psum) in enumerate(bins):
        if n == 0:
            continue
        pred = psum / n
        actual = hits / n
        ece += n / total * abs(actual - pred)
        rows.append(
            {
                "bin_low": idx / N_BINS,
                "bin_high": (idx + 1) / N_BINS,
                "count": n,
                "mean_predicted": pred,
                "actual": actual,
                "gap": actual - pred,
            }
        )
    return rows, ece


def nll(samples, temperature):
    total = 0.0
    for logits, best_index in samples:
        probs = softmax(logits, temperature)
        total -= math.log(max(probs[best_index], 1e-12))
    return total / len(samples)


def fit_temperature(samples):
    """Golden-section search on a log grid; the NLL is unimodal in T."""
    best_t, best_v = 1.0, nll(samples, 1.0)
    t = 0.25
    while t <= 4.0001:
        v = nll(samples, t)
        if v < best_v:
            best_t, best_v = t, v
        t += 0.05
    return best_t, best_v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", type=Path)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--rating", type=int, default=2700)
    ap.add_argument("--time-class", type=int, default=2)
    ap.add_argument("--policy-kind", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for p in (args.package, args.corpus, args.labels):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    from reference_forward_unarchitectured_v1 import read_package

    weights = read_package(args.package)
    labels = json.loads(args.labels.read_text())
    rows = []
    with args.corpus.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if "fen" in rec:
                rows.append(rec)
    if args.limit:
        rows = rows[: args.limit]

    samples = collect(
        weights, rows, labels, args.rating, args.time_class, args.policy_kind
    )
    if not samples:
        print("no scored positions", file=sys.stderr)
        return 2

    curve, ece = reliability(samples)
    best_t, best_nll = fit_temperature(samples)
    tuned_curve, tuned_ece = reliability(samples, best_t)

    right, wrong = [], []
    for logits, best_index in samples:
        probs = softmax(logits)
        top = max(range(len(probs)), key=lambda i: probs[i])
        (right if top == best_index else wrong).append(probs[top])

    print(f"positions: {len(samples)}   "
          f"legal moves: {sum(len(s[0]) for s in samples)}\n")
    print("reliability of softmax(logits) as P(move is teacher-best)\n")
    print(f"{'bin':>12s} {'n':>7s} {'predicted':>10s} {'actual':>8s} {'gap':>8s}")
    print("-" * 50)
    for r in curve:
        print(f"{r['bin_low']:.1f}-{r['bin_high']:.1f} {r['count']:11d} "
              f"{r['mean_predicted']:10.3f} {r['actual']:8.3f} {r['gap']:+8.3f}")

    print(f"\nECE (raw)          {ece:.4f}")
    print(f"optimal temperature {best_t:.2f}   (NLL {best_nll:.4f} "
          f"vs {nll(samples, 1.0):.4f} at T=1)")
    print(f"ECE (tempered)     {tuned_ece:.4f}")
    print(f"\ntop-1 confidence when correct   {statistics.fmean(right):.3f} "
          f"(n={len(right)})")
    print(f"top-1 confidence when wrong     {statistics.fmean(wrong):.3f} "
          f"(n={len(wrong)})")
    print(f"separation                      "
          f"{statistics.fmean(right) - statistics.fmean(wrong):+.3f}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "positions": len(samples),
                    "ece_raw": ece,
                    "ece_tempered": tuned_ece,
                    "optimal_temperature": best_t,
                    "nll_at_one": nll(samples, 1.0),
                    "nll_at_optimum": best_nll,
                    "reliability": curve,
                    "reliability_tempered": tuned_curve,
                    "top1_confidence_correct": statistics.fmean(right),
                    "top1_confidence_wrong": statistics.fmean(wrong),
                    "n_correct": len(right),
                    "n_wrong": len(wrong),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
