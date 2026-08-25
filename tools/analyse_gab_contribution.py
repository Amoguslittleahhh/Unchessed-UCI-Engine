#!/usr/bin/env python3
"""Measure how much Geometric Attention Bias actually contributes in our
checkpoint, and compare its capacity against the Chessformer paper's.

Why
---

`Chessformer: A Unified Architecture for Chess Modeling`
(arXiv:2605.19091) is the canonical reference for the architecture
Unarchitectured v1 derives from: square tokens, an attention-based
source-destination policy head, and **Geometric Attention Bias** -- a dynamic
positional encoding generated from a compressed board representation and mixed
from a set of learned templates.

The paper's central ablation claim is that **GAB is the key driver** of the
architecture's advantage, contributing significant improvements over absolute
and relative position encodings for Elo, puzzle accuracy, and policy accuracy.

That claim is testable on our own weights, and it matters here because our GAB
is configured far smaller than any model in the paper:

    dimension          ours    paper 5M    paper 23M/79M
    d1 (token proj)       8          32               32
    d2 (compress)        32          64              128
    d3 (templates)       32          64              128

Our d1 is 4x smaller than the smallest configuration the paper reports, and
d2/d3 are half. If GAB really is the key driver, an under-provisioned GAB is a
concrete, named architectural deficiency in this checkpoint -- worth knowing
before anyone spends another round on kernel micro-optimisation.

What this measures
------------------

Two ablations on the real exported package, replayed over the calibration
corpus:

1. **GAB zeroed** -- set the geometric bias to zero, leaving pure dot-product
   attention. This is the paper's "no positional encoding" baseline. If GAB
   carries little signal, accuracy will barely move.

2. **GAB templates shuffled** -- permute the template bank so the bias is
   structurally intact but semantically wrong. This separates "GAB matters"
   from "any bias tensor of the right shape matters".

Both are inference-time interventions on frozen weights. Nothing is retrained,
so this measures how much the *trained* model relies on GAB, not how much a
differently-sized GAB would help. That distinction is stated again in the
output.

Usage
-----
    python3 tools/analyse_gab_contribution.py \\
        artifacts/unarchitectured-v1-final.unarchv1 \\
        artifacts/unarchitectured-v1-calibration-corpus.jsonl \\
        artifacts/unarchitectured-v1-calibration-labels.json

Requires torch (install separately; see tools/requirements-dev.txt).
"""

from __future__ import annotations

import argparse
import json
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

MATE_CLAMP_CP = 2000
CONFIG = {"d_model": 256, "heads": 8, "history_width": 32, "policy_adapter_rank": 16}


def clamp_cp(v: int) -> float:
    return float(max(-MATE_CLAMP_CP, min(MATE_CLAMP_CP, v)))


def build_batch(board, rating, time_class, policy_kind):
    import torch

    enc = encode_position(board)
    actions = list(enc["legal_actions"])
    n = len(actions)
    padded = actions + [0xFFFF] * (218 - n)
    batch = {
        "pieces": torch.tensor([enc["pieces"]], dtype=torch.long),
        "castling": torch.tensor([enc["castling"]], dtype=torch.long),
        "ep_file": torch.tensor([enc["ep_file"]], dtype=torch.long),
        "halfmove_bucket": torch.tensor([enc["halfmove_bucket"]], dtype=torch.long),
        "rating": torch.tensor([rating], dtype=torch.long),
        "time_class": torch.tensor([time_class], dtype=torch.long),
        "policy_kind": torch.tensor([policy_kind], dtype=torch.long),
        "safe_actions": torch.tensor([padded], dtype=torch.long),
        "legal_mask": torch.tensor([[i < n for i in range(218)]]),
    }
    return batch, enc["legal_moves"], n


def evaluate(weights, rows, labels, args, tag):
    """Return (top1, mean_regret) for one weight variant."""
    from reference_forward_unarchitectured_v1 import forward

    hits = 0
    scored = 0
    regrets = []
    for rec in rows:
        fen = rec["fen"]
        label = labels.get(fen)
        if not label:
            continue
        scores = label["scores"]
        board = chess.Board(fen)
        batch, legal_moves, n = build_batch(
            board, args.rating, args.time_class, args.policy_kind
        )
        if not all(m.uci() in scores for m in legal_moves):
            continue
        out = forward(weights, batch, CONFIG, layers=8, width=256)
        logits = out["logits"][0][:n].tolist()
        top = max(range(n), key=lambda i: logits[i])
        chosen = legal_moves[top].uci()
        # Count a hit as zero regret, not string equality with one arbitrary
        # argmax. 32 of the 600 positions have two or more moves sharing the
        # top teacher score; picking any of them is equally correct, and
        # string comparison would mark all but one wrong. This matches the
        # definition used by analyse_unarchitectured_v1_ordering_risk.py, so
        # the baselines are directly comparable.
        regret = clamp_cp(label["best_score"]) - clamp_cp(scores[chosen])
        hits += int(regret == 0)
        regrets.append(regret)
        scored += 1
    return {
        "variant": tag,
        "positions": scored,
        "top1_accuracy": hits / scored if scored else 0.0,
        "mean_regret_cp": statistics.fmean(regrets) if regrets else 0.0,
    }


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

    import torch

    from reference_forward_unarchitectured_v1 import read_package

    base = read_package(args.package)
    labels = json.loads(args.labels.read_text())
    rows = []
    with args.corpus.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if "fen" in rec:
                rows.append(rec)
    if args.limit:
        rows = rows[: args.limit]

    templates = base["gab.templates"]
    n_templates = templates.shape[0]

    # Report the capacity gap against the paper's own configurations.
    d1 = base["gab.token_projection"].shape[0]
    d2 = base["gab.compress.weight"].shape[0]
    print("GAB capacity vs arXiv:2605.19091 configurations")
    print(f"{'dimension':22s} {'ours':>6s} {'paper 5M':>9s} {'paper 23M/79M':>14s}")
    print("-" * 55)
    print(f"{'d1 (token projection)':22s} {d1:6d} {32:9d} {32:14d}")
    print(f"{'d2 (compress)':22s} {d2:6d} {64:9d} {128:14d}")
    print(f"{'d3 (templates)':22s} {n_templates:6d} {64:9d} {128:14d}")

    variants = {}
    variants["baseline"] = base

    zeroed = dict(base)
    zeroed["gab.templates"] = torch.zeros_like(templates)
    variants["gab_zeroed"] = zeroed

    shuffled = dict(base)
    generator = torch.Generator().manual_seed(20260825)
    perm = torch.randperm(n_templates, generator=generator)
    shuffled["gab.templates"] = templates[perm].clone()
    variants["gab_shuffled"] = shuffled

    results = []
    for tag, weights in variants.items():
        results.append(evaluate(weights, rows, labels, args, tag))

    print(f"\nablation results ({results[0]['positions']} positions)\n")
    print(f"{'variant':16s} {'top1':>8s} {'mean regret':>12s} {'delta top1':>11s}")
    print("-" * 52)
    ref = results[0]["top1_accuracy"]
    for r in results:
        print(
            f"{r['variant']:16s} {r['top1_accuracy']:8.4f} "
            f"{r['mean_regret_cp']:12.1f} {r['top1_accuracy'] - ref:+11.4f}"
        )

    print(
        "\nNote: these are inference-time ablations on frozen weights. They "
        "measure how much the TRAINED model relies on GAB, not how much a "
        "larger GAB would help -- that would need a retrain."
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "capacity": {
                        "ours": {"d1": d1, "d2": d2, "d3": n_templates},
                        "paper_5m": {"d1": 32, "d2": 64, "d3": 64},
                        "paper_23m_79m": {"d1": 32, "d2": 128, "d3": 128},
                    },
                    "ablations": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
