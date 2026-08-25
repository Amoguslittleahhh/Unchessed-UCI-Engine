#!/usr/bin/env python3
"""Measure whether the `rating` and `policy_kind` inputs actually condition
the Unarchitectured v1 policy.

Motivation
----------

Maia ships a *separate network per rating band* (Maia 600, 800, ... 2600) to
model players of different strengths. Chessformer (arXiv:2605.19091) takes the
other approach and conditions one model on rating, reporting that its Maia-3
family reaches state-of-the-art human move-matching precisely because a single
rating-conditioned model works.

Unarchitectured v1 is built for the conditioned approach. It has:

  - a `rating` input, projected through `rating_weight`/`rating_bias` into the
    32-wide history vector;
  - a `policy_kind` input selecting between `POLICY_HUMAN` (0) and
    `POLICY_GUIDE` (1), driving LoRA-adapted policy heads.

Both are live in the shipped engine: `uci.rs` clamps the `UCI_Elo` option to
500..3200 and passes it straight through as the rating. So a user changing
`UCI_Elo` believes they are changing how the model plays.

**Nothing in this repository has ever evaluated either input.** Every
calibration, ablation and analysis tool hardcodes `--rating 2700` and
`--policy-kind 1`. This tool varies them and measures whether the outputs
respond.

What it measures
----------------

Across a corpus of real positions, for each rating in a sweep:

  - how often the top-1 move changes relative to the lowest rating;
  - the maximum and mean absolute change in the legal-move logits;
  - agreement with the teacher's best move, per rating, when labels are
    supplied.

And separately, `POLICY_HUMAN` versus `POLICY_GUIDE` at a fixed rating.

An input that is genuinely conditioning the model should move the top-1 move
for a meaningful fraction of positions across a 2600-point rating span. An
input that changes logits by less than typical float noise is decorative.

Usage
-----
    python3 tools/analyse_rating_conditioning.py \\
        artifacts/unarchitectured-v1-final.unarchv1 \\
        artifacts/unarchitectured-v1-calibration-corpus.jsonl \\
        --labels artifacts/unarchitectured-v1-calibration-labels.json

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

CONFIG = {"d_model": 256, "heads": 8, "history_width": 32, "policy_adapter_rank": 16}
# The Maia ladder, which is the comparison this is really about.
DEFAULT_RATINGS = [600, 1000, 1400, 1800, 2200, 2600, 3200]
MATE_CLAMP_CP = 2000


def clamp_cp(v: int) -> float:
    return float(max(-MATE_CLAMP_CP, min(MATE_CLAMP_CP, v)))


def policy_logits(weights, board, rating, policy_kind, time_class=2):
    import torch

    from reference_forward_unarchitectured_v1 import forward

    enc = encode_position(board)
    actions = list(enc["legal_actions"])
    n = len(actions)
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
    out = forward(weights, batch, CONFIG, layers=8, width=256)
    return out["logits"][0][:n].tolist(), enc["legal_moves"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", type=Path)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--ratings", type=int, nargs="+", default=DEFAULT_RATINGS)
    ap.add_argument("--policy-kind", type=int, default=1)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for p in (args.package, args.corpus):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    from reference_forward_unarchitectured_v1 import read_package

    weights = read_package(args.package)
    labels = json.loads(args.labels.read_text()) if args.labels else {}

    rows = []
    with args.corpus.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if "fen" in rec:
                rows.append(rec)
    if args.limit:
        rows = rows[: args.limit]

    base_rating = args.ratings[0]
    per_rating = {r: {"changed": 0, "hits": 0, "scored": 0, "deltas": []}
                  for r in args.ratings}
    kind_changed = 0
    kind_scored = 0

    for rec in rows:
        board = chess.Board(rec["fen"])
        base_logits, moves = policy_logits(
            weights, board, base_rating, args.policy_kind
        )
        base_top = max(range(len(base_logits)), key=lambda i: base_logits[i])
        label = labels.get(rec["fen"])

        for rating in args.ratings:
            if rating == base_rating:
                logits, mv = base_logits, moves
            else:
                logits, mv = policy_logits(weights, board, rating, args.policy_kind)
            top = max(range(len(logits)), key=lambda i: logits[i])
            bucket = per_rating[rating]
            bucket["changed"] += int(top != base_top)
            bucket["deltas"].append(
                max(abs(a - b) for a, b in zip(logits, base_logits))
            )
            if label:
                best = max(label["scores"], key=lambda k: label["scores"][k])
                bucket["hits"] += int(mv[top].uci() == best)
                bucket["scored"] += 1

        human, hmoves = policy_logits(weights, board, 1500, 0)
        guide, gmoves = policy_logits(weights, board, 1500, 1)
        htop = hmoves[max(range(len(human)), key=lambda i: human[i])].uci()
        gtop = gmoves[max(range(len(guide)), key=lambda i: guide[i])].uci()
        kind_changed += int(htop != gtop)
        kind_scored += 1

    n = len(rows)
    print(f"positions: {n}   baseline rating: {base_rating}\n")
    print(f"{'rating':>7s} {'top1 changed':>13s} {'mean |dlogit|':>14s} "
          f"{'max |dlogit|':>13s} {'agree w/ teacher':>17s}")
    print("-" * 70)
    report = {"positions": n, "baseline_rating": base_rating, "ratings": {}}
    for rating in args.ratings:
        b = per_rating[rating]
        agree = f"{b['hits']/b['scored']:.4f}" if b["scored"] else "n/a"
        print(f"{rating:7d} {b['changed']:6d} ({b['changed']/n*100:4.1f}%) "
              f"{statistics.fmean(b['deltas']):14.6f} {max(b['deltas']):13.6f} "
              f"{agree:>17s}")
        report["ratings"][rating] = {
            "top1_changed": b["changed"],
            "top1_changed_pct": b["changed"] / n * 100,
            "mean_abs_delta": statistics.fmean(b["deltas"]),
            "max_abs_delta": max(b["deltas"]),
            "teacher_agreement": (b["hits"] / b["scored"]) if b["scored"] else None,
        }

    print(f"\nPOLICY_HUMAN vs POLICY_GUIDE at rating 1500: "
          f"top-1 differs in {kind_changed}/{kind_scored} "
          f"({kind_changed/kind_scored*100:.1f}%)")
    report["policy_kind"] = {
        "changed": kind_changed,
        "scored": kind_scored,
        "changed_pct": kind_changed / kind_scored * 100,
    }

    span = report["ratings"][args.ratings[-1]]
    if span["top1_changed"] == 0:
        print(
            "\nVERDICT: the rating input never changed the chosen move across the "
            "full sweep. It is not conditioning the policy in any way a user "
            "could observe."
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
