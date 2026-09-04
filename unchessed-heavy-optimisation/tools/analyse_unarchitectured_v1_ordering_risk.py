#!/usr/bin/env python3
"""Explain *why* the Unarchitectured v1 root hint costs Elo, using offline data.

The open question
-----------------

Round 7 ran four real SPRT batches. Every one landed negative, never positive
(-26.1, -15.1, -5.8 Elo). It also ran a depth/time calibration which showed the
obvious explanation is wrong: depth loss from the hint was negligible (mostly
0, occasionally +1, rarely -1) and the charged inference cost was 0-2ms. So the
"clock tax reduces depth" story is not supported by the project's own data, and
the round-7 writeup says plainly that the cause is still unexplained.

Its alternative hypothesis was that the *ordering itself* occasionally steers
the first pass toward a worse line that costs more to refute even at equal
nominal depth. That is testable offline, without games.

Why average accuracy cannot answer this
---------------------------------------

The existing calibration reports top-1 accuracy (0.2550 vs a 0.1567 heuristic
baseline) -- the model is clearly better *on average*. But move ordering is not
paid on the average. Alpha-beta gets almost all its benefit from searching a
good move first: ordering is nearly free when right and expensive when
confidently wrong, because a bad first move raises no useful bound and the
whole subtree must be re-refuted.

So a policy can be more accurate on average and still be a worse orderer, if
its errors are *worse* errors. That is the asymmetry this tool measures, using
the teacher's own per-move centipawn scores as ground truth:

  - **first-move regret**: cp lost by searching the policy's top move first
    rather than the teacher's best. This is the quantity ordering actually
    pays for.
  - **tail risk**: how often the first move is a blunder (>=200cp, >=500cp
    worse than best). Rare disasters dominate ordering cost.
  - **rank of the true best move**: how deep into the move list the search has
    to go before it meets the move that will eventually win the argument.
  - **confidently wrong**: cases where the policy is *sure* (large logit gap
    over its own second choice) and also badly wrong. These are the worst
    possible ordering inputs, because a confident wrong hint is exactly what
    stops the search reconsidering early.

Every metric is computed for the neural policy and for the free MVV-LVA
heuristic on the same positions, so the comparison controls for position
difficulty. Nothing here is a substitute for an SPRT; it is a mechanism probe
that explains a result the games already established.

Usage
-----
    python3 tools/analyse_unarchitectured_v1_ordering_risk.py \\
        artifacts/unarchitectured-v1-final.unarchv1 \\
        artifacts/unarchitectured-v1-calibration-corpus.jsonl \\
        artifacts/unarchitectured-v1-calibration-labels.json

Requires torch (not in tools/requirements-dev.txt; install separately).
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

from calibrate_unarchitectured_v1_policy import heuristic_move_score  # noqa: E402
from unarchitectured_v1_position_encoding import encode_position  # noqa: E402

BLUNDER_CP = 200
DISASTER_CP = 500
CONFIDENT_GAP = 0.5

# Mate scores arrive as +/-100000 (`mate_score=100000` in the labeller). Left
# raw they dominate every mean -- a single missed mate outweighs hundreds of
# ordinary positions and the average stops describing typical behaviour. They
# are clamped to +/-MATE_CLAMP_CP so a missed mate still counts as the worst
# possible ordinary error without swamping the statistic. Blunder and disaster
# *rates* are unaffected by the clamp (both thresholds sit far below it), so
# the tail metrics stay honest either way.
MATE_CLAMP_CP = 2000


def clamp_cp(value: int) -> float:
    return float(max(-MATE_CLAMP_CP, min(MATE_CLAMP_CP, value)))


def build_batch(board: chess.Board, rating: int, time_class: int, policy_kind: int):
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


def summarise(name, regrets, ranks, confident_regrets, legal_counts):
    n = len(regrets)
    if n == 0:
        return {"name": name, "positions": 0}
    srt = sorted(regrets)

    def pct(p):
        if n == 1:
            return float(srt[0])
        pos = (n - 1) * p
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return float(srt[lo] + (srt[hi] - srt[lo]) * (pos - lo))

    return {
        "name": name,
        "positions": n,
        "top1_accuracy": sum(1 for r in regrets if r == 0) / n,
        "mean_first_move_regret_cp": statistics.fmean(regrets),
        "median_first_move_regret_cp": pct(0.50),
        "p90_first_move_regret_cp": pct(0.90),
        "p99_first_move_regret_cp": pct(0.99),
        "max_first_move_regret_cp": float(srt[-1]),
        "blunder_rate_200cp": sum(1 for r in regrets if r >= BLUNDER_CP) / n,
        "disaster_rate_500cp": sum(1 for r in regrets if r >= DISASTER_CP) / n,
        "mean_best_move_rank": statistics.fmean(ranks),
        "median_best_move_rank": statistics.median(ranks),
        "best_move_in_top3_rate": sum(1 for r in ranks if r < 3) / n,
        "best_move_in_bottom_half_rate": sum(
            1 for r, lc in zip(ranks, legal_counts) if r >= lc / 2
        )
        / n,
        "confident_cases": len(confident_regrets),
        "confident_mean_regret_cp": (
            statistics.fmean(confident_regrets) if confident_regrets else 0.0
        ),
        "confident_blunder_rate": (
            sum(1 for r in confident_regrets if r >= BLUNDER_CP)
            / len(confident_regrets)
            if confident_regrets
            else 0.0
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", type=Path)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--rating", type=int, default=2700)
    ap.add_argument("--time-class", type=int, default=2)
    ap.add_argument("--policy-kind", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap positions (0 = all)")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args()

    for p in (args.package, args.corpus, args.labels):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    from reference_forward_unarchitectured_v1 import forward, read_package

    weights = read_package(args.package)
    config = {
        "d_model": 256,
        "heads": 8,
        "history_width": 32,
        "policy_adapter_rank": 16,
    }
    labels = json.loads(args.labels.read_text())

    rows = []
    with args.corpus.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if "fen" in rec:
                rows.append(rec)
    if args.limit:
        rows = rows[: args.limit]

    neural = {"regret": [], "rank": [], "confident": [], "legal": []}
    heur = {"regret": [], "rank": [], "confident": [], "legal": []}
    movegen = {"regret": [], "rank": [], "confident": [], "legal": []}
    worst = []
    skipped = 0

    for rec in rows:
        fen = rec["fen"]
        label = labels.get(fen)
        if not label:
            skipped += 1
            continue
        scores = label["scores"]
        best_cp = label["best_score"]

        board = chess.Board(fen)
        batch, legal_moves, n = build_batch(
            board, args.rating, args.time_class, args.policy_kind
        )
        if not all(m.uci() in scores for m in legal_moves):
            skipped += 1
            continue

        out = forward(weights, batch, config, layers=8, width=256)
        logits = out["logits"][0][:n].tolist()

        # Teacher ranking: index 0 is objectively best.
        order_true = sorted(range(n), key=lambda i: -scores[legal_moves[i].uci()])

        for tag, key in (
            ("neural", [logits[i] for i in range(n)]),
            (
                "heuristic",
                [float(heuristic_move_score(board, legal_moves[i])) for i in range(n)],
            ),
            # The ACTUAL baseline the hint displaces. At depth == start_depth
            # every root score is still -MATE, so `sort_by_key(-score)` is a
            # no-op and the order is simply movegen order. This is what the
            # hint is really being compared against in the SPRT, not MVV-LVA.
            ("movegen", [float(-i) for i in range(n)]),
        ):
            bucket = {"neural": neural, "heuristic": heur, "movegen": movegen}[tag]
            order = sorted(range(n), key=lambda i: -key[i])
            first = order[0]
            regret = clamp_cp(best_cp) - clamp_cp(scores[legal_moves[first].uci()])
            rank_of_best = order.index(order_true[0])

            bucket["regret"].append(float(regret))
            bucket["rank"].append(rank_of_best)
            bucket["legal"].append(n)

            if n >= 2:
                gap = key[order[0]] - key[order[1]]
                if gap >= CONFIDENT_GAP:
                    bucket["confident"].append(float(regret))

            if tag == "neural":
                worst.append(
                    {
                        "fen": fen,
                        "phase": rec.get("phase"),
                        "legal_count": n,
                        "policy_first_move": legal_moves[first].uci(),
                        "policy_first_move_cp": scores[legal_moves[first].uci()],
                        "teacher_best_move": legal_moves[order_true[0]].uci(),
                        "teacher_best_cp": best_cp,
                        "regret_cp": int(regret),
                        "teacher_best_rank_in_policy": rank_of_best,
                    }
                )

    worst.sort(key=lambda r: -r["regret_cp"])
    report = {
        "positions_scored": len(neural["regret"]),
        "positions_skipped": skipped,
        "blunder_threshold_cp": BLUNDER_CP,
        "disaster_threshold_cp": DISASTER_CP,
        "confident_logit_gap": CONFIDENT_GAP,
        "neural": summarise(
            "neural policy",
            neural["regret"],
            neural["rank"],
            neural["confident"],
            neural["legal"],
        ),
        "heuristic": summarise(
            "MVV-LVA heuristic",
            heur["regret"],
            heur["rank"],
            heur["confident"],
            heur["legal"],
        ),
        "movegen": summarise(
            "movegen order (real baseline)",
            movegen["regret"],
            movegen["rank"],
            movegen["confident"],
            movegen["legal"],
        ),
        "worst_cases": worst[:15],
    }

    n_rep, h_rep, m_rep = (
        report["neural"],
        report["heuristic"],
        report["movegen"],
    )
    print(f"positions scored: {report['positions_scored']} (skipped {skipped})\n")
    print(f"{'metric':42s} {'neural':>12s} {'heuristic':>12s} {'movegen':>12s}")
    print("-" * 81)
    for key, fmt in (
        ("top1_accuracy", "{:.4f}"),
        ("mean_first_move_regret_cp", "{:.1f}"),
        ("median_first_move_regret_cp", "{:.1f}"),
        ("p90_first_move_regret_cp", "{:.1f}"),
        ("p99_first_move_regret_cp", "{:.1f}"),
        ("max_first_move_regret_cp", "{:.0f}"),
        ("blunder_rate_200cp", "{:.4f}"),
        ("disaster_rate_500cp", "{:.4f}"),
        ("mean_best_move_rank", "{:.2f}"),
        ("median_best_move_rank", "{:.1f}"),
        ("best_move_in_top3_rate", "{:.4f}"),
        ("best_move_in_bottom_half_rate", "{:.4f}"),
        ("confident_cases", "{:.0f}"),
        ("confident_mean_regret_cp", "{:.1f}"),
        ("confident_blunder_rate", "{:.4f}"),
    ):
        print(
            f"{key:42s} {fmt.format(n_rep[key]):>12s} "
            f"{fmt.format(h_rep[key]):>12s} {fmt.format(m_rep[key]):>12s}"
        )

    print("\nworst neural first-move choices:")
    for w in report["worst_cases"][:5]:
        print(
            f"  {w['regret_cp']:6d}cp  played {w['policy_first_move']:6s} "
            f"({w['policy_first_move_cp']:+6d}) vs best {w['teacher_best_move']:6s} "
            f"({w['teacher_best_cp']:+6d})  best ranked "
            f"{w['teacher_best_rank_in_policy']}/{w['legal_count']}"
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
