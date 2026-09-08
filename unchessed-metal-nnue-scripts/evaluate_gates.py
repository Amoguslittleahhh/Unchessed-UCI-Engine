#!/usr/bin/env python3
"""Evaluate a student JSONL against reference score fields.

Student rows must preserve position_id and contain student_cp; reference rows
must contain score_cp or score_mate. This intentionally reports failures rather
than inventing a parity score for missing terminal labels.

Ported from manus/research-facilities@e4851ba with one real bug fixed, found
in review before any real run: the original `score()` encoded mate references
as +-30000ish and folded them into the SAME mae_cp average as ordinary
centipawn scores. That is exactly the mistake that already produced one
bogus "29544 cp MAE" headline number earlier in this same research thread
(comparing Stockfish's search-found mate scores against a student's ordinary
score on forced-mate positions) -- a handful of mate positions can swing the
whole gate by tens of thousands of cp, drowning out the real, ordinary-position
signal the 100cp gate is supposed to measure. mate-labeled and terminal
reference rows are now scored separately; mae_gate_pass is judged on the
ordinary (non-mate, non-terminal) positions only, matching the training
procedure's own stated requirement for "separate MAE for tactical and
terminal strata" and "mate-distance accuracy" as distinct gates.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def is_decisive_reference(row: dict) -> bool:
    """True for a reference row this script won't put in the ordinary MAE pool:
    a proven mate, or a terminal (no-legal-move) position from label_uci.py."""
    return row.get("score_mate") is not None or row.get("terminal") is True


def mate_signed_cp(mate_plies: int) -> float:
    """A large, clearly-out-of-ordinary-range magnitude, sign matching the
    side-to-move mating (positive) or being mated (negative). Not intended to
    be compared against ordinary centipawn scores -- see module docstring."""
    sign = 1.0 if mate_plies > 0 else -1.0
    return sign * (30_000 - min(abs(mate_plies), 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--mae-limit", type=float, default=100.0)
    ap.add_argument("--mate-mae-limit", type=float, default=None,
                     help="optional separate gate for mate-labeled positions; unset = report only, no pass/fail")
    args = ap.parse_args()

    ref = {}
    for x in Path(args.reference).read_text().splitlines():
        if x.strip():
            row = json.loads(x)
            ref[row["position_id"]] = row

    ordinary_errors: list[float] = []
    mate_errors: list[float] = []
    skipped_terminal = 0
    skipped_unaligned = 0

    for raw in Path(args.student).read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        r = ref.get(row["position_id"])
        b = row.get("student_cp")
        if r is None or b is None:
            skipped_unaligned += 1
            continue
        if r.get("terminal") is True and r.get("score_mate") is None:
            # No-legal-move position with no resolved mate score to compare
            # against (see label_uci.py's terminal handling) -- can't score it.
            skipped_terminal += 1
            continue
        if r.get("score_mate") is not None:
            mate_errors.append(abs(mate_signed_cp(int(r["score_mate"])) - float(b)))
            continue
        a = r.get("score_cp")
        if a is None:
            skipped_unaligned += 1
            continue
        ordinary_errors.append(abs(float(a) - float(b)))

    if not ordinary_errors and not mate_errors:
        raise SystemExit("no aligned score rows")

    result = {
        "ordinary_positions": len(ordinary_errors),
        "mate_positions": len(mate_errors),
        "skipped_terminal_no_mate_score": skipped_terminal,
        "skipped_unaligned": skipped_unaligned,
    }
    gate_pass = True
    if ordinary_errors:
        mae = statistics.mean(ordinary_errors)
        result.update(
            mae_cp=mae,
            median_abs_error_cp=statistics.median(ordinary_errors),
            max_abs_error_cp=max(ordinary_errors),
            mae_gate_pass=mae <= args.mae_limit,
        )
        gate_pass = gate_pass and (mae <= args.mae_limit)
    else:
        result["mae_gate_pass"] = None
    if mate_errors:
        mate_mae = statistics.mean(mate_errors)
        result.update(
            mate_mae_cp=mate_mae,
            mate_median_abs_error_cp=statistics.median(mate_errors),
            mate_max_abs_error_cp=max(mate_errors),
        )
        if args.mate_mae_limit is not None:
            mate_pass = mate_mae <= args.mate_mae_limit
            result["mate_mae_gate_pass"] = mate_pass
            gate_pass = gate_pass and mate_pass

    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
