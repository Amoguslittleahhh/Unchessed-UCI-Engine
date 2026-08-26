#!/usr/bin/env python3
"""Simulate the engine's time management without a Rust toolchain.

Faithful transcription of `Limits::budget` (unchessed-core/src/search.rs)
plus the situation-based soft-budget multiplier applied in
`go_with_root_hints`. Used two ways:

1. Audit evidence: `docs/parameter-calibration-audit.md` quotes the table
   this tool prints, i.e. what fraction of the remaining clock the current
   (never-tuned) ratios actually allocate at common time controls.
2. Stage-3 baseline for the calibration work order: after the ratios are
   exposed as UCI options and re-tuned on real hardware, re-running this
   tool with the new values shows the effect (the Rust
   `budget_speeds_up_as_clock_drains` test pins the behavior this must keep
   agreeing with).

The transcription is integer-for-integer with the Rust (u64 division
truncates toward zero; `(s as f64 * situation) as u64` truncates too).
`tools/test_simulate_time_budget.py` pins the exact values the Rust test's
assertions imply, so any Rust-side formula change that diverges from this
file fails the Python suite until the transcription is re-derived.

Usage:
  python3 tools/simulate_time_budget.py --t 300000,60000,20000,10000,6000,3000,2000,1000,500 \
      --inc 50,0,0,50,50,50,50,50,50 --legal-count 30
"""
from __future__ import annotations

import argparse
import sys


def budget(t: int, inc: int, movestogo: int | None = None, movetime: int | None = None) -> tuple[int, int]:
    """Mirror of Limits::budget for game mode. Returns (soft_ms, hard_ms)."""
    if movetime is not None:
        x = max(movetime - 25, 5)
        return (x, x)
    mtg = max(movestogo if movestogo is not None else 30, 1)
    soft = t // mtg + (inc * 3) // 4
    hard = max(t // 5 + inc // 2, soft)
    if t < 20_000:
        soft = min(soft, t // 35 + inc // 2)
        hard = min(hard, t // 10 + inc // 2)
    if t < 6_000:
        soft = min(soft, t // 60 + inc // 2)
        hard = min(hard, t // 16 + inc // 2)
    if t < 2_000:
        soft = min(soft, max(inc // 2, 30))
        hard = min(hard, t // 8)
    ceiling = max(t - 60, 5)
    hard = min(hard, ceiling)
    hard = max(hard, 5)
    soft = min(soft, hard)
    soft = max(soft, 3)
    return (soft, hard)


def situation_factor(legal_count: int, in_check: bool) -> float:
    """Mirror of the situation multiplier in go_with_root_hints."""
    width = min(max(0.65 + legal_count / 45.0, 0.75), 1.3)
    sharp = 1.25 if in_check else 1.0
    return width * sharp


def final_soft(soft: int, hard: int, legal_count: int, in_check: bool) -> int:
    """Mirror of `((s as f64 * situation) as u64).clamp(3, hard)`."""
    return min(max(int(soft * situation_factor(legal_count, in_check)), 3), hard)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--t", default="300000,60000,20000,10000,6000,3000,2000,1000,500",
                        help="comma-separated remaining clock (ms), one per row")
    parser.add_argument("--inc", default="50,0,0,50,50,50,50,50,50",
                        help="comma-separated increment (ms), one per row")
    parser.add_argument("--movestogo", type=int, default=None)
    parser.add_argument("--legal-count", type=int, default=30,
                        help="root legal move count for the situation factor")
    parser.add_argument("--in-check", action="store_true", help="apply the in-check sharp factor")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    ts = [int(x) for x in args.t.split(",") if x.strip() != ""]
    incs = [int(x) for x in args.inc.split(",") if x.strip() != ""]
    if len(ts) != len(incs):
        print("--t and --inc must have the same number of entries", file=sys.stderr)
        return 2
    print(f"{'time ms':>9} {'inc':>5} | {'soft':>6} {'hard':>7} | {'soft/final':>10} {'soft %':>6} {'final %':>7}")
    for t, inc in zip(ts, incs):
        soft, hard = budget(t, inc, args.movestogo)
        final = final_soft(soft, hard, args.legal_count, args.in_check)
        print(f"{t:>9} {inc:>5} | {soft:>6} {hard:>7} | {final:>10} {100.0 * soft / t:>5.1f}% {100.0 * final / t:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
