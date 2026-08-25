#!/usr/bin/env python3
"""Pentanomial SPRT / Elo analysis for paired-game match results.

Why this exists
---------------

Fishtest analyses results with the **pentanomial** model, not the trinomial
one (see the Fishtest mathematics page). Paired games -- the same opening
played once with each colour -- are not independent, and treating them as 2N
independent games overstates the information available. The pentanomial model
scores each *pair* on five outcomes (LL, LD, DD/WL, WD, WW) and, in Fishtest's
own words, "leads to a substantial saving of testing resources".

Every result this project has recorded is trinomial. Round 7's batches were
reported as W-L-D with a +/- Elo band:

    MinTime=1000, tc=5+0.05   600 games  172-217-211   -26.1 +/- 22.4
    MinTime=1000 (replication) 600 games  184-210-206   -15.1 +/- 22.5
    MinTime=30000, tc=60+0.6   300 games   72-77-151     -5.8 +/- 27.7

Those runs used `cutechess-cli -repeat -games 2`, so they *were* paired -- the
pairing information existed and was discarded at analysis time. Nothing in
this repository could compute a pentanomial interval, so this tool adds that
capability for any future run, and can re-derive the trinomial numbers for
comparison.

What it computes
----------------

From a pentanomial count vector `[LL, LD, DD+WL, WD, WW]`:

  - score, its variance under the pairing model, and an Elo estimate with a
    confidence interval;
  - **normalized Elo** (Elo divided by the per-game standard deviation),
    which Fishtest uses for bounds because expected test duration then
    depends only on the bounds, not on the draw ratio or opening book;
  - the SPRT log-likelihood ratio against `elo0`/`elo1` bounds, plus the
    accept/reject thresholds;
  - the variance ratio against the equivalent trinomial analysis, which is
    the concrete "resource saving" the pairing buys.

It can also read a cutechess-cli PGN and derive the pentanomial counts
directly, so a real run needs no manual bookkeeping.

Usage
-----
    # from counts (LL LD DD+WL WD WW)
    python3 tools/pentanomial_sprt.py --counts 12 88 210 84 6

    # from a cutechess PGN produced with -repeat -games 2
    python3 tools/pentanomial_sprt.py --pgn results.pgn --engine Hint

    # with explicit SPRT bounds
    python3 tools/pentanomial_sprt.py --counts 12 88 210 84 6 \\
        --elo0 0 --elo1 5 --alpha 0.05 --beta 0.05

Only the standard library is required for `--counts`. `--pgn` needs no
dependencies either: the parser reads `[White]`, `[Black]` and `[Result]`
tags directly.

Caveat: this is a *post-hoc* analysis of finished results. A real sequential
test must evaluate the LLR as games arrive and stop at a boundary; applying
these bounds to a batch that was already run does not give the same error
guarantees.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Elo <-> score conversion uses the standard logistic model.
LOG10_OVER_400 = math.log(10.0) / 400.0


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def pentanomial_stats(counts):
    """Return score, per-pair variance, and total pairs.

    Pair outcomes are scored in units of *points per game*: a pair is worth
    0, 0.25, 0.5, 0.75 or 1.0 on average across its two games.
    """
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    n = sum(counts)
    if n == 0:
        raise ValueError("no pairs")
    mean = sum(c * v for c, v in zip(counts, values)) / n
    var = sum(c * (v - mean) ** 2 for c, v in zip(counts, values)) / n
    return mean, var, n


def trinomial_from_pentanomial(counts):
    """Approximate W/D/L game counts implied by pentanomial pair counts.

    Exact for LL/WW and for the outer mixed cells; the middle cell is the
    documented ambiguity (DD and WL both land there), so it is split as all
    draws. That is the standard convention and is why the trinomial view
    understates variance.
    """
    ll, ld, mid, wd, ww = counts
    wins = 2 * ww + wd
    losses = 2 * ll + ld
    draws = 2 * mid + wd + ld
    return wins, draws, losses


def trinomial_stats(wins, draws, losses):
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games")
    mean = (wins + 0.5 * draws) / n
    var = (
        wins * (1.0 - mean) ** 2 + draws * (0.5 - mean) ** 2 + losses * mean**2
    ) / n
    return mean, var, n


def sprt_llr(counts, elo0: float, elo1: float):
    """GSPRT-style log-likelihood ratio using the pentanomial variance.

    Follows the Brownian-motion approximation Fishtest documents: the LLR for
    a drift test with variance estimated from the sample. This is the
    practical form used by cutechess and fishtest alike.
    """
    mean, var, n = pentanomial_stats(counts)
    if var <= 0.0:
        return 0.0
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2.0 * var)


def analyse(counts, elo0, elo1, alpha, beta):
    mean, var, pairs = pentanomial_stats(counts)
    games = 2 * pairs

    # Standard error of the mean score, in points per game.
    stderr = math.sqrt(var / pairs) if pairs else float("inf")
    elo = score_to_elo(mean)
    lo = score_to_elo(mean - 1.959963985 * stderr)
    hi = score_to_elo(mean + 1.959963985 * stderr)

    # Normalized Elo: Fishtest's scale-free measure.
    normalized = (mean - 0.5) / math.sqrt(var) if var > 0 else 0.0

    wins, draws, losses = trinomial_from_pentanomial(counts)
    t_mean, t_var, t_games = trinomial_stats(wins, draws, losses)
    t_stderr = math.sqrt(t_var / t_games) if t_games else float("inf")
    t_lo = score_to_elo(t_mean - 1.959963985 * t_stderr)
    t_hi = score_to_elo(t_mean + 1.959963985 * t_stderr)

    llr = sprt_llr(counts, elo0, elo1)
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))

    if llr >= upper:
        decision = "accept H1"
    elif llr <= lower:
        decision = "reject H1 (accept H0)"
    else:
        decision = "inconclusive"

    # Probability the change is an improvement, from the normal approximation.
    los = 0.5 * (1.0 + math.erf((mean - 0.5) / (stderr * math.sqrt(2.0)))) if stderr else 0.5

    return {
        "counts": list(counts),
        "pairs": pairs,
        "games": games,
        "score": mean,
        "pentanomial": {
            "variance_per_pair": var,
            "elo": elo,
            "elo_ci95": [lo, hi],
            "elo_pm": (hi - lo) / 2.0,
            "normalized_elo": normalized,
        },
        "trinomial": {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "elo": score_to_elo(t_mean),
            "elo_ci95": [t_lo, t_hi],
            "elo_pm": (t_hi - t_lo) / 2.0,
        },
        "variance_ratio_trinomial_over_pentanomial": (
            (t_var / 2.0) / var if var > 0 else float("inf")
        ),
        "los": los,
        "sprt": {
            "elo0": elo0,
            "elo1": elo1,
            "alpha": alpha,
            "beta": beta,
            "llr": llr,
            "upper_bound": upper,
            "lower_bound": lower,
            "decision": decision,
        },
    }


def counts_from_pgn(path: Path, engine: str):
    """Derive pentanomial pair counts from a cutechess `-repeat -games 2` PGN."""
    results = []
    white = black = result = None
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("[White "):
            white = line.split('"')[1]
        elif line.startswith("[Black "):
            black = line.split('"')[1]
        elif line.startswith("[Result "):
            result = line.split('"')[1]
            if white and black and result and result != "*":
                if result == "1/2-1/2":
                    score = 0.5
                elif result == "1-0":
                    score = 1.0 if white == engine else 0.0
                elif result == "0-1":
                    score = 1.0 if black == engine else 0.0
                else:
                    white = black = result = None
                    continue
                results.append(score)
            white = black = result = None

    if len(results) % 2:
        results = results[:-1]  # drop an unpaired trailing game

    counts = [0, 0, 0, 0, 0]
    for i in range(0, len(results), 2):
        total = results[i] + results[i + 1]  # 0, 0.5, 1.0, 1.5, or 2.0
        counts[int(total * 2)] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--counts",
        nargs=5,
        type=int,
        metavar=("LL", "LD", "DD_WL", "WD", "WW"),
        help="pentanomial pair counts",
    )
    ap.add_argument("--pgn", type=Path, help="cutechess PGN from a -repeat -games 2 run")
    ap.add_argument("--engine", help="engine name to score from the PGN")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.pgn:
        if not args.engine:
            ap.error("--pgn requires --engine")
        if not args.pgn.is_file():
            print(f"missing: {args.pgn}", file=sys.stderr)
            return 2
        counts = counts_from_pgn(args.pgn, args.engine)
    elif args.counts:
        counts = args.counts
    else:
        ap.error("give --counts or --pgn")

    if sum(counts) == 0:
        print("no complete game pairs found", file=sys.stderr)
        return 2

    r = analyse(counts, args.elo0, args.elo1, args.alpha, args.beta)
    p, t = r["pentanomial"], r["trinomial"]

    print(f"pairs {r['pairs']}  games {r['games']}  score {r['score']:.4f}")
    print(f"counts  LL {counts[0]}  LD {counts[1]}  DD/WL {counts[2]}  "
          f"WD {counts[3]}  WW {counts[4]}\n")
    print(f"{'model':14s} {'Elo':>9s} {'95% CI':>22s} {'+/-':>8s}")
    print("-" * 58)
    print(f"{'pentanomial':14s} {p['elo']:9.1f} "
          f"[{p['elo_ci95'][0]:8.1f},{p['elo_ci95'][1]:8.1f}] {p['elo_pm']:8.1f}")
    print(f"{'trinomial':14s} {t['elo']:9.1f} "
          f"[{t['elo_ci95'][0]:8.1f},{t['elo_ci95'][1]:8.1f}] {t['elo_pm']:8.1f}")
    print(f"\nnormalized Elo {p['normalized_elo']:+.5f}")
    print(f"variance ratio (trinomial/pentanomial) "
          f"{r['variance_ratio_trinomial_over_pentanomial']:.3f}")
    print(f"LOS {r['los']*100:.1f}%")
    s = r["sprt"]
    print(f"\nSPRT elo0={s['elo0']} elo1={s['elo1']} "
          f"alpha={s['alpha']} beta={s['beta']}")
    print(f"  LLR {s['llr']:+.3f}  bounds [{s['lower_bound']:.2f}, "
          f"{s['upper_bound']:.2f}]  -> {s['decision']}")

    if args.json:
        args.json.write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
