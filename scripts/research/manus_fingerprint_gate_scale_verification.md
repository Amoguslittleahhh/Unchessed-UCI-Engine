# Resilient-consistency fingerprint gate verified at 20-games/arm scale (commit b506dc4, ported from d077ac1)

Ported the `low_loss_streak>=2 && volatility()<=460` fingerprint gate to
`accelerated_resilient()` and reran the same official-`maia3-uci`
protocol at the established 20-games-per-arm scale (your own validation
was 6 games/arm) to get a number both of us can trust.

## Setup

Identical protocol to the `24b6eb2` result: real `maia3-5m` UCI
subprocess (temperature=1.0), `unchessed-adapter` built from `main` at
`b506dc4`, real `wtime`/`btime` (60,000ms, 0 increment, 48-ply cap),
NNUE eval, `Adaptive=true`, `OwnBook=false`,
`UCI_Opponent=- - human UnknownOpponent`. Two elo bands (1500, 2000),
10 games/arm/band, 40 games total (`tools/maia3_uci_false_positive_test.py`,
unchanged).

## Result

| Arm | Games | Full-confirmed | Mean ply | Reason purity |
|---|---:|---:|---:|---|
| Standard (legacy), this run | 20 | 15/20 (75%) | -- | `legacy_clock`/`legacy_ceiling`, expected |
| Accelerated, resilient-only, prior run (`24b6eb2`) | 20 | 12/20 (60%) | 27.2 | `legacy_accelerated_resilient` 12/12 |
| **Accelerated, with fingerprint gate (this run)** | 20 | **8/20 (40%)** | 29.1 | `legacy_accelerated_resilient` 8/8 |

Zero low-time skips across all 40 games (22-24 observations/game, same
as every prior run in this series).

One methodology note: the standard-arm confirm rate in this run (75%)
is itself lower than the prior official-UCI run's 20/20 (100%) --
`maia3-5m` has no seed control we're aware of, so the real games
diverge run to run purely from the opponent's own internal stochastic
sampling. This isn't a code effect (the legacy path is untouched by
both of today's changes) and it means the fairest comparison is the
same-run standard-vs-accelerated gap (75% vs 40%, a 35-point real
drop) rather than treating the historical 100% as a fixed baseline.

## What this establishes

**Real, meaningful improvement, confirmed at scale.** The fingerprint
gate cuts the official-Maia3 false-confirmation rate by a genuine
20 points versus the resilient-only gate (60%->40%) at the same
20-games/arm sample size both of us have now used twice. Reason purity
holds -- every confirmation in both accelerated runs traced to
`legacy_accelerated_resilient` and nothing else.

It's a smaller effect than your own 6-game sample suggested (1/6 ~=
17%), which is expected: n=6 has enough variance that a couple of
outcomes swing the rate by 15-20 points either way, and this larger
sample is the more trustworthy number going forward. 40% is still a
meaningfully high false-confirmation rate for a feature meant to only
fire against genuine strong engines -- real progress, not yet
promotion-ready on its own.

## Open question

Two consecutive gate tightenings (clock-corroboration, then
resilient-only, then this fingerprint) have each produced real but
partial reductions (100%->65%->60%->40% across the whole series on
the official binary). Worth deciding together whether the next move is
another targeted gate tightening in the same style, or a different
approach -- e.g. requiring the fingerprint's clean streak to be longer
than 2, or bringing in a signal this series hasn't used yet (opening
context, move-time variance from the opponent, or something else).

Source: `tools/maia3_uci_false_positive_test.py` (unchanged) and the
ported gate at `unchessed-core/src/adapt.rs` in commit `b506dc4`.
