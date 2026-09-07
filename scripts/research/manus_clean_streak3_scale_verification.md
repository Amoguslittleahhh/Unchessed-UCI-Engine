# Clean-streak-3 tightening verified at scale -- 20% false-confirmation (commit 2ed28f0)

Ported the `low_loss_streak>=2 -> >=3` tightening (commit `7bd40f0`)
and reran the same official-`maia3-uci` 20-games/arm protocol used for
every step in this series.

## Setup

Identical to the two prior runs in this series: real `maia3-5m` UCI
subprocess (temperature=1.0), `unchessed-adapter` built from `main` at
`2ed28f0`, real `wtime`/`btime` (60,000ms, 0 increment, 48-ply cap),
NNUE eval, `Adaptive=true`, `OwnBook=false`, two elo bands (1500,
2000), 10 games/arm/band, 40 games total, `tools/maia3_uci_false_positive_test.py`
unchanged.

## Result -- full series

| Gate | Games | Confirmed | Mean ply | Reason purity |
|---|---:|---:|---:|---|
| Legacy (AcceleratedDetection=false) | 20 | 18/20 (90%, this run) | -- | `legacy_clock`/`legacy_ceiling` |
| Clock-corroboration (f3cf694) | 20 | 13/20 (65%) | 24.6 | mixed clock/fusion |
| Resilient-only (99aad93) | 20 | 12/20 (60%) | 27.2 | resilient 12/12 |
| + fingerprint, streak>=2 (d077ac1) | 20 | 8/20 (40%) | 29.1 | resilient 8/8 |
| **+ fingerprint, streak>=3 (7bd40f0, this run)** | 20 | **4/20 (20%)** | 25.3 | resilient 4/4 |

Zero low-time skips across every game in every run in this series.
Standard-arm confirm rate again shows real run-to-run variance (90%
this time vs 100%/75% in prior runs) from `maia3-5m`'s own unseeeded
sampling -- the legacy path is untouched by any of this series'
changes, so the accelerated-vs-same-run-standard gap (90% vs 20%, a
70-point real spread) is the number that matters.

## What this establishes

Three consecutive real tightenings on the same real official-Maia3
protocol, each independently re-verified at 20-games/arm:
100%-ish -> 65% -> 60% -> 40% -> 20%. This is now a genuinely
reasonable false-positive rate for a feature meant to fire only
against real strong engines, and the trend has been monotonic across
every step measured so far.

Reason purity holds at every step: all 4 confirmations in this run
were `legacy_accelerated_resilient` and nothing else.

## Open question

20% is close to what most people would call "acceptable" for this
kind of feature, but it's still not zero, and the corroborating
strength side (does this still confirm quickly enough against a real
strong engine, i.e. Stockfish 19) hasn't been re-checked since the
streak requirement changed -- the resilient channel's own
consistency-fingerprint requirement (3 consecutive near-perfect moves)
could plausibly delay genuine detections too, not just false ones.
Worth a matched Stockfish 19 run at the same scale before calling this
series done, to make sure the false-positive gains aren't coming at
the cost of detection latency against real engines.

Source: `tools/maia3_uci_false_positive_test.py` (unchanged) and the
ported tightening at `unchessed-core/src/adapt.rs` in commit `2ed28f0`.
