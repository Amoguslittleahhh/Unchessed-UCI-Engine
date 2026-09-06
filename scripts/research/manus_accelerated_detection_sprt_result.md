# AcceleratedDetection built and real-SPRT tested (commit dfc6f86)

Built the ablation you proposed in the opponent-detection-latency paper
(11683a8) and ran a real paired-game SPRT, not a simulation. Summary
below; happy to run the follow-up experiments this points to if
useful.

## What was built

New UCI option `AcceleratedDetection` (default off), implemented in
`unchessed-core/src/adapt.rs`/`uci.rs` (commit b5e5b32). It's a second,
independent confirmation path (`SuspectReason::LegacyAcceleratedCeiling`)
alongside the existing legacy weight>=10.0/mean>=2450.0 ceiling, which is
unchanged. It fires only on:

- `samples >= 8` (post-opening, matches the existing opening gate elsewhere)
- `mean >= 2550`, `confidence <= 220`, `volatility <= 300` (stays below
  `trend()`'s own 380 erratic cutoff)
- a non-negative single-step trend
- **two consecutive** qualifying observations, or one plus a live clock tell

Verified in isolation first: with a realistic 0.4 difficulty-weight per
observation, this path confirms 9 observations sooner than the legacy
ceiling (obs 21 vs obs 30) in that synthetic scenario -- real numbers
depend on actual gameplay, which is what the SPRT below tests.

## Real SPRT result

1000-game real cutechess-cli match (not simulated): same main binary
both sides, same explicit NNUE, `Adaptive=true`, `tc=5+0.05`,
`Baseline` (option off) vs `Accelerated` (option on), `elo0=0 elo1=5
alpha=beta=0.05`. Hit the 1000-game round cap without reaching an SPRT
bound (llr stayed at -0.434, well inside the [-2.94, 2.94] continuation
region).

**Elo difference: -3.8 +/- 16.7, LOS 32.7%, DrawRatio 39.7%.** No
measurable strength difference either direction. Zero illegal moves or
crashes on either side across all 1000 games.

## Why this is expected, not a dead end

This was a mirror match: both sides are the same engine, so they
correctly detect each other as a strong computer quickly regardless of
which side has `AcceleratedDetection` on -- final game score was never
going to isolate this feature's actual target metric, which is
**detection latency** (moves until `Mode::Full`), not raw strength. The
test does establish something real, though: **no strength cost** from
enabling the option, across 1000 real games with zero stability
issues. That's a legitimate prerequisite before promoting anything,
even if it doesn't confirm the latency benefit.

## Suggested next step

A test that can actually measure the claimed benefit needs an
asymmetric setup: `Accelerated` vs a *distinct* strong opponent (not a
mirror of itself), with telemetry capturing `mode_before`/`mode_after`
transitions, comparing moves-to-`Full`-confirmation with the option on
vs off. That's a direct test of the paper's own primary metric rather
than a proxy through final score.
