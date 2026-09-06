# Resilient channel independently replicated at scale (commits 5d43f6a + db48f6e)

Ported the fusion recalibration and the new resilient channel to main
(commit 02eb694) and ran a larger real clock-controlled test, matching
your own validation protocol but at 8x the sample size.

## Setup

Real UCI games vs the official Stockfish 19 Windows universal binary,
16 games per arm (32 total), 16 fixed opening prefixes, real
`wtime`/`btime` clocks (60,000ms start, 500ms increment per side, not
`movetime`), 80-ply cap, one thread, 64 MiB hash, `Adaptive=true`,
`OwnBook=false`, `AdapterTelemetry=true`,
`UCI_Opponent=- - human UnknownOpponent`. Telemetry parsed for the
first `persona_decision` record with `mode_after=FULL`, plus
observation/low-time-skip counts per game.

## Result

| Arm | Games | Mean first-Full ply | Median | Range | Total low-time skips |
|---|---:|---:|---:|---:|---:|
| Standard | 16 | 27.875 | 28 | 26-31 | 10 |
| Accelerated (resilient) | 16 | 23.875 | 24 | 22-26 | 4 |

**~4 plies (2 full moves) earlier confirmation on average, consistent
across all 16 games -- the ranges barely overlap** (standard's min is
26, accelerated's max is 26). This independently confirms your own
n=2 validation (26/24 vs 30/30) at a much larger sample and lands in
the same magnitude and direction. Per-game plies -- standard: 26, 29,
28, 27, 28, 31, 28, 29, 28, 27, 28, 29, 28, 27, 26, 27. Accelerated:
22, 25, 24, 23, 24, 25, 24, 25, 26, 23, 22, 25, 24, 25, 22, 23.

Secondary observation, not a primary claim: accelerated also had fewer
total low-time observation skips (4 vs 10) across the batch --
plausibly because confirming Full sooner means less of the game is
spent later running the more expensive Match-mode machinery under
time pressure. Worth checking directly rather than assuming.

## What this does and doesn't establish

This is a real, well-replicated positive result for the actual target
metric (detection latency against a genuinely strong opponent) -- the
first time in this whole investigation a change has shown a
consistent, non-noise-level improvement on that metric rather than a
null or mixed result.

It does not yet establish two things that would need to hold before
this is promotion-ready:

1. **False-positive rate.** Everything tested so far, by both of us,
   is against opponents that are genuinely strong (Stockfish 16/19).
   The resilient channel's looser tolerance for noisy play (that's the
   whole point of it) raises the real question of whether it also
   confirms too eagerly against opponents that only look inconsistent
   without actually being strong -- a decent human, or a Maia model.
   That's the safety side of this feature and it hasn't been tested by
   either of us yet.
2. **Strength/outcome SPRT specific to this path.** The earlier
   "no strength cost" mirror-match result (commit dfc6f86) was for the
   original streak-based accelerated path, not the resilient channel.
   Confirming Full sooner is only a win if it doesn't also produce bad
   early mode transitions in games where it matters.

## Suggested next steps, in order

1. Maia-model opponent test (the one both of us have now independently
   flagged as the real discriminating case) -- same protocol, watching
   specifically for premature Full confirmation against a model that
   plays inconsistently but isn't a strong-engine-level opponent.
2. A real paired-game strength SPRT with the resilient channel enabled,
   not just a latency measurement.

Source: `scripts/research/asymmetric_latency_test_v2.py` (this
reviewer's scratch harness, not yet committed -- happy to commit it if
useful) and the ported code at `unchessed-core/src/adapt.rs`/`uci.rs`
in commit 02eb694.
