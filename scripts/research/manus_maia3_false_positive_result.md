# Real Maia-3 false-positive test of the clock-corroboration fix (commit f3cf694, ported from f23b915)

Ported `accelerated_clock_corroborated()` to main and ran a real batch
against the actual Maia-3 model to independently check the safety
question your own 4-game run flagged as unresolved.

## Setup

Real Maia-3 (`mcognetta/simple-maia3-inference` ONNX export, the same
mirror your `maia3_simplified.onnx` uses) driven directly via
onnxruntime with temperature-1 policy sampling -- Maia has no UCI
binary, so its moves are sampled from the model exactly as
`tools/selfplay_elo_mixer.py` does, not simulated. `unchessed-adapter`
built fresh from `main` at `f3cf694` (includes the ported fix), real
UCI subprocess, real `wtime`/`btime` (60,000ms, 0 increment, 48-ply
cap), `Adaptive=true`, `OwnBook=false`, NNUE eval loaded,
`UCI_Opponent=- - human UnknownOpponent`.

Two elo bands (`elo_self=elo_oppo=1500` and `2000`, mid-club and
strong-club/near-expert -- the range your write-up called out as the
real discriminating case), 10 games/arm/band, 40 games total,
alternating colors, fixed opening prefixes.

## Result

| Arm | Games | Full-confirmed | Confirmation ply (mean / range) |
|---|---:|---:|---:|
| Standard (legacy) | 20 | 20/20 (100%) | 10.0 / 6-19 |
| Accelerated (with fix) | 20 | 13/20 (65%) | 24.6 / 19-36 |

The fix is a real, measurable improvement, not a full close:

- **False-positive rate drops from 100% to 65%.** 7/20 accelerated
  games correctly never confirmed Full against Maia-3 at all, where
  the legacy path always did.
- **Confirmation is much later when it still happens** -- mean ply
  24.6 vs 10.0, and the ranges barely overlap (legacy's max is 19,
  accelerated's min is 19).
- **The corroboration gate is doing its job as coded** -- in a
  smoke-test game inspected directly, `legacy_clock` only fired once
  `estimate_elo>=2450`, `samples>=10`, and
  `accel_resilient_streak>=2` were all simultaneously true, never on
  a bare clock tell. The remaining false positives are not a bug in
  the gate; they happen because Maia-3's actual played move quality,
  as judged by Unchessed's own cp-loss-based Elo estimator, genuinely
  clears 2450+ often enough in real games. The simplified ONNX export
  has no one-ply opponent-response ranking (documented in your own
  `selfplay_elo_mixer.py` docstring as an approximation of the
  official `maia3-uci`), so its temperature-sampled moves may look
  stronger to a cp-loss estimator than the official binary would.
- **The dedicated resilient-channel reason never independently
  fired** across all 40 games (`legacy_accelerated_resilient`: 0/40).
  Every accelerated confirmation came through the now-gated
  `legacy_clock` path (11/20) or `legacy_accelerated_fusion` (5/20,
  overlapping with clock in some games) -- worth knowing when reasoning
  about which channel is actually driving detections against
  human-like opponents in practice.
- Elo band (1500 vs 2000) did not show a qualitatively different
  pattern; both false-positive at similar rates.

## What this does and doesn't establish

This closes about a third of the false-positive gap you flagged, with
a real, non-trivial sample (40 games vs your 4) and confirms the fix
works exactly as designed -- but it does not make
`AcceleratedDetection` false-positive-safe against Maia-3. 65% is
still a high rate for a feature that's meant to only fire against
genuine strong engines. Two live questions before this could be
promotion-ready:

1. Whether the remaining false positives are an artifact of the
   simplified ONNX export specifically (no opponent-response
   ranking) or would also occur against the official `maia3-uci`
   binary with full history/ranking -- worth testing if that binary
   is available on your end.
2. Whether raising the corroboration bar further (e.g. requiring the
   resilient channel to have actually fired, not just its score/streak
   fields to be high enough) would close more of the gap without
   pushing genuine-engine detection latency back up.

Source: `tools/maia3_false_positive_test.py` (this run's harness,
committed) and the ported fix at `unchessed-core/src/adapt.rs` in
commit `f3cf694`.
