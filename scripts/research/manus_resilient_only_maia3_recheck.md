# Resilient-only restructure re-tested at scale against Maia-3 -- result contradicts the small-sample claim (commit 99aad93)

Ported the strict resilient-channel-only restructure to main (commit
`efc08d0`) and reran the exact same 40-game Maia-3 protocol used for
the earlier 374b6dd result, so this is a direct, apples-to-apples
comparison, not a new setup.

## Setup

Identical to the 374b6dd run: real Maia-3
(`mcognetta/simple-maia3-inference` ONNX export, onnxruntime,
temperature-1 sampling), `unchessed-adapter` built fresh from `main`
at `efc08d0`, real UCI subprocess, real `wtime`/`btime` (60,000ms, 0
increment, 48-ply cap), NNUE eval, `Adaptive=true`, `OwnBook=false`,
`UCI_Opponent=- - human UnknownOpponent`. Two elo bands (1500, 2000),
10 games/arm/band, 40 games total, same openings and seeds as before
(`tools/maia3_false_positive_test.py`, unchanged).

## Result

| Arm | Games | Full-confirmed | Confirmation ply (mean / range) | Reason |
|---|---:|---:|---:|---|
| Standard (legacy) | 20 | 20/20 (100%) | 10.0 / 6-19 | `legacy_clock` (unchanged, as expected) |
| Accelerated, clock-corroboration fix (f3cf694, prior result) | 20 | 13/20 (65%) | 24.6 / 19-36 | `legacy_clock` 11, `legacy_accelerated_fusion` 5 |
| Accelerated, strict resilient-only (99aad93, this result) | 20 | 14/20 (**70%**) | 25.7 / 22-34 | `legacy_accelerated_resilient` 14/14 -- exclusively |

The semantic part of the restructure worked exactly as designed: every
single confirmation now routes through `legacy_accelerated_resilient`
and nothing else, closing the "correlated evidence promotes without a
real resilient confirmation" gap the last result identified. But **the
false-positive rate did not improve on this sample -- it is
marginally worse** (70% vs 65%), and confirmation timing is
essentially unchanged (25.7 vs 24.6 mean ply). The resilient channel
itself, once it's the only path, turns out to be about as easy to
satisfy against Maia-3 as the correlated clock+fusion combination was.

## This contradicts your own validation

Your `maia_false_positive_complete_fix_20260907.md` reported the
official `maia3-uci` 5M CPU runtime confirming in only 1/5 strict-mode
games (20%) at ply 36, a large drop from the 5/5 legacy baseline. My
40-game run on the simplified ONNX export shows 14/20 (70%) at a
similar mean ply (~26) to before the restructure. Both samples are
small enough that the gap could just be model-pipeline variance (your
own doc flags this explicitly: no one-ply opponent-response ranking in
the ONNX export vs the official UCI runtime's ranking), but a
20-points-of-percentage gap between "closes the problem" and "barely
moves it" on the exact same feature is large enough that it needs
resolving before either number is trustworthy on its own. Possibilities,
roughly in order of plausibility:

1. **Real model-pipeline difference.** The official `maia3-uci`
   binary's opponent-response ranking may make its move choices
   noticeably less "engine-like" than the simplified export's raw
   temperature sampling, so it clears the resilient bar less often.
   If true, the simplified ONNX export is simply not a faithful proxy
   for the false-positive question, and any future false-positive
   work should standardize on the official binary.
2. **Small-sample noise on your end** -- n=5 is a thin sample for a
   70/30-ish split; a couple of games either way would move your
   1/5 substantially.
3. **A real difference in test protocol** (elo settings, opening
   set, clock) beyond what's documented in your write-up -- worth
   comparing configs directly if useful.

## Suggested next step

If you have the official `maia3-uci` binary available, the cleanest
resolution is running it through `tools/maia3_false_positive_test.py`
(committed, argparse-based, drop-in for any UCI-speaking opponent --
would need a small adapter shim since the tool currently drives Maia
via the ONNX model directly rather than a UCI subprocess, happy to add
a `--maia-uci-path` mode if that's useful) at a size closer to 20/arm
rather than 5/arm, so both of us are looking at the same runtime at a
comparable sample size.

Source: `tools/maia3_false_positive_test.py` (unchanged from the
374b6dd run) and the ported restructure at
`unchessed-core/src/adapt.rs` in commit `efc08d0`.
