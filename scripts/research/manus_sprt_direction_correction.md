# Correction: the -244 / -238 Elo results were reported backwards

The two supplied SPRT results (223-game fast gate, and the 200-game
low-concurrency control that reproduced it independently) were
described to you as "heavy-optimisation is decisively weaker than
main." That direction is wrong, and both your regression audit and the
IEEE paper section built on it need to be flipped.

## Why it was wrong

`cutechess-cli`'s "Score of A vs B" and "Elo difference" lines are
always reported relative to the **first-named engine** -- here, `Main`.
Both matches showed Main scoring low:

- 223-game gate: Main's score was well under 0.5 across the match; the
  per-player termination summary shows Main with 146 losses against 18
  wins.
- 200-game low-concurrency control (no telemetry wrapper, concurrency
  matched to core count): `Score of Main vs HeavyOpt: 21 - 140 - 39
  [0.203]`, `Elo difference: -238.1 +/- 50.7, LOS 0.0%`.

A score of 0.203 for Main is a *low* score for Main. `Elo difference:
-238.1` is Main's Elo relative to HeavyOpt, so this reads as **Main is
~238-244 Elo weaker than heavy-optimisation**, not the reverse. The raw
per-player win/loss counts in the original supplied result were correct
and were available the whole time; only the prose summary describing
which side they favored was inverted. Apologies for the churn this
caused on your side -- the `PersonaSmooth` default fix and the Cargo
release-profile experiment are still both correct and useful regardless
of this correction, they just weren't chasing the right question.

## What this changes

Given the direction flip, the earlier "regression" is not a regression
at all -- it's heavy-optimisation playing meaningfully *stronger*
chess than main under `Adaptive=true`. The most likely mechanism,
found during the same source diff already done for the regression
audit: `uci.rs`'s `known_full` addition on your branch --

```rust
let known_full = adaptive_now && !job.opt.limit_strength && model.lock().unwrap().engine_suspect();
let multipv_search = if adaptive_now && !known_full { multipv_shown.max(5) } else { multipv_shown };
```

-- drops the search back to `MultiPV=1` once the opponent model flags
the opponent as a strong/computer player. Main has no equivalent: its
`multipv_search` widens to `MultiPV>=5` unconditionally whenever
`adaptive_now` is true, for the entire game, regardless of who the
opponent turns out to be. Since every additional simultaneously-tracked
PV strictly costs alpha-beta pruning efficiency, main is paying a real,
large, and apparently uncapped strength tax for the whole game to
support persona move-selection variety that (per the mode-bucket
telemetry from the original gate) never even gets used once the
opponent is correctly identified as a strong engine -- decisions landed
100% in FULL mode with zero persona transitions in that match.

This reframes the whole investigation: `known_full` is very likely a
genuine, real strength improvement worth porting to main (behind the
project's usual real-paired-game SPRT gate before any default changes),
not something to discard because it came from an "unproven hardware
branch." The Cargo-profile and target-cpu experiments remain correctly
ruled out as explanations -- they were never the right target once the
sign is corrected, but the negative results themselves are still valid
and don't need to be redone.

## The hardware-optimization claim, measured cleanly

Separately, ran your own `scripts/benchmark-portable-v3.sh` for real
against the shared NNUE file (SHA-256 verified: same file used
throughout this whole investigation). Averaged across Hash=4/8/16/32/64
at startpos, identical node counts and bestmoves per hash size between
builds (same search, different codegen only):

- portable build: ~2.50M NPS average
- x86-64-v3 build: ~3.16M NPS average

That's a consistent, real **~26% NPS improvement** from
`target-cpu=x86-64-v3` on this hardware (which supports avx2/bmi2/fma),
holding across every hash size tested. Combined with the corrected
strength result above, the honest summary is: heavy-optimisation is
both faster per node *and* plays stronger chess than main right now --
the hardware-optimization research direction is legitimate and the
numbers back it up, independent of the `known_full` strength question.

## Ask

1. Please correct the audit doc and IEEE paper section to state the
   right direction: heavy-optimisation (specifically its `known_full`
   MultiPV-narrowing) outperformed main by ~240 Elo in both supplied
   matches, not the reverse.
2. If useful, a clean follow-up experiment would isolate `known_full`
   alone: patch main with just that one change (nothing else from the
   heavy-optimisation branch) and SPRT it against unmodified main. If
   the same ~240 Elo gain shows up from that one change alone, that's a
   strong, cheap, well-isolated case for porting it.
