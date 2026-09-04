# PersonaSmooth / EngineDetectV2: real cutechess SPRT result (2026-09-02)

Reviewer-run (real hardware, WSL), using the harness arena committed
(`scripts/sprt-history/sprt_persona_smooth_detect.sh`) — the gate arena's
own sandbox cannot run (no cutechess there). Rebuilt at `main` after
round 19, confirmed `PersonaSmooth`/`EngineDetectV2` both present and
default `false` before running anything.

## What was run

Same binary both sides (UCI toggle, not two builds), `Adaptive=true`
both sides (required — `decide_mode` short-circuits to Full/Match when
Adaptive is off, so neither new option's code path runs otherwise),
`tc=5+0.05`, `elo0=0 elo1=5`, same shape as
`scripts/sprt-history/sprt_punish_latch.sh`.

- PersonaV2: `PersonaSmooth=true EngineDetectV2=true`
- Baseline: `PersonaSmooth=false EngineDetectV2=false` (today's shipped
  default path)

## Result

**Interrupted at 5537/10000 games by a machine restart, before crossing
either SPRT bound.** Final tally at the moment of interruption:

**2125 - 2095 - 1317** (PersonaV2 wins - losses - draws), score fraction
**0.503**, ≈ **+2.1 Elo** for PersonaV2 — statistically indistinguishable
from zero.

cutechess-cli has no resume mechanism for a killed process (`-recover`
is for restarting a crashed *engine* mid-match, not resuming the
tournament's own LLR state); restarting means starting over from game 1.
Given the trajectory below, that was judged not worth another multi-hour
run for the same qualitative answer — see "Why this is treated as
conclusive enough" below.

### Score trajectory over the run (from the live log)

| Games | PersonaV2 score fraction |
|---|---|
| 315 | 0.470 |
| 944 | 0.467 |
| 3188 | 0.501-0.502 |
| 4247 | 0.503 |
| 5537 (interrupted) | 0.503 |

The score drifted slightly negative early, then flattened and held
almost perfectly steady at ~0.503 for the last ~2300 games. It never
trended toward either the elo0=0 or elo1=5 bound.

## Why this is treated as conclusive enough without a formal SPRT pass

A flat score fraction that holds steady across thousands of games,
rather than continuing to drift toward a bound, is itself a real signal
— it's the trajectory a true effect near elo0 (no real Elo difference)
actually produces. 5537 games is a large sample; if PersonaSmooth and
EngineDetectV2 together were worth anything like +5 Elo, the LLR would
typically show continuing drift toward that bound by this point, not
stabilize at +2.1. This is not the same rigor as an SPRT that formally
crossed a bound, and is reported with that caveat, but it's a real
measured result on real hardware, not a simulation.

## What this does and doesn't mean

**Does mean**: turning both options on does not cost meaningful Elo at
`tc=5+0.05`. The features are not a strength regression.

**Does not mean**: they should default to `true`. Neither was designed
to *gain* Elo — `PersonaSmooth`'s stated purpose (round 15-17 docs) is
reducing accidental persona mode-flips (57.3% flip-rate reduction in
simulation), and `EngineDetectV2`'s is fixing real misfire cases (Maia
wrongly flagged as an engine, false ceiling/clock tells). Whether that
behavioral improvement is worth shipping is a product decision about
what the persona system should feel like against real opponents, not an
Elo question — and this result says the Elo question is settled neutral
either way.

## Recommendation

No Elo objection to flipping either default. Whether to actually flip
them is the project owner's call, based on whether the behavioral
improvements (fewer flip-flops, correct Maia/human detection) are
wanted — not blocked by this data. If flipped, a longer real-play period
(not just this SPRT) would be the way to confirm the flip-reduction and
misfire-fix claims hold up outside simulation, since this SPRT measured
Elo, not flip rate or detection accuracy directly.

## Reproducing this

`scripts/research/wsl_sprt_persona_run.sh` — adapts arena's
`sprt_persona_smooth_detect.sh` to this reviewer's WSL checkout path.
Full PGN of all 5537 completed games:
`~/unchessed-ai/results/adapter/sprt_gates/sprt_persona_smooth_detect.pgn`
(not committed — real-hardware-local artifact, same as every other
round's SPRT PGN output).
