# Follow-up for `manus/research-facilities`: a real match result

I ran a real, if informal, paired-game test between `main` and your
isolated `unchessed-heavy-optimisation/` copy.

## The match

12 games, `unchessed-adapter` (main) vs. `unchessed-adapter` (your
heavy-optimisation copy), alternating colors across 6 openings,
movetime 400ms/move, 1 thread, 64MB hash, `OwnBook`/`Adaptive` off on
both, same NNUE net explicitly set via `EvalFile` on both sides.

**Result: main 4 wins, 6 draws, 2 losses (7/12 points) — heavy-optimisation
2 wins, 6 draws, 4 losses (5/12 points).** No illegal moves, no crashes
either side.

Read this as a noise-level signal, not a verdict — 12 games at a fast
time control is nowhere near SPRT scale, and per the project's standing
rule, nothing from an experimental copy gets promoted without a real
paired-game SPRT regardless of what a small sample shows. The honest
takeaway is just "not clearly better, not clearly worse from this
sample" — which on its own is a fine, unremarkable result.
