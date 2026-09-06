# The 0.25 overall score vs Stockfish (commit cd2ea4a) isn't a strength result -- it's the persona working as designed

Checked the new `elo_matrix_real` benchmark (0 wins / 3 draws / 3
losses across 6 games, `Adaptive=true`, `movetime=30ms`, Stockfish 19
capped at 1320/1800/2400). Before reading this as any kind of strength
signal, worth knowing what `Adaptive=true` actually does with no prior
observations of the opponent.

## Reproduced directly

```
$ printf 'uci\nsetoption name Adaptive value true\nsetoption name OwnBook value false\nisready\nposition fen <midgame fen>\ngo movetime 30\nquit\n' | unchessed-adapter
info string [Unchessed] mode=MATCH opponent~1500 (+/-450) eval 0 cp: human-plausible at ~1560 (loss 0 cp)
bestmove a2a3
```

With zero observations of the opponent, the persona defaults to
`Mode::Match` targeting **~1500 Elo**, not full engine strength. It
stays there unless/until the opponent model accumulates enough evidence
to flag them as a strong/computer-like player (`engine_suspect()`),
which is exactly the mechanism `known_full` (commit 63101a8, ported to
main in `0df9a86`) hooks into.

## What this means for the benchmark

`Adaptive=true` against Stockfish capped at 1320/1800/2400 isn't
testing "can the engine beat these opponents" -- it's testing "does the
persona correctly imitate ~1500 Elo play," which is a different
question with a different expected answer. Losing to the 1800 and 2400
caps and roughly breaking even at 1320 is the *expected* outcome of a
persona deliberately targeting ~1500, not evidence the engine is weak.
For an actual strength/regression signal, `Adaptive=false` is the
correct setting -- that's what this reviewer's own absolute-Elo ladder
used (~2940 Elo estimate vs. real Stockfish 18 at movetime=1000ms,
see `scripts/research/elo_ladder_vs_stockfish_result_20260905.md`).

Two benchmarks measuring two different things (raw strength vs.
persona-target fidelity) aren't in tension with each other -- just
worth labeling which one a given result actually is before drawing
conclusions from it. It might be worth adding a note to the benchmark
README distinguishing "Adaptive=false: raw strength check" from
"Adaptive=true: persona-target fidelity check," since the current
README's framing ("performance degraded as the opponent cap increased")
reads like a strength claim even though it explicitly disclaims
statistical significance elsewhere in the doc.

The `movetime=30ms` control is also worth a second look independently
-- it's tight enough that persona/opponent-modeling overhead could
matter, separate from the Match-mode-by-design explanation above -- but
the mode-selection behavior is almost certainly the dominant factor
here, not the time budget.
