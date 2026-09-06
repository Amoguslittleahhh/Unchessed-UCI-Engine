# Real-game finding: opponent detection took 17 moves against a genuinely strong, consistent engine

From a real 1|1 bullet game (`Unchessed Game Adapter` vs `Dragon by
Komodo`, lost 0-1), analysed from the adapter's own UCI log (not a
simulation).

## What happened

For the first 17 moves, every opponent observation classified Dragon
by Komodo as **"erratic (sandbagging?)"**, keeping the persona in
`Mode::Match` (deliberately human-plausible play targeting an
estimated ~2600 Elo) against an opponent that was, in fact, a
consistently strong engine the whole time:

```
move 8:  opponent~2411 (+/-381), steady
move 9:  opponent~2468 (+/-364), trending up
move 10: opponent~2469 (+/-318), erratic (sandbagging?)
move 11: opponent~2531 (+/-298), erratic (sandbagging?)
move 12: opponent~2580 (+/-278), erratic (sandbagging?)
move 13: opponent~2580 (+/-248), erratic (sandbagging?)
move 14: opponent~2616 (+/-233), erratic (sandbagging?)
move 15: opponent~2623 (+/-211), erratic (sandbagging?)
move 16: opponent~2650 (+/-199), erratic (sandbagging?)
move 17: opponent~2634 (+/-183), pinned at measurement ceiling -- engine suspected
move 18: persona MATCH -> FULL (eval -19 cp, opponent ~2634)
```

The estimate itself climbed steadily and correctly (2411 -> 2650) the
whole time -- the model was tracking real Elo accurately. What lagged
was the *classification* label flipping from "erratic (sandbagging?)"
to "engine suspected," which is what actually gates the
`Mode::Match -> Mode::Full` transition (via `known_full` /
`engine_suspect()`, the same mechanism from the main-vs-heavy-optimisation
SPRT investigation). 17 moves of a consistently-strong, non-erratic
opponent were read as possible sandbagging before the model committed.

## Why this is worth a look, not just a curiosity

In this specific game it didn't cause the loss -- the critical error
(16.Qh4?, verified separately against move-by-move engine eval) was
identical to what the engine would have played in `Mode::Full` at the
same search depth, so the mode lag was inert here. But the mechanism
that makes `known_full` valuable (dropping the wasted `MultiPV>=5`
persona overhead once a strong opponent is confirmed) doesn't
kick in until the classification flips -- and here that took 17 moves
against an opponent whose Elo estimate was already accurate and
climbing the entire time. If a critical decision happens to fall in
that window against a real strong opponent, the delay could matter
more than it did in this game.

**Ask, concretely:** is 17 moves of "erratic (sandbagging?)" against a
steadily-estimated 2400+ opponent the intended sensitivity, or would a
faster confirmation path (e.g. weighting sustained high-confidence
Elo estimates more than the erratic/sandbagging heuristic once the
estimate itself has stopped moving) be worth a controlled test? Real
log evidence for this is attached in spirit above; the raw UCI log
and PGN are `2026.09.06_Unchessed Game Adapter - Dragon by Komodo.pgn`
and the corresponding CSV, both local to this reviewer -- happy to
relay the full files if useful.
