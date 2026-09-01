# Live Elo detector: high-level misfires and ecosystem integration (2026-09-01)

The adapter’s real-time opponent-Elo model (`OpponentModel` in `adapt.rs`)
feeds **persona**, **book troll gating**, and **FULL vs MATCH**. Three
misfires showed up when the rest of the stack (Maia labels, UCI_Opponent,
PersonaState, MATCH at 2200+) is actually wired together.

## Misfires found

| Path | Old behaviour | Who it hit | Integration damage |
|---|---|---|---|
| `is_computer` ⇒ `engine_suspect` | **any** computer, including Maia 1600 | Maia / human-like engines | MATCH never ran vs the opponent we train to imitate — always FULL |
| Ceiling tell `weight≥10 && mean≥2450` | 10 clean moves from the 1500 prior | 2200–2500 humans in theory | FULL vs a titled player who wanted MATCH |
| Clock tell `suspicion≥3` in the opening | 3 instant book/premoves | blitz 2000+ | FULL after move 6 |
| Climbing `var_accum` veto | path 1500→2700 *is* high variance | real engines | ceiling tell never fired when we wanted it |

At high human ratings MATCH is already near-ceiling (`target_elo =
estimate+60`, `max_loss` floor 12 cp, HeuristicPrior off above 2200).
Flipping to FULL therefore buys almost no strength and **does** kill
troll-book and the human-policy path. So a false FULL is a persona bug,
not a free Elo.

## Contract after the fix

1. **GUI computer, strong** (`mean ≥ 2400`, Stockfish/Rubi/…): FULL immediately. Book troll off.
2. **GUI computer, human-like** (Maia 1600): **not** suspect. MATCH at the seeded Elo.
3. **GUI human, any declared Elo**: ceiling tell **off**. Clock tell still on (cheater with instant middlegame moves). MATCH tracks live cp-loss.
4. **Anonymous**: ceiling tell only after **16** observations, weight ≥ 11, mean ≥ 2500, **12-move low-loss streak** (≤40 cp). Clock tell only after sample ≥ 8, **4** instant-strong middlegame replies.
5. **Opening observations** (first 8) have difficulty weight ×0.5 so a clean Ruy does not pin 2900 by move 8.
6. PersonaState still emergency-FULL on a true suspect. Adaptive stays on.

## Tests (stdlib replica + 200 games/band)

`python3 -m unittest tools.test_elo_detector -q`

Declared-human bands 1200–2500, 24 moves, realistic cp-loss and think times:
**0/200 engine flags per band** (`artifacts/elo-detector-misfire.json`).

Also: Maia not FULL, Stockfish FULL, 10 opening greens not flagged, 7
opening premoved instants not flagged, 4 middlegame instants after sample
8 flagged, 20 anonymous perfects flagged.

Cargo tests in `adapt.rs` cover the same contract (need rustc).

## What still needs a live SPRT

Anonymous cheat detection vs a 2500 human who happens to play 16 perfect
moves. That is now *later* (move 16+) and streak-gated; a real game vs
IMs on the clock is the remaining gate. Book troll vs Maia should now
follow the 1600 estimate instead of being hard-locked off.
