# Unchessed AI — master research & engine-strength brief for `manus/research-facilities`

This consolidates every prior prompt sent to this branch (the rustc
trial, the extended research brief, the label-noise update, and both
engine-improvement topic lists) into one file, corrected where later
verification changed the picture. Treat this as the current source of
truth; the earlier separate files can be ignored from here on.

## Credit budget — read first

**3,422 credits total, for everything below, indefinitely** — not per
task. Work cheapest-and-most-diagnostic first. **Before starting
anything in Tier 2 or Tier 3, post a short plan and a credit-cost
estimate and wait for it to be acknowledged.** If unsure whether
something is cheap or expensive, treat it as expensive and ask. Silence
is not consent to spend.

## Repo and branches

Repo: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine

- **`main` is read-only.** Never push to it, never merge into it.
- **Work on `manus/research-facilities`** (this branch's current name —
  it was `manus/rustc-bootstrap-trial` earlier, same history). It
  already has substantial, independently-verified work on it: the
  rustc-bootstrap script, five Tier 1 investigation docs, a Tier 2.2 toy
  RL calibration prototype, and (from the reviewer side) a real
  paired NNUE label-noise measurement. `cargo test --workspace --release`
  was last confirmed at 123/123 passing, three named parity gates
  passing, after a real Windows portability bug in
  `tools/nnue_relabel_existing.py` (directory fsync failing outright on
  Windows) was found and fixed.

## What's already established

### The three live subsystems

1. **NNUE evaluator** (`unchessed-core/src/nnue.rs`, `tools/train_nnue.py`)
   — the shipped default. Best v4 retrain is −155.6 ± 47.7 Elo behind
   the shipped v3 default at 108M positions.
   **Updated finding, supersedes the round-14 assumption**: round 14's
   simulation-based analysis assumed a ~50-56cp label-noise floor from
   the 5000-node HCE search and used that to argue the NNUE plateau was
   a labeling problem. A real, paired, live-generation measurement
   (`docs/nnue-label-noise-real-measurement.md`) — sidestepping the
   real blocker your own `08-label-noise.md` correctly identified (the
   existing shards can't be legally re-searched, they're missing
   castling rights and en passant) by generating fresh paired labels
   during PGN replay instead — found **MAE 17-22cp, Pearson 0.93-0.98**
   across four PGN sources and two depth multipliers (10x, 20x). That's
   well below the assumed floor. **Update your `08-label-noise.md` and
   the label-noise row in `11-tier1-synthesis.md` to reflect this: the
   stream moves from "defer, blocked" to "reopened, real measurement
   contradicts the working assumption."** This also means any reasoning
   in `06-rl-selfplay.md` that leaned on "label noise is the real
   problem, so RL doesn't help" should be re-checked — your other,
   independent reasons for deferring self-play RL (no MCTS/PUCT, no
   policy/value ABI, no measured NNUE throughput) are unaffected and
   still stand on their own.
2. **Unarchitectured Metal** — experimental transformer move-ordering
   hint. Real speed work exists (~7.5ms/call forward pass); the hint
   itself has never trended positive across four SPRT batches. Must
   stay `UnarchitecturedHint=false`.
3. **Persona/adapter** (`adapt.rs`) — `PersonaSmooth`/`EngineDetectV2`,
   both default `false`. One real SPRT found Elo-neutral (≈+2.1cp,
   noise, interrupted before a formal bound but stable for ~2300 games).
   Behavioral claims (flip-rate reduction, misfire fixes) remain
   simulation-only.

### Standing rules (non-negotiable, apply to everything below)

1. No search/live-behavior default flip without a real paired-game
   cutechess SPRT. Simulation and unit tests are evidence for a
   writeup, never a substitute.
2. The three named parity tests must keep passing after any
   `unarchitectured_metal_runtime.rs` change: `start_position_matches_python_reference`,
   `midgame_position_matches_python_reference`,
   `position_to_input_matches_hand_built_start_position`.
3. State what you verified vs. assumed. "Should work" is not "I ran it."
4. Defaults don't move without evidence.
5. Report negative results as plainly as positive ones.

### What's already implemented — verify before proposing these as new

Confirmed present: reverse futility pruning, null-move pruning, late-move
reductions, aspiration windows, ProbCut (optional SEE filter), plain
futility pruning, killer-move and history-heuristic ordering, **SEE-based
capture ordering used broadly in move ordering already** (not just
MVV-LVA — `move_score` in `search.rs` orders every capture/promotion by
its actual SEE value, in both main search and qsearch, correcting an
earlier note that called this "MVV-LVA ordering"), repetition detection
(bounded scan tested against a full scan), Lazy SMP multi-threaded
search sharing one lock-free TT, exact `go nodes N` enforcement, TT
prefetching, situational time management (sharp-vs-quiet soft-time
scaling, tested panic-mode fallback down to the millisecond), bishop
pair, rook-on-open/semi-open-file, passed-pawn/mobility/knight-outpost
terms (all UCI-tunable). A **prior king-safety attempt exists and was
reverted after a real SPRT failure** — check
`docs/parameter-calibration-audit.md` before re-proposing king safety
from scratch; the documented lesson from that failure is specifically
"this engine's own SPSA harness can't discover a term's magnitude from
scratch — start from validated real numbers instead."

---

## Tier 1 — cheap research/design, do freely

### NNUE / label quality

1. **Move/piece prediction objective** — is there a training-objective
   change (not a bigger model) that would help the existing policy net
   or Unarchitectured Metal's prediction quality, given what's already
   failed? Ground this against `docs/unarchitectured-metal-why-the-hint-costs-elo.md`
   (the hint's failure is structural/budget, not prediction quality) —
   be honest if a better objective doesn't address a budget problem.
2. **NNUE label-noise follow-up** — see the updated finding above. If
   you want to extend it (more PGN sources, a larger depth multiplier),
   the reproduction command is in `docs/nnue-label-noise-real-measurement.md`
   — needs only `cargo build --release -p unchessed-datagen` and a PGN
   file, no Torch, no cutechess, no cloud.
3. **What's the real ceiling if not label noise?** Now that the
   label-noise explanation is weaker, research whether architecture
   capacity or effective-data-volume-at-reachable-positions (not raw
   shard count) is a better explanation, before assuming either.

### Search

4. **AlphaZero-style RL self-play viability** — is it worth pursuing at
   all here, given the checked-in evaluator is a scalar alpha-beta net
   with no MCTS/PUCT/policy-value ABI? Real compute-cost estimate
   required, not just literature reference numbers.
5. **Search parameter RL/bandit tuning** — the project's prior SPSA
   attempt was underpowered/poorly controlled, not proof tuning fails.
   RL is a poor first method for sparse, delayed, game-outcome reward.
   Research a principled alternative (finite-arm allocation, CLOP,
   Bayesian/local search over 2-3 safe coordinates) rather than
   unconstrained RL over all ~13 parameters.
6. **Search extensions: confirmed absent entirely.** No check
   extensions, no singular extensions, no recapture extensions —
   genuinely unusual given how much pruning already exists (RFP + NMP +
   LMR + ProbCut + futility + aspiration, with zero compensating
   extensions). Research which extension is most likely additive here,
   starting with the cheapest and most standard: check extensions.
7. **Qsearch SEE-based pruning** (distinct from ordering, which already
   exists): does `qsearch` skip clearly-losing captures outright, or
   only order them last while still searching them? Verify, don't
   assume.
8. **Internal iterative reduction (IIR)** for TT-miss nodes — research
   fit given how much of the existing pruning already depends on good
   move ordering.
9. **Multi-cut pruning** — research fit given NMP/ProbCut already exist;
   be honest if this looks redundant with existing coverage.
10. **Razoring** — distinct from the futility pruning already present;
    research interaction with existing RFP/futility margins before
    proposing a combined candidate.
11. **Late-move / move-count pruning** — distinct from LMR (which
    reduces depth; this skips moves entirely past a count threshold at
    shallow depth). Research whether it's additive given LMR+futility+RFP
    already cover this space heavily.
12. **Countermove heuristic + continuation history** — natural extension
    of the existing killers/history infrastructure; order moves by how
    good they were as a *response* to the opponent's last move(s), not
    just in general. One of the more standard, lower-risk items here.
13. **King safety, retried properly**: the prior attempt failed under
    blind SPSA. Research whether a version starting from validated real
    numbers (a comparable published engine's king-safety term
    magnitudes, the way `docs/parameter-calibration-audit.md` describes
    doing for other terms) would fare differently — not a reason to
    just re-run the same failed approach.
14. **NNUE accumulator correctness under null-move pruning + Lazy SMP**
    — null moves flip side-to-move without moving pieces, which under a
    mover-perspective NNUE representation means swapping which
    accumulator half is "STM." Verify this is handled correctly,
    especially interacting with multiple Lazy SMP threads sharing
    infrastructure — a well-known bug source in other engines' NNUE
    implementations, not a confirmed bug here, just unverified.
15. **MultiPV + aspiration window interaction** — a known-tricky
    combination in many engines (a narrow aspiration window can starve
    secondary PV lines of a fair search). Verify correctness, don't
    assume either way.

### Protocol completeness / infrastructure (lower priority than the above, but cheap)

16. **`searchmoves`** (restrict analysis to specific candidate moves)
    and **`go mate N`** (search specifically for mate in N) — both
    confirmed absent from the UCI layer.
17. **`MoveOverhead`** — confirmed absent. A near-universal UCI option
    (reserve N ms per move for GUI/network lag) that's a real gap given
    how sophisticated the rest of the time management already is.
18. **`UCI_AnalyseMode`** — confirmed absent. Some engines behave
    differently in analysis vs. play (e.g. disabling contempt);
    research whether that distinction matters here given `Contempt`
    already exists as a tunable option.
19. **WDL-style output** — the NNUE already trains on a WDL-blended
    target internally (`train_nnue.py`'s loss); nothing surfaces a
    win/draw/loss estimate to the GUI. Research whether a real
    calibrated-against-outcomes probability (not just repurposing the
    training blend) is derivable from the existing eval scale.
20. **Insufficient-material / dead-position draw detection** — confirmed
    absent. Verify whether this matters in practice (does the eval
    already score these positions near zero reliably?) before treating
    it as a priority.
21. **TT generation/aging** — confirmed absent beyond depth-preferred
    replacement. Research whether this is a real practical issue given
    how the engine is actually deployed (does the harness reliably send
    `ucinewgame` between games?) before prioritizing.
22. **Continuous SPRT testing infrastructure** (meta, not chess) — every
    SPRT in this project's history has been a manually-orchestrated
    one-off. Research whether a lightweight, self-hosted continuous
    testing setup (fishtest-style, scaled down) would make every other
    item on this list faster to actually validate.

### Persona

23. **Persona real-play measurement protocol** — design (don't execute
    without real cutechess access) the labeled real-game study needed to
    actually measure `PersonaSmooth`/`EngineDetectV2`'s flip-rate and
    misfire claims, which are currently simulation-only.

---

## Tier 2 — needs a posted plan first, wait for acknowledgment

Anything from Tier 1 that concludes "worth implementing" graduates
here: a real NNUE relabel+retrain if the label-noise finding motivates
one, a minimal RL self-play prototype if 4 concludes it's worth trying,
the oracle rating-conditioning sweep if a checkpoint becomes available,
or any search-extension/pruning candidate that clears its Tier 1
research and fixed-position safety checks.

## Tier 3 — real compute, real SPRT, explicit human go-ahead

Any full NNUE v4 candidate retrain at scale, any Unarchitectured Metal
GAB-widening/quantization retrain, any full self-play RL pipeline, any
cloud spend, endgame tablebase file acquisition/hosting, Chess960's
final validation, and any search-extension/pruning campaign's real SPRT
run. Do not start without explicit go-ahead, not just an acknowledged
plan.

## Reporting format

One doc per investigation under `docs/reinforcement/` (or `docs/` for
non-RL engine topics), continuing the existing numbering convention.
State what you actually ran vs. design-only, verified vs. assumed, real
numbers if you have them, and an honest recommendation (pursue / defer
/ drop) even when the answer is "drop." Commit and push to
`manus/research-facilities` only.
