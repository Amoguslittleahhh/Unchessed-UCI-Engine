# Unchessed AI — extended research brief for `manus/research-facilities`

## Read this paragraph before anything else: credit budget

The account funding this has **3,422 credits total, for everything below,
indefinitely** — not per-task. This document is intentionally large and
ambitious because the *menu* should be big; your actual spend should not
be. Work in the priority order given, cheapest and most diagnostic first.
**Before starting anything in Tier 2 or Tier 3, post a short plan and an
honest credit-cost estimate and wait for it to be acknowledged** — not
because permission is required for every command, but because an
expensive compute job started on a wrong assumption is unrecoverable
credit loss, and a design doc is not. If you are ever unsure whether
something is "cheap" or "expensive" for this specific budget, treat it as
expensive and ask. Silence is not consent to spend; a stalled task
costs nothing, a wasted training run does.

## Repo and branches

Repo: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine

- **`main` is read-only.** Never push to it, never merge into it.
  Someone reviews and merges by hand after independently checking the
  work — same as for every other contributor.
- **Work on `manus/research-facilities`** (renamed from
  `manus/rustc-bootstrap-trial` — same branch, same history, just a
  clearer name for what it's actually for now). It already has real,
  independently-verified work on it: `cargo test --workspace --release`
  passes 123/123 as of the last review, three named parity gates pass,
  `unarchitectured_metal.rs`'s change is comment-only, `search.rs`'s real
  change is a narrow node-limit correctness fix. Build on it, don't
  restart from `main`.
- Commit and push there. Don't open a PR into `main` yourself.

## What's already been established (read the actual docs, this is a summary)

You (a previous Manus session on this same branch) already wrote
`docs/reinforcement/00-synthesis.md` through `05-oracle.md` — a genuinely
disciplined gating analysis distinguishing "implement now" (safe,
default-preserving) from "blocked" (missing assets: real corpus, Torch,
oracle checkpoint, current match infra) from "never promote without a
real SPRT." That discipline is exactly right and should continue to
govern everything below — re-read those five files before starting,
they're the actual current state of this line of work, more precise
than this summary.

The project's three live subsystems, in one line each:

- **NNUE evaluator** (default eval, `unchessed-core/src/nnue.rs`,
  `tools/train_nnue.py`) — real SPRTs across five data scales found the
  strength ceiling is likely **label noise from the 5000-node HCE
  search**, not architecture or data volume (`docs/nnue-v4-108m-recipe-result.md`,
  `docs/ieee-low-cp-val-mae-and-persona.md`). Best v4 retrain is still
  −155.6 ± 47.7 Elo behind the shipped v3 default.
- **Unarchitectured Metal** — an experimental transformer move-ordering
  hint (`unarchitectured_metal_runtime.rs`, `unarchitectured_metal.rs`). Real speed work
  exists (~7.5ms/call forward pass); the hint itself has **never once
  trended positive** across four real SPRT batches and must stay
  `UnarchitecturedHint=false`.
- **Persona/adapter** (`adapt.rs`) — `PersonaSmooth`/`EngineDetectV2`,
  both default `false`. One real SPRT found Elo-neutral (≈+2.1, noise);
  the behavioral claims (flip-rate reduction, misfire fixes) are
  simulation-only, never measured from real game telemetry.

## Standing rules (non-negotiable, apply to everything below)

1. **No search/live-behavior default flip without a real paired-game
   cutechess SPRT.** Simulation, unit tests, and diagnostics are
   evidence for a writeup, never a substitute for the gate. This
   project had one catastrophic failure early on from skipping this —
   it's the reason the rule exists, not a formality.
2. **The three named parity tests must keep passing** after any
   `unarchitectured_metal_runtime.rs` change: `start_position_matches_python_reference`,
   `midgame_position_matches_python_reference`,
   `position_to_input_matches_hand_built_start_position`
   (`cargo test -p unchessed-core --release`).
3. **State what you actually verified vs. what you're claiming without
   checking.** "Should work" is not "I ran it." If your sandbox lacks an
   asset (Torch, a checkpoint, cutechess, current rustc), say so plainly
   and mark the affected claim as unverified — don't round up.
4. **Defaults don't move without evidence.** Every UCI default that's
   `false`/off today stays that way unless a specific item below says
   otherwise and the required gate has actually been cleared.
5. **Report negative results as plainly as positive ones.** A
   well-run experiment that disproves the hypothesis is real progress
   and should be written up with the same care as a win — this project
   has several of those already (`docs/unarchitectured-metal-why-the-hint-costs-elo.md`,
   `docs/nnue-v4-retrain-data-scaling-finding.md`) and treats them as
   genuine contributions, not failures to hide.

---

## Tier 1 — cheap, diagnostic, do these first (design docs, analysis, small code, no big compute)

Pick freely among these; they don't need to happen in order, and doing
several is fine before moving to Tier 2.

### 1.1 Reinforcement-learning self-play: is it worth pursuing here at all?

**The real question, not a foregone conclusion**: this project's NNUE
ceiling looks like it's capped by 5000-node HCE label noise
(see above). AlphaZero-style self-play RL sidesteps needing external
search-derived labels entirely — the network trains against its own
play outcomes. That's a genuinely different lever than anything tried
so far (relabeling, more data, architecture changes).

Before touching any code: write a research doc answering, honestly and
with real numbers where you can find or estimate them:

- What would a *minimal* viable self-play RL loop look like for a net
  this small (NNUE: 5.7M params, HalfKAv2_hm features)? Compare against
  what Stockfish's own NNUE training actually uses (it's not classic
  AlphaZero self-play — it's supervised on strong-engine-labeled data,
  which is closer to what this project already does). Is "add RL"
  solving the actual diagnosed problem (label noise) or a different
  problem this project doesn't have?
- Real compute cost estimate for a self-play loop that could plausibly
  move the needle at this scale — training games needed, search depth
  per self-play move, wall-clock/dollar estimate on cloud hardware. Be
  honest if the answer is "this needs orders of magnitude more compute
  than this project has spent on anything so far."
- What's the smallest real experiment that would tell you if this
  direction has legs, without committing to a full pipeline? (E.g.:
  a tiny-scale self-play run on a toy position set, checked for whether
  eval quality *improves at all* over a few iterations, before deciding
  whether the full version is worth the compute.)
- Concrete recommendation: pursue, defer, or drop — and why, in terms
  of this project's actual diagnosed bottleneck, not RL's general
  reputation.

This is research writing, not implementation — cheap regardless of the
conclusion.

### 1.2 Move/piece prediction: is the *objective* the actual bottleneck?

The project already has two move-prediction-adjacent systems: a
Maia-style human-move policy net (rating-conditioned priors used by the
persona system) and Unarchitectured Metal (a transformer hint that's never
helped Elo despite good top-1 accuracy work). Both are trained with a
fairly standard "predict the next move/label" objective.

Research question: is there a training-objective change — not a bigger
model, not more data — that would plausibly help either system, given
what's already been tried? Concrete angles to actually investigate
(not just list):

- Full-game continuation prediction (predicting several plies ahead,
  not just one move) vs. single-move classification — does the
  literature (and this project's own `docs/research-notes-*.md` survey
  files, check what's already been read) suggest this changes anything
  for a model this size, or is it a scale-dependent trick that doesn't
  transfer down?
- The Unarchitectured Metal hint's actual measured failure mode
  (`docs/unarchitectured-metal-why-the-hint-costs-elo.md`) is structural —
  it only orders the first iterative-deepening pass, and the real cost
  is the forward-pass time budget, not prediction quality. Does a
  better prediction *objective* address a budget problem at all, or is
  this a dead end regardless of how good the predictions get? Give the
  honest answer.
- If there's a real, specific, testable idea here (not "try a bigger
  transformer"), scope the smallest experiment that would validate it
  before proposing real training compute.

### 1.3 NNUE label-noise experiment — actually run the cheap part

`tools/nnue_relabel_existing.py` exists (you already extended it
substantially — 873 lines now). The `compare` mode needs only a sidecar
of new scores and the existing shards, no training, no GPU. If you have
any way to generate a small sidecar (deeper HCE search on a modest
sample, or self-distillation scores from the shipped net at high node
count, on whatever positions you can access) — even a few thousand
positions — run `compare` and report the actual old-vs-new MAE/Pearson.
This is the cheapest possible real signal on whether label noise is
really the ceiling, without touching Torch or spending on a full
retrain.

If you have no way to generate real alternative scores in your sandbox,
say so and stop here rather than fabricating a sidecar — a synthetic
sidecar built by adding noise to the existing labels would just tell
you what you already assumed.

### 1.4 Search-side reinforcement learning: automatic parameter tuning

This project has prior history worth knowing before proposing anything
here: a past attempt at self-tuning search parameters via SPSA is
recorded in project memory as having failed to discover a term's
magnitude from scratch — the working approach was starting from
validated real numbers, not blind search. (Check
`docs/parameter-calibration-audit.md` for the specifics.) The dozen or
so tunable search parameters already exposed as UCI options
(`RFPMargin`, `NullMoveBase`/`Divisor`, `LMRMinDepth`/`MinMoveNumber`/
`BigMoveNumber`, `AspirationDelta`/`MinDepth`, `ProbCutMargin`/
`Reduction`/`MinDepth`, `FutilityMargin`/`MaxDepth`) already exist on
`main` for exactly this purpose, unrelated to your branch.

Research question, not implementation yet: is there a principled
RL/bandit-style tuning approach for these that would actually do better
than the documented SPSA failure, given the same constraint (a real
signal here requires real games, which cost real compute)? Or does the
prior failure generalize — i.e., is "don't blind-search this, use
principled starting points" still the right conclusion regardless of
which optimizer you'd use? Be honest if the answer is the latter.

### 1.5 Persona real-play measurement design (not execution yet)

Round 20's synthesis doc already flagged this correctly (P2 in the
gating matrix): the flip-rate and misfire claims for `PersonaSmooth`/
`EngineDetectV2` have never been measured from real game telemetry, only
simulation. `AdapterTelemetry=false` and the parser exist per your own
prior work. Design (don't yet run, unless you have real cutechess
access) the actual measurement protocol: what labeled game set would
be needed, what the pre-registered success criteria would look like,
how many games for a statistically meaningful flip-rate confidence
interval. This is planning work, cheap regardless of whether it's ever
executed.

---

## Tier 2 — moderate cost, needs a plan posted and acknowledged first

Do not start these without posting your plan and cost estimate first,
per the budget rule at the top.

### 2.1 NNUE relabel-and-retrain, if 1.3 showed real signal

If Tier 1.3's cheap `compare` step showed non-trivial MAE between old
and new labels, the next step is a real relabel + retrain + SPRT on a
meaningful slice — but only if you have real compute access to do a
retrain (Torch, enough CPU/GPU, and — critically — a way to actually
run a cutechess SPRT against the shipped default, since a lower MAE is
explicitly not a promotion result per the gating matrix). Post the plan
first: data volume, recipe (reuse the round-13 defended recipe:
best-checkpoint export, early-stop, correct batch size — don't
re-derive from scratch), and how you'll run the SPRT.

### 2.2 Minimal self-play RL prototype, if 1.1 concluded it's worth trying

Only if Tier 1.1's research honestly concluded this is worth a real
experiment (not a foregone "let's try it"): scope and, with a posted
plan, run the smallest version that would produce a real answer — not
a full pipeline. Expect this to be judged mostly on whether the
experiment design itself was sound, not on whether it beat anything on
the first try.

### 2.3 Oracle-side rating-conditioning sweep, if the checkpoint becomes available

`tools/analyse_oracle_rating_conditioning.py` (your own prior work)
already implements the fail-closed sweep tool per the O1 gate. If a
trusted `UNARCHV1_ORACLE_TRAINING_V1_DDP` checkpoint and Torch become
available in your environment, run the 200-position/7-rating sweep and
report the raw result — this is cheap once the checkpoint exists, the
blocker has only ever been asset availability.

---

## Tier 3 — real compute, real SPRT, project-owner-level decisions

Do not start anything here without explicit go-ahead from a human, not
just an acknowledged plan. This tier is: any full NNUE v4 candidate
retrain at real scale, any Unarchitectured Metal GAB-widening or
quantization retrain, any full self-play RL pipeline, any cloud spend
of any kind. All of round 13-19's own history in this repo (the −155.6
Elo NNUE result, the persona SPRT) went through exactly this kind of
gate before real compute was spent — follow that precedent.

---

## Reporting format

For each item you work on, write a doc under `docs/reinforcement/`
(continuing your own existing numbering/naming convention) stating:
what you actually ran vs. what's design-only, what you verified vs.
assumed, the real numbers if you have them, and an honest
recommendation (pursue / defer / drop) even if the honest answer is
"drop." Commit and push to `manus/research-facilities`. Someone will
review before anything moves toward `main` or real compute spend —
same process as always.
