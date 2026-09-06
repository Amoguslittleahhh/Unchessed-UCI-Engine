# Parameter calibration audit — and the work order to finish it

"Calibrate everything in the engine for maximum performance and results",
audited against the code on 2026-08-26. This document (a) inventories every
tunable in the engine, (b) separates what is already calibrated — with the
evidence — from what never has been, (c) records the static findings, and
(d) is the executable work order for the calibration that actually
requires measurement, which this sandbox cannot provide (no Rust
toolchain: no compile, no perft, no SPRT).

**Nothing in this round changes any parameter value, and no Rust was
modified.** Per project discipline, every behavior-affecting value change
requires a fresh SPRT on real hardware
(`docs/engine-commit-feed-review.md`: transplanted tuned constants are
"the sort of change this project's SPRT discipline exists to reject").
This round adds the audit, a drift guard, and a time-budget baseline.

> **Addendum (2026-08-26, later round):** the "no Rust toolchain" premise
> has since changed — rustc 1.97.0 is now available in this sandbox via the
> `arena-rust-toolchain` PyPI package (see
> `docs/dev-environment.md`), the workspace compiles cleanly with **zero
> source changes** (debug 2.9 s, release+LTO 16.8 s, no warnings), and the
> full test suite passes: 104/104 unit tests, the deep perft, the three
> 5e-3 parity gates, and the UCI smokes including the matetrack `Ra8#`.
> Stage 0 of the work order (build + perft + correctness sanity) is
> therefore executable in-sandbox from now on; Stage 5 can be compiled
> here but its nps numbers do not transfer off this 2-core Xeon. Stages
> 1–4 (SPRT) still need cutechess + an opponent engine + a book, i.e. the
> reviewer's hardware.

## What "calibrated" means here (evidence hierarchy)

1. **Measured on real deployment hardware** (285H / reviewer hardware):
   e.g. thread counts, inference threads, TT probe cost.
2. **SPRT-validated** (paired games, LLR crossing a bound): the HCE terms
   below; the `UnarchitecturedMinTime` candidates.
3. **Hand-picked default** ("previously hard-coded", per the `SearchParams`
   doc comment): untested. This is the calibration debt.

## Inventory

### A. Search pruning parameters — 13 spins + 1 check, all hand-picked

`SearchParams` (`unchessed-core/src/search.rs:20-70`), advertised and
clamped in `uci.rs:264-279`. The struct's doc comment states the design
intent: "exposed as UCI options so they can be adjusted without a rebuild
(and eventually driven by automated tuning, e.g. SPSA)" — **the tuning run
itself was never executed**:

| UCI option | default | advertised range | guard (all reachable — see §static) |
|---|---:|---|---|
| RFPMargin | 90 | 10–300 | `depth <= 6` (hard-coded), !PV, !in-check, non-mate beta (`search.rs:536-543`) |
| NullMoveBase / NullMoveDivisor | 3 / 6 | 1–6 / 2–12 | `depth >= 3` (hard-coded), !PV, !in-check, `static_eval >= beta`, non-pawn present (`search.rs:550-560`) |
| LMRMinDepth / LMRMinMoveNumber / LMRBigMoveNumber | 3 / 3 / 12 | 1–8 / 0–20 / 4–40 | quiet, !in-check, !gives-check (`search.rs:758-771`) |
| AspirationDelta / AspirationMinDepth | 25 / 4 | 5–200 / 1–12 | non-mate-range prev score (`search.rs:1003-1017`) |
| ProbCutMargin / ProbCutReduction / ProbCutMinDepth | 200 / 4 / 5 | 50–400 / 2–6 / 3–10 | !PV, !in-check, `rdepth >= 1` enforced (`search.rs:586-597`) |
| FutilityMargin / FutilityMaxDepth | 150 / 8 | 30–400 / 1–12 | `legal_count > 1`, !PV, quiet, non-mate alpha (`search.rs:716-729`) |
| ProbcutSeeFilter | **false** | — | deliberately off "until a paired-game SPRT says otherwise" (`search.rs:56-61`) — the gate it references is **Stage 2** below |

### B. HCE eval terms — SPRT-validated, but only active in the HCE fallback

`EvalParams` (`unchessed-core/src/eval.rs:80-112`), advertised in
`uci.rs:284-293`. The defaults carry their own validation history:

| Term | value | evidence (from the `Default` impl comments) |
|---|---:|---|
| passed pawn mg/eg | 100/100 | SPRT 2026-08-02: +25.7 ± 12.3 Elo, 2156 games, LLR crossed upper bound |
| mobility | 100 | SPRT 2026-08-03: +52.3 ± 18.1 Elo, 984 games — biggest eval-term gain so far |
| rook file/7th | 100 | SPRT 2026-08-04: +10.5 ± 7.2 Elo, 6019 games |
| knight outpost | 100 | SPRT 2026-08-10: +12.0 ± 7.8 Elo, 4873 games (first 10000-game attempt inconclusive at +5.2 ± 5.4, extended to 30000-game cap, converged) |

**These are calibrated.** Two caveats:

- The engine auto-loads the shipped `unchessed-nnue.bin` (23 MB) from the
  binary's directory as the default evaluator (`uci.rs:441-455`). The five
  `*Pct` options are applied only when `eval_is_hce` (`uci.rs:664-700`) —
  **in the default NNUE configuration they are silently inert**: a GUI can
  set `RookPct 50` and nothing changes. (Making that an explicit error
  message is a reviewable UX change, not done here.)
- The comment also records the standing discipline from the *failed*
  king-safety attempt: "this engine's own SPSA harness can't discover a
  term's magnitude from scratch, so starting from validated real numbers
  is the standing discipline" (the mobility table is taken from
  RubiChess's SPSA-tuned values).

### C. Time management — hard-coded, never exposed, never tuned

`Limits::budget` (`search.rs:139-175`) and the situation factor in
`go_with_root_hints` (`search.rs:919-927`). All of the following are
hand-picked and **not** UCI options:

- movetime overhead: `t - 25` ms floored at 5 (`search.rs:144`);
- base split: `soft = t/mtg + 3/4·inc`, `hard = t/5 + 1/2·inc`,
  `mtg` default 30 (`search.rs:152-154`);
- low-clock tiers: `t<20000 → soft ≤ t/35+inc/2, hard ≤ t/10+inc/2`;
  `t<6000 → soft ≤ t/60+inc/2, hard ≤ t/16+inc/2`;
  `t<2000 → soft ≤ max(inc/2, 30), hard ≤ t/8` (`search.rs:156-168`);
- reserve: ceiling `t - 60`, hard ≥ 5, soft ≥ 3 (`search.rs:169-171`);
- situation factor: `width = clamp(0.65 + legal/45, 0.75, 1.3)`,
  `×1.25` when in check, applied to soft only (`search.rs:919-927`).

What the current (untuned) ratios allocate, at 30 root legal moves, from
`tools/simulate_time_budget.py` (faithful transcription, pinned by
`tools/test_simulate_time_budget.py` against the Rust
`budget_speeds_up_as_clock_drains` assertions):

| time left | tc 5+0.05 (inc 50) | tc 60+0.6 (inc 600) | tc 1+0 (inc 0) |
|---:|---:|---:|---:|
| full clock | soft 10037 / hard 60025 ms (3.3% / 20% of clock) | 12450 / 72300 (3.5% / 20%) | 2000 / 12000 (3.3% / 20%) |
| 30000 | 1037 / 6025 (3.5%) | 1450 / 6300 (4.8%) | 1000 / 6000 (3.3%) |
| 10000 | 310 / 1025 (3.1%) | 585 / 1300 (5.8%) | 285 / 1000 (2.9%) |
| 3000 | 75 / 212 (2.5%) | 350 / 487 (11.7%) | 50 / 187 (1.7%) |
| 1000 | 30 / 87 (3.0%) | — | 16 / 62 (1.6%) |

The soft stop is a stable ~3.3–4.5% of the remaining clock at full clock
across all three time controls — plausibly sane, but it is a guess, and
time-management changes move allocation across *every* game, so they are
the highest-risk calibration class (Stage 3).

### D. Infrastructure — mostly already calibrated by measurement

| Knob | current | status |
|---|---|---|
| `Threads` | `available_parallelism()`, cap 32 (`uci.rs:86-87`) | **measured/fixed**: was default 1 (6% of a 16-core chip); regression test asserts the default (`docs/tuning-core-ultra-9-285h-and-low-end.md`) |
| `Hash` | 128 MB, range 1–2048 | **measured**: probe cost 2.5 ns/probe ≤64 MB vs 14.9 ns at 256 MB; 128 MB sensible, low-end 32–64 MB (same doc) |
| `UNCHESSED_INFERENCE_THREADS` | env, default 1 (`unarchitectured_metal_runtime.rs:70-80`) | **measured** (round 7): 1 thread 7.92 ms was fastest; old default 4 was 11.01 ms |
| `UnarchitecturedMinTime` | 30000 (1000–600000) | **best of 3 SPRT-tested values** (1000 → −26.1/−15.1 Elo; 30000 → −5.8, LOS 34%); only meaningful while the hint is on, which it is not by default |
| MultiPV | 1 (3 in adaptive mode) | not tuned |
| BookDepth / OwnBook | 16 / true (adaptive mode) | not tuned (adaptive/persona mode, out of scope for strength) |
| UCI_Elo / Contempt / Troll | 2400 / 25 / Auto | limit-strength/persona features, not strength knobs |

### E. Unarchitectured Metal runtime

- Architecture constants (`D_MODEL 256`, 8 layers, etc.,
  `unarchitectured_metal_runtime.rs:25-41`) are fixed by the checkpoint — not tunable.
- `TOKEN_BLOCK = 4` kernel unroll (`unarchitectured_metal_runtime.rs:1109+`):
  hand-picked, **never micro-benchmarked** on the target CPU. Tuning it is
  a Stage-5 microbenchmark, not SPRT (it changes speed, not the tree).
- int16 activations: closed as retrain-gated
  (`docs/int8-activation-calibration-finding.md`); VNNI dot-product kernels
  for the int16 *weight* path remain the open kernel item (needs Rust +
  hardware).

### F. NNUE

Single 512-wide head, v3 format. Bucketed output heads (8 piece-count
buckets) and int16/QAT are **retrain** items with their own documented
gates (`docs/research-notes-moe-2507.11181.md`,
`docs/performance-ceiling-and-gpu-viability.md`) — no runtime calibration
can substitute for the retrain.

### G. Datagen (training track, listed for completeness)

Quiet-position filter margins `UNCHESSED_QUIET_MARGIN_QSEARCH/SEARCH` =
60/70 cp: env-overridable, "published margins were tuned on Xiangqi, not
Western chess … a starting point, not a constant to transfer"
(`docs/nnue-dataset-quiet-filters.md`). Retuning affects *training data*,
not the runtime; it belongs to the retrain backlog.

## Static findings (this round)

1. **No dead parameters.** All 13 search parameters' guard conditions are
   reachable in normal play (guard table in §A). No parameter is shadowed
   or unreachable.
2. **The advertised surface and the code agree, verified.** All 19 spin
   options + 1 check: UCI advertised default == struct default, advertised
   `[min, max]` == handler `v.clamp`, and the `UnarchitecturedMinTime`
   Options literal (30_000) == advertised 30000 with matching clamp. This
   round adds `tools/check_search_param_consistency.py` to keep it that
   way: it exits 1 and names any drift (8 mutation negative controls in
   its test file). A tuning campaign that commits a new default to only one
   of the two places would otherwise silently desync the shipped behavior
   from the config the SPRT was run with.
3. **The five `*Pct` options are silently inert in the default (NNUE)
   configuration** (§B). If a GUI or tuning script sets them, nothing
   changes. Documented; no behavior change made.
4. **Hard-coded ceilings adjacent to tunables**: RFP `depth <= 6`
   (`search.rs:540`), NMP `depth >= 3` (`search.rs:553`), the movetime 25 ms
   overhead. If Stage 1's SPSA wants to interact with these (e.g. a larger
   RFP margin is pointless without a higher ceiling), exposing them is a
   reviewable, behavior-preserving first step.
5. **Rust not compiled** (no toolchain in the sandbox). All of the above
   is read-level; the two new Python tools are the executable portion.

## Work order (real hardware; nothing here runs in the sandbox)

Conventions are the repo's own: `scripts/sprt-history/*.sh`
(cutechess-cli, `-sprt elo0=0 elo1=5 alpha=0.05 beta=0.05`, `tc=5+0.05`,
`Threads=1`, `Hash=256`, 5000-round scripts, `smoke_*` pre-checks), pentanomial
analysis via `tools/pentanomial_sprt.py`, one change at a time against the
current winner (the passed-pawn → mobility → rook → knight-outpost sequence
in §B is the template).

**Stage 0 — harness sanity.** Build the current tree; run the existing
perft/correctness tests; a 1000-game SPRT of two identical builds must sit
at ~0 Elo (this validates book, tc, and analysis tooling before any
candidate is spent).

**Stage 1 — SPSA over the 13 search parameters.** Start at current
defaults (the validated state; the king-safety failure is the reason we do
not start from 0 or from Stockfish's constants). Search ranges: the
advertised UCI bounds (they were designed as the tuning range —
`SearchParams` doc comment). Sequential, one parameter at a time, each
candidate gated by ≥2000 games at `tc=5+0.05`; keep only positive-LLR
results; re-baseline after each acceptance; record every negative in
`benchmarks/unarchitectured-metal/` with the game count, per project
discipline. If a parameter is inconclusive at a 30000-game cap, it stays
at its default and is documented as such (the first knight-outpost attempt
is the precedent).

**Stage 2 — `ProbcutSeeFilter` on/off.** The exact gate the code comment at
`search.rs:56-61` was written to be run. Tree-changing, so the conservative
tc is worth a second look: `tc=60+0.6` (the round-7 conservative profile)
in addition to `tc=5+0.05`.

**Stage 3 — time management.** First a reviewable, behavior-preserving
diff exposing the §C ratios (including the RFP depth ceiling) as UCI
options — the new consistency linter then enforces advertised == code
defaults. Then SPSA on the soft/hard base split and the low-clock tier
thresholds, gated at both `tc=5+0.05` and `tc=60+0.6` (low-clock behavior
only shows up in the latter's tail and in fast time controls). Re-run
`tools/simulate_time_budget.py` with the accepted values and commit the
before/after table as the artifact.

**Stage 4 — `*Pct` scales: skip unless the HCE fallback is a deployment
target.** In the default NNUE configuration they are inert (§B finding 3);
tuning them for a configuration nobody ships is theater. If a no-bin
deployment ever matters, run the same Stage-1 protocol with
`EvalFile` unset.

**Stage 5 — microbenchmarks (speed, not SPRT).** On the 285H:
`TOKEN_BLOCK ∈ {2, 4, 8}` (nps on a fixed-position set + depth at fixed
budget), and a `Hash` size sweep at the actual TT occupancy profile.
Record in `benchmarks/`; commit only values that win and are reproducible.

**Non-goals under any outcome** (standing policy, restated so the work
order cannot be misread as waiving them):

- `UnarchitecturedHint` stays default **false**;
- `runtime_safety_suite` stays **false**;
- no `unarchitectured_metal_runtime.rs` change from this work order (the 5e-3 parity
  gates `start_position_matches_python_reference`,
  `midgame_position_matches_python_reference`,
  `position_to_input_matches_hand_built_start_position` are untouched);
- retrain items (output-head bucketing, int16/QAT, phase-specialized
  experts per `docs/research-notes-moe-mcts-2401.16852.md`, GAB capacity)
  stay on the retrain backlog;
- no parameter value is changed by committing a tuned number that has not
  crossed its own SPRT bound on this engine, with this eval, at this tc.

## Verification status (this round)

- `tools/check_search_param_consistency.py` run against this tree: **20
  options checked, 0 drift(s)**; its 8 mutation-based negative controls
  each flip the exit code and name the option.
- `tools/simulate_time_budget.py` pinned to the exact values the Rust
  `budget_speeds_up_as_clock_drains` assertions imply (soft 6000/666/83/16
  at 180s/20s/5s/1s with inc 0; hard ≤ 245 at 300 ms); the §C table above
  is its output.
- No Rust file modified; no Rust compiled (no toolchain in the sandbox);
  `rust_bracket_check --all` run as the standing syntax guard.
- `UnarchitecturedHint` default `false` and `runtime_safety_suite` `false`
  re-verified in the tree.

## Files

- `tools/check_search_param_consistency.py` — advertised/struct/clamp
  cross-check for all 20 options (stdlib only).
- `tools/test_search_param_consistency.py` — 12 tests incl. 8 mutations.
- `tools/simulate_time_budget.py` — faithful `Limits::budget` + situation
  factor transcription, CLI table.
- `tools/test_simulate_time_budget.py` — 10 tests pinning the Rust test's
  fixture values.
- This document: inventory + work order.
