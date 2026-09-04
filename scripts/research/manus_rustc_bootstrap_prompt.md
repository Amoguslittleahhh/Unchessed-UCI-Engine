# Task: get a working Rust toolchain in this VM, then verify the repo builds

## Repo and branches — read this first

Repo: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine

- **`main` is read-only.** Never push to it, never open a PR into it,
  never merge into it yourself. Someone else reviews and merges into
  `main` by hand after checking the work independently — same process
  used for every other contributor to this repo.
- **Work on `manus/rustc-bootstrap-trial`.** Create it from `main` if it
  doesn't exist yet (`git checkout -b manus/rustc-bootstrap-trial`), and
  push everything there:
  `git push -u origin manus/rustc-bootstrap-trial`.
- Clone/pull `main` to get the current code (including
  `scripts/setup-rust-toolchain.sh`), but do all commits and pushes on
  the branch above, not `main`.

## Context

It's a from-scratch Rust UCI chess engine plus a Python
training/tooling stack.

Another AI coding agent ("arena") has been doing engineering work on
this repo across ~20 rounds, in a sandboxed VM whose outbound HTTPS is
filtered: GitHub's own HTML/API is reachable, but `rustup.rs`,
`static.rust-lang.org`, and the Debian package CDN (`deb.debian.org`)
all fail TLS from that sandbox. That's meant every round touching a
`.rs` file has had no way to actually compile or test its own changes —
several rounds pushed real compile errors as a result, caught later by
independent review on different hardware.

This is a **temporary trial**: seeing whether a Manus virtual computer
has a network policy that actually allows getting a Rust toolchain, as
an alternative environment for verifying Rust changes when arena's own
sandbox can't.

## What to actually do

1. **Check if a Rust toolchain is already present**: `rustc --version
   && cargo --version`. If yes, skip to step 3.
2. **If not, try to install one.** In order of preference:
   - `rustup` via the official installer:
     `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
   - If that fails (TLS/network error), try apt:
     `sudo apt-get update && sudo apt-get install -y rustc cargo`
   - If that also fails, try Homebrew (different infra — its installer
     is on `raw.githubusercontent.com`, its packages come from `ghcr.io`,
     both GitHub-hosted rather than Rust's or Debian's own CDN):
     ```
     NONINTERACTIVE=1 bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     brew install rust
     ```
   - `scripts/setup-rust-toolchain.sh` in the repo root automates all
     three of the above, in order, and reports exactly what happened
     with each — just `source` it after cloning, no need to redo this
     by hand.
3. **Clone the repo, create the working branch, build it**:
   ```
   git clone https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine.git
   cd Unchessed-UCI-Engine
   git checkout -b manus/rustc-bootstrap-trial
   cargo test --workspace --release
   ```
   If this trial produces anything worth keeping (e.g. an improved
   `scripts/setup-rust-toolchain.sh`), commit it on this branch and
   `git push -u origin manus/rustc-bootstrap-trial` — do not push to
   `main`.
4. **Report back**:
   - Which install path worked (rustup / apt / Homebrew / already
     present), or that all failed with the exact error output.
   - The real `cargo test --workspace --release` result (pass/fail
     counts, any failures verbatim).
   - `rustc --version` / `cargo --version` if it worked.

That's the whole ask for this trial — establishing whether this
environment can do the one thing arena's sandbox categorically cannot.
No engineering changes needed yet; if this works, there's a real backlog
of open items in
`scripts/research/arena_agent_unarchitectured_v1_runtime_speed_prompt.md`
that could use a second environment capable of compiling Rust, but
that's a separate follow-up conversation once this baseline is confirmed.

## If it works

Say so plainly with the version numbers and test results. Don't
speculate about *why* it worked (different VM provider, different
default network policy, etc.) unless you actually have evidence — a
working result is the useful part.

## If it also fails

Report the exact error output for whichever step failed, the same way
`scripts/setup-rust-toolchain.sh` already logs it. That's still useful:
it narrows down whether this is a Rust-ecosystem-wide CDN issue (would
affect any sandboxed VM) versus something specific to arena's own
sandbox's network policy.

---

## Project orientation (read if this trial succeeds and turns into more)

Everything below is context for follow-on work, not required for the
rustc trial itself. Skim it once so you're not starting cold if asked
to do more.

### What this project is

Unchessed AI: a from-scratch UCI chess engine in Rust, plus a large
Python training/tooling stack. Two Rust binaries share one core crate:

- **`unchessed-adapter`** — the live-play binary. Implements a persona
  system (`Adaptive` UCI option) that adjusts play style to an estimated
  opponent Elo, on top of full-strength search.
- **`unchessed-reviewer`** — a raw-strength binary, no persona logic.

Workspace crates: `unchessed-core` (engine logic, evaluators, adapter),
`unchessed-adapter`, `unchessed-reviewer`, `unchessed-datagen` (training
data generation).

### The three live subsystems, current state

1. **NNUE evaluator** (`unchessed-core/src/nnue.rs`,
   `tools/train_nnue.py`) — the shipped default evaluator
   (`unchessed-nnue.bin`, format v3/v4). A newer 8-bucket format (v4) and
   a defended training recipe (best-checkpoint export, early-stop,
   correct batch size) exist and are verified, but no v4 net has beaten
   the shipped v3 default in a real SPRT yet — closest so far is
   −155.6 ± 47.7 Elo at 108M training positions, still a real gap. Full
   history and the current best next idea (relabeling existing shards
   with stronger search-derived scores, not more of the same data —
   `tools/nnue_relabel_existing.py`) are in the backlog doc below.
2. **Unarchitectured v1** — an experimental transformer-based move-
   ordering hint (`unchessed-core/src/aegis_v4_runtime.rs`,
   `unarchitectured_v1.rs`). Real, measured runtime speedups exist
   (currently ~7.5ms/call forward pass on reference hardware), but the
   hint itself (`UnarchitecturedHint` UCI option) has **never once
   trended positive** across four real SPRT batches and **must stay
   default `false`**. Speed work on the forward pass is welcome and
   ongoing; re-enabling the hint in search is not, absent new SPRT
   evidence that actually changes that conclusion.
3. **Persona smoothing / engine detection** (`unchessed-core/src/adapt.rs`) —
   two newer UCI options, `PersonaSmooth` (EMA + dwell hysteresis on
   persona-mode switching) and `EngineDetectV2` (fixed misfires in
   opponent-engine detection). Both **default `false`**. A real cutechess
   SPRT found no Elo difference either way (≈+2.1 Elo, statistically
   noise) — so flipping either default is a product decision about
   whether the flip-flop/misfire fixes are wanted, not something blocked
   by strength data. Don't flip either default without being asked to.

### Standing rules (apply to any engineering work here, not just this trial)

- **Never push to `main`.** Work on a branch, push there, someone else
  reviews and merges by hand.
- **No search/live-behavior integration without a real SPRT.** A
  Python/stdlib simulation is useful supporting evidence, not a
  substitute — this project had one catastrophic failure early on from
  wiring an unvalidated hint directly into search, and treats that as
  the reason the rule exists.
- **Any change to `aegis_v4_runtime.rs` must keep passing** these three
  tests exactly (`cargo test -p unchessed-core --release`):
  `start_position_matches_python_reference`,
  `midgame_position_matches_python_reference`,
  `position_to_input_matches_hand_built_start_position`. They're the
  only thing proving the Rust forward pass matches
  `tools/reference_forward_unarchitectured_v1.py` numerically. Don't
  trade correctness for speed without re-checking against that reference.
- **State benchmark numbers as host-specific.** Never imply a speed
  number measured on one machine transfers 1:1 to another.
- **If an experiment fails a correctness/statistical gate, document the
  rejection — don't loosen the gate to make it pass.**
- **Don't claim verification you didn't actually run.** "Should compile"
  or "(need rustc)" as a footnote is not the same as a real
  `cargo test` pass — this project has been burned by that exact gap
  more than once, which is the whole reason this rustc trial exists.
- **Byte-exact file formats need `.gitattributes -text` rules.** Windows
  `core.autocrlf` has silently corrupted committed PGN/EPD/JSONL
  training data before by converting line endings on checkout.

### Build and test

```
cargo test --workspace --release          # Rust: engine, evaluators, adapter logic
python -m pytest tools/ -q                # Python: training/tooling tests
python tools/rust_bracket_check.py --all  # cheap bracket-balance sanity check, no rustc needed
```

A known-safe baseline right now: `cargo test --workspace --release`
should be 118/118 passing (6 ignored — real-hardware benchmarks, run
explicitly with `-- --ignored`), and `pytest tools/` should be in the
high 370s/380s passing with exactly one known, pre-existing failure
(`test_ddp_gloo_two_rank_smoke`, a Windows-only gloo limitation, not a
real regression, confirmed fine on Linux/WSL). If either baseline looks
meaningfully different after a clean clone of `main`, say so plainly —
that itself would be a finding.

### Key files to know about

- `scripts/research/arena_agent_unarchitectured_v1_runtime_speed_prompt.md`
  — the **living status/backlog document** for this project's ongoing
  engineering work. Read it for the actual current open-items list,
  round-by-round history, and the full pre-flight checklist referenced
  above. This is the single most useful file to read before doing any
  real engineering task here.
- `unchessed-core/src/nnue.rs`, `unchessed-core/src/aegis_v4_runtime.rs`,
  `unchessed-core/src/adapt.rs`, `unchessed-core/src/uci.rs` — the four
  files where almost all recent engineering work has happened.
- `tools/train_nnue.py`, `tools/nnue_train_control.py`,
  `tools/nnue_relabel_existing.py` — NNUE training/relabeling tooling.
- `docs/` — one markdown writeup per finding/round, all named
  descriptively (e.g. `nnue-v4-108m-recipe-result.md`,
  `unarchitectured-v1-why-the-hint-costs-elo.md`). Grep this directory
  before assuming something hasn't been tried.
- `scripts/setup-rust-toolchain.sh` — the toolchain bootstrap this trial
  is testing.

### If asked to do real engineering work later (not this trial)

Read the backlog doc above first, pick an item, and apply the same
discipline: build clean, run the real test suite, state what you
actually verified vs. what you're claiming without having checked, and
push to your own branch for review — never `main`.

### Where this project is genuinely weakest — the things most worth reinforcing

Ranked by how underserved they are relative to how much they matter,
not by ease:

1. **Raw playing strength ("brute force") has had almost no direct
   attention this whole project.** Every recent round has been NNUE
   retraining, the Unarchitectured v1 hint, or persona/adapter behavior
   — none of which have produced a strength gain yet (the hint never
   has; the best NNUE v4 retrain is still −155.6 Elo behind the shipped
   default). **Nobody has touched classic alpha-beta search tuning
   (`unchessed-core/src/search.rs`) this entire session** — late-move
   reductions, null-move pruning parameters, aspiration windows,
   extensions, quiescence search depth/margins, transposition table
   sizing/replacement policy. If the goal is a stronger engine rather
   than a more interesting one, this is the most neglected lever, not
   NNUE label noise or persona behavior. Any change here still needs a
   real SPRT, same as everything else.
2. **NNUE strength ceiling is real but not yet attacked at the right
   layer.** The evidence (round-13 through round-19 findings,
   `docs/nnue-v4-108m-recipe-result.md`,
   `docs/ieee-low-cp-val-mae-and-persona.md`) points at label noise from
   the 5000-node HCE search as the actual bottleneck, not architecture
   or data volume — more of the same labels won't help. The relabel tool
   (`tools/nnue_relabel_existing.py`) exists to test this directly
   (deeper search or self-distillation labels on the *same* positions)
   but the actual relabeling + retrain + SPRT hasn't been run yet. This
   is the best-supported unexplored lever for real Elo gain right now.
3. **Persona-stability claims are simulation-validated only, never
   confirmed in real play.** `PersonaSmooth`'s headline number (57.3%
   reduction in accidental mode-flips) and `EngineDetectV2`'s misfire
   fixes (Maia wrongly flagged as an engine, false ceiling/clock tells)
   both come from Python/stdlib simulations
   (`tools/persona_stability_sprt.py`, `tools/test_elo_detector.py`),
   not from watching real games. The one real cutechess SPRT that ran
   measured *Elo* (found neutral, ≈+2.1) — it did not and could not
   measure *flip rate* or *detection accuracy* directly, since cutechess
   doesn't log persona-internal state. If this is ever going to actually
   ship enabled, the flip-rate and misfire claims need a way to be
   measured from real game logs (not just PGN results), not just trusted
   from simulation. Nobody has built that measurement yet.
4. **GAB (the attention-bias mechanism in Unarchitectured v1) is
   provisioned at roughly a quarter of the capacity used in the
   comparable published architecture** (`docs/gab-capacity-finding.md`),
   and **weights sit 2.06x outside the int8 quantization range**
   (`docs/fishtest-and-quantization-notes.md`) — both real, measured,
   unaddressed findings from round 9. Both are retrain-gated (need the
   oracle checkpoint or a full retrain budget) and explicitly a project-
   owner decision, not something to act on unilaterally, but they're the
   most concrete named reasons the hint underperforms structurally, as
   opposed to just "not enough SPRT evidence yet."
5. **Oracle-side rating conditioning is an open, cheap, never-run
   experiment.** The student model's rating conditioning is confirmed
   inert (0/200 moves change across a 600→3200 Elo sweep,
   `docs/rating-conditioning-finding.md`), but whether the *oracle*
   (the larger model the student was distilled from) has the same defect
   is unknown — the oracle checkpoint isn't in this repo. If it ever
   becomes available, the same sweep methodology applies directly and
   would take CPU-minutes, not a real training run — it's cheap and
   just waiting on the checkpoint existing somewhere accessible.
