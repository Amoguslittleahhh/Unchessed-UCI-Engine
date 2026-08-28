# Unchessed AI

A UCI chess engine family built from scratch in Rust: full-strength
eval/search plus the **Unarchitectured v1** human-like policy layer, and a
growing, fully-validated training-data foundation for *level-conditioned*
human play (real humans across every rating band plus real Maia-3
self-play at random UCI elo limits).

- **Unchessed Game Adapter** (`unchessed-adapter`) — an adaptive engine
  that estimates its opponent's strength *live from their moves* (fully
  offline, no rating lookups) and shapes its play to match, punish, or
  clinch.
- **Unchessed Game Reviewer** (`unchessed-reviewer`) — a full-strength
  analysis engine: same `unchessed-core` eval/search as the adapter, but
  no persona/opponent-modeling layer — always plays at its raw strength.

Both binaries share the same `unchessed-core` crate (eval, search,
movegen, UCI protocol) and are always rebuilt/redeployed together after
any change; they differ only in the UCI options and default behavior
their own `main.rs` exposes. `unchessed-datagen` generates training
data for the neural components.

## Status: current milestone

| Component | State |
|---|---|
| Bitboard movegen | ✅ perft-verified (startpos d6 = 119,060,324; Kiwipete d5 = 193,690,690, exact) |
| Search | ✅ iterative deepening alpha-beta, quiescence, TT, null-move, LMR, killers/history, MultiPV, clock-aware time management; flag-free verified in 10s+0.1s blitz |
| NNUE evaluator | ✅ `unchessed-nnue.bin` (UNCHNNUE), SPRT-validated over the hand-crafted HCE fallback (which stays for missing-file use); incremental accumulator updates SPRT-validated +68.6 ± 21.0 Elo at real time controls |
| **Unarchitectured v1** (policy prior) | ✅ shipped, Python-parity-validated, int8-weights/int16-activations AVX2 backend, frozen parity gates (`benchmarks/unarchitectured-v1/`). **Default-off as a UCI hint** — measured findings, not a timeout: rating input inert (0/200 moves change, 600→3200), GAB provisioned at a quarter of the paper's smallest config, and the hint currently costs more than it adds |
| Adaptive adapter | ✅ live opponent-Elo model, MATCH/PUNISH/CLINCH/DEFEND personas, engine-tell detection, human-plausible move selection |
| Opening book | ✅ ~45 embedded main lines with ECO names + troll tier + external Polyglot `.bin` support |
| Training data | ✅ four committed, move-legality-validated corpora: `data/training/` (71,961 games, rating-banded), `data/training-elo/` (35,812 games, every 100-elo band 100-3200), `data/selfplay/` (real Maia-3 at random UCI elo), `data/archive/` (123,385 games, 1834-2022, era/theme curated) |
| Toolchain | ✅ zero external Rust deps; Python tooling from `tools/requirements-dev.txt`; sandbox build recipe in `docs/dev-environment.md` |

**Honest goals note:** beating full-strength Stockfish is not a realistic
outcome for any hand-built engine — Stockfish is 15+ years of distributed
testing. The achievable target this architecture is built for: play
convincingly at *any* human level from ~600 Elo to master+ and adapt
between them automatically. The eval/search ceiling keeps rising with
NNUE work; the human-play side is a level-conditioned retrain away — the
data and the design spec are in place (below), and the retrained net
still has to pass a paired-game SPRT before `UnarchitecturedHint` ever
turns on.

## Building

```sh
cargo build --release     # standalone binaries, no runtime dependencies
cargo test                # perft, search, book, model, parity gates
```

`scripts/build-and-test.sh` runs the full gate set (build + tests + UCI
smoke + matetrack). This repo has **zero external Rust dependencies**;
the environment recipe (including the sandbox's PyPI-distributed Rust
toolchain) is in [`docs/dev-environment.md`](docs/dev-environment.md).
The Python suite: `python -m pytest tools/ -q`.

## Repository layout

- `unchessed-core/` — shared eval, search, movegen, UCI protocol, and the
  Unarchitectured v1 runtime (loader, SIMD forward, hint integration).
- `unchessed-adapter/`, `unchessed-reviewer/` — the two binary crates
  (thin `main.rs` wrappers around `unchessed-core`).
- `unchessed-datagen/` — training-data generation for the neural
  components.
- `unchessed-nnue.bin` — the search evaluator (UNCHNNUE format,
  auto-loaded next to the exe; `EvalFile` to point elsewhere).
- `artifacts/unarchitectured-v1-final.unarchv1` — the Unarchitectured v1
  package (UNARCHV1 format, auto-located; `UnarchitecturedFile` to
  override).
- `config/` — v1 architecture spec, student/oracle configs, the
  pretrain config (`pretrain_v1_training.json`: dual-elo oracle,
  widened GAB, pinned 58,486,415 parameters), and the runtime
  capability manifest
  (`unarchitectured_v1_runtime_capabilities.json`).
- `data/` — the four committed training corpora (below).
- `tools/` — the entire Python pipeline: data curation/validation,
  labeling, calibration, analysis, training, SPRT, and the cloud
  self-play generator (`tools/maia3_cloud_selfplay/`).
- `benchmarks/unarchitectured-v1/` — host-specific instrumentation JSONs
  (runtime forward, calibration, integration trial, rating
  conditioning, theme breakdown) with a
  [`README`](benchmarks/unarchitectured-v1/README.md).
- `scripts/` — `build-and-test.sh`, `exhibition/` (game runners),
  `nnue-pipeline/` (cloud NNUE training scripts), `pretrain-pipeline/`
  (CPU/GPU split for the move-prediction retrain), `sprt-history/`
  (committed SPRT launchers — a record of specific past experiments,
  mix of passed and reverted), `research/` (research prompts/notes).
- `docs/` — findings and research notes (indexed at the bottom).
- `nnue-shards-safe/` — reference NNUE shards + research brief.

SPRT gate logs/PGNs and exhibition-series results themselves are not
committed (large, regenerated per-run).

## Training data & the level-conditioning pipeline

Four committed corpora, all with pinned source provenance (manifest
sha256s per file) and full per-game move-legality validation (the
python-chess 1.11 dropped-token rule — a game that logs even one
warning is dropped, never silently kept):

| Set | Size | What it is |
|---|---|---|
| [`data/training/`](data/training/README.md) | 71,961 games | rating-banded blocks: lichess 2022 mega-clean (6 mean-elo bands), TWIC issues 400/1000/1649, WCC 1990/1993, Carlsen & Nakamura archives, Bundesliga/British championship |
| [`data/training-elo/`](data/training-elo/) | 35,812 games | **every 100-elo band 100-3200** (+3300 overflow) from a full scan of 3,267,641 rated lichess games; first 2,000 per band verbatim. The honest shape: real rated games below mean ~1000 essentially don't exist (28 in 3.9 GB) and above mean ~2900 are equally rare (81) |
| [`data/selfplay/`](data/selfplay/README.md) | 200 games, 13,076 labeled moves | **real Maia-3** (the official platform's ONNX, pinned mirror) vs itself, each side a uniform random UCI elo 100-3200 (1-elo accuracy); the measured conditioning gradient: top-1 confidence 0.323 at elo 100-199 rising to ~0.52-0.62 at 1900-3200 |
| [`data/archive/`](data/archive/README.md) | 123,385 games | era/theme breadth 1834-2022: world championships in all eras, national championships, women's chess, ICCF correspondence, GM-annotated classics |

The pipeline around them (`tools/`):

- `training_blocks.py` / `archive_blocks.py` / `build_elo_bands.py` —
  fetch (pinned), split, validate, clean, verify; all reproducible from
  a fresh clone.
- `selfplay_elo_mixer.py` — the Maia-3 random-elo generator (the
  committed 200-game set); `maia3_cloud_selfplay/` — its many-core cloud
  scale-out, default **5,000,000 games** from a four-engine pool
  (Maia-3 + Stockfish 18 + LC0 v0.32.1 + RubiChess, each with the
  strength mechanism it actually supports — native UCI_Elo for
  Maia-3/Stockfish, thinking-budget and NPS-cap ladders for LC0/Rubi,
  the latter two labelled `EloQuality: approximate`), per-game
  deterministic substreams, resident per-worker engine pools, fsync'd
  resume, built-in full validation + conditioning calibration, and a
  Verda-AI-targeted README with measured cost (5M mixed ≈ 95-110 h /
  ≈$205-240 on a 180-vCPU node; pilot command gives the real rate).
- `build_level_conditioned_moves.py` — turns any of these sets into
  per-move `(FEN, level-window, move, elo_self, elo_oppo)` labels
  (Maia-style both-players windows; 800,971 rows from `data/training/`
  committed as profile + deterministic sample in
  `benchmarks/unarchitectured-v1/`).

**Why this exists** — `docs/research-notes-maia-levels-reverse-engineering.md`
reverse-engineered all three Maia generations from source: the strength
ladder is one unified model conditioned on **two discrete/continuous
skill inputs (self + opponent)**, not a scalar rating. That is the
design spec for retraining our policy net (our scalar rating input was
measured inert — below), and the dual-elo labels above are exactly what
that retrain trains on.

## Unarchitectured v1

The canonical current architecture: a human-like **policy prior** —
64 board tokens, d512 transformer with GAB (Generalized Attention
Bases), a legal-move decoder, and policy/value/regret/concept heads —
shipped as the `UNARCHV1` package in
[`artifacts/unarchitectured-v1-final.unarchv1`](artifacts/unarchitectured-v1-final.unarchv1),
auto-loaded by the engine. int8 package weights with dynamic int16
activations (i32 accumulation), AVX2/FMA SIMD backend; the full
8-layer/256-wide forward was optimized from 208.61 ms to 15.45 ms
(alternating-round measurement) on the two-visible-CPU sandbox (`docs/unarchitectured-v1-runtime-optimization.md`,
`benchmarks/unarchitectured-v1/runtime-forward-*.json`). Python
cross-check parity gates and drift gates are frozen and pass in `cargo
test`.

**What it is, honestly:** the net is structurally sound, validated, and
loadable — but as a *hint* it is **default-off** (`UnarchitecturedHint`
= false) for three measured reasons:

- [`docs/rating-conditioning-finding.md`](docs/rating-conditioning-finding.md)
  — the rating input does nothing: **0/200 moves change from 600 to
  3200** (max logit perturbation 0.004), with a suspiciously linear
  response = one scalar diluted through a 32-wide projection. `POLICY_HUMAN`
  vs `POLICY_GUIDE` moves 4/200.
- [`docs/gab-capacity-finding.md`](docs/gab-capacity-finding.md) — the
  GAB component is provisioned at a quarter of the paper's smallest
  configuration.
- [`docs/unarchitectured-v1-why-the-hint-costs-elo.md`](docs/unarchitectured-v1-why-the-hint-costs-elo.md)
  — full-exit top-1 0.255 vs a 0.157 free MVV-LVA heuristic, p90
  centipawn loss 422: the hint currently costs more than it adds.

The capability manifest
(`config/unarchitectured_v1_runtime_capabilities.json`) keeps
`runtime_safety_suite: false` until the blockers list (provenance-disjoint
deployment calibration, deployment-CPU measurements, broad integrated
depth/NPS + tactical safety, isolated paired-game SPRT) is proven. See
also `docs/policy-prior-calibration.md`,
`docs/unarchitectured-v1-theme-breakdown.md`,
`docs/unarchitectured-v1-calibration.md`, and
`docs/unarchitectured-v1-integration-trial.md`.

**The unblock path** (all retrain-only; a retrained net still needs its
own SPRT): (1) level-conditioned retrain with the dual-elo data above —
conditioning design per the Maia reverse-engineering note; (2) widen GAB
to at least the paper's 5M configuration; (3) theme-balanced sampling;
(4) weight clipping so the result is quantizable. Verification gate:
`tools/analyse_rating_conditioning.py` — the 0/200 sweep must *invert*
before the hint is trusted.

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 128 | transposition table MB |
| `Threads` | (autodetect) | search threads (1-64) |
| `MultiPV` | 1 | analysis lines shown |
| `Adaptive` | true | the whole adapter brain; off = always best move |
| `UCI_LimitStrength` / `UCI_Elo` | false / 2400 | hard cap on playing strength (500-3200) |
| `Contempt` | 25 | drive to win drawish games (fuels CLINCH mode) |
| `Troll` | Auto | `Off` / `Auto` (model-gated) / `On` (forced clowning) |
| `OwnBook` | true | use the opening book in games |
| `BookFile` | — | path to any Polyglot `.bin` book |
| `BookDepth` | 16 | max plies to stay in book |
| `PolicyFile` | auto | legacy per-rating policy net path; weights not committed — without the file MATCH mode uses the built-in heuristic priors and says so in the log |
| `EvalFile` | auto | path to the UNCHNNUE evaluator; default `unchessed-nnue.bin` next to the exe |
| `UCI_Opponent` | — | standard GUI-supplied opponent info; seeds the model for engines |
| `UnarchitecturedHint` | false | experimental root-ordering candidate; stays off until the retrain + SPRT gates above pass |
| `UnarchitecturedFile` | — | explicit `UNARCHV1` model package (default: auto-located `artifacts/unarchitectured-v1-final.unarchv1`) |
| `UnarchitecturedMinTime` | 30000 | minimum remaining clock (ms) before the candidate may submit and wait up to 100 ms for a shallow exact-position hint |
| search terms | — | `Aspiration*`, `Futility*`, `LMR*`, `NullMove*`, `Probcut*`, `RFPMargin`, `RookPct`, `MobilityPct`, `KnightOutpostPct`, `PassedPawn*Pct`, `ProbcutSeeFilter` — the calibrated search-term suite (see `tools/check_search_param_consistency.py`) |

## How the adapter thinks

1. **Pre-game:** if the GUI sends `UCI_Opponent`, known engines
   (Stockfish, Leela, Komodo, …) seed the model at their real strength —
   trolling is hard-locked off against strong engines. Humans always
   start neutral: declared ratings are never trusted as truth.
2. **Live model:** every opponent move is compared against the engine's
   own analysis; centipawn loss (weighted by position difficulty, book
   moves discounted) feeds a Bayesian running Elo estimate that
   converges in ~8-12 moves and keeps tracking.
3. **Personas** (selection only — the search underneath always runs
   full strength): **MATCH** blends to the opponent's level with
   human-plausible moves; **PUNISH** snaps to forcing best moves the
   moment they blunder; **CLINCH** picks venomous, trap-laden lines in
   drawish late games (contempt wired into the search so drawn lines
   score negative while chasing a win — neutral again when DEFENDING);
   **DEFEND** digs in when worse. Transitions have hysteresis, and every
   change is logged: `persona MATCH -> PUNISH (eval 990 cp, opponent
   ~1011)`.
4. **Engine-tell detection:** near-instant, near-perfect replies in
   positions with real choice raise a suspicion score (fed by the
   opponent's clock usage). A suspected engine gets full-strength chess
   and zero trolling; erratic play (brilliancies mixed with blunders)
   widens the model's uncertainty instead of narrowing it — the
   sandbagger pattern.
5. **UCI_Elo semantics:** with `UCI_LimitStrength` on, the engine plays
   *at* `UCI_Elo` in every mode, matching standard UCI behavior.
6. **Book:** popularity-weighted theory with ECO names; a separately-
   tagged troll tier (Bongcloud, Scholar's mate attempts, Stafford,
   Fried Liver, …) gated by the live Elo model — big game detected →
   mainlines only, and a bail-out guard eval-checks the position before
   continuing any troll line.

## The NNUE evaluator

`unchessed-nnue.bin` (auto-loaded next to the exe, `EvalFile` to point
elsewhere, falls back to HCE if absent/unreadable) — the search
evaluator, trained via `tools/train_nnue.py` (format/gradient sanity:
`train_nnue.py selfcheck`). Historical SPRT record for the current net:
+107.1 ± 27.0 Elo over the hand-crafted eval (532 games, LOS 100%),
incremental accumulator updates +68.6 ± 21.0 Elo (657 games, LOS 100%)
at real game time controls. The cloud training pipeline lives in
`scripts/nnue-pipeline/`; reference shards in `nnue-shards-safe/`.
Quantization notes: `docs/fishtest-and-quantization-notes.md`,
`docs/int8-activation-calibration-finding.md`,
`docs/nnue-architecture-audit.md`.

## En Croissant setup

**Engines → Add new → Local** and point it at the built
`unchessed-adapter` binary (release). Play against it on a new game; the
engine log panel narrates every adapter decision via
`info string [Unchessed] ...` lines (opponent estimate, persona
switches, book/troll choices).

## Roadmap

1. **Level-conditioned retrain of the policy net** — now built as a
   two-stage move-prediction pipeline
   (`docs/move-prediction-pretrain-plan.md`,
   `scripts/pretrain-pipeline/`): stage 1 pretrain = next-move
   prediction (legal-only cross-entropy, 16-bit action encoding in
   the 4096×5 = 20480 vocabulary) on the whole mixed corpus with
   **dual-elo conditioning** — the objective that forces the level
   axis to be informative (the v1 single-rating input was measured
   inert, 0/200); stage 2 fine-tune on the trusted-only subset
   (calibrated + native + human, no approximate ladder rows) at a
   lower LR. The CPU/GPU split: `cpu_stage.sh` builds v5 dual-elo
   shards (`tools/pretrain_v5_data.py`, STM-normalized encoding
   matching `unchessed-datagen`, game-disjoint splits, target-in-
   legal guard) on the 180-vCPU box; `gpu_stage.sh` runs the
   dual-elo oracle (58.5M params, GAB widened to the paper's 5M
   config per `gab-capacity-finding.md`) on one A100 with the 0/200
   conditioning sweep as a per-epoch gate metric. Sandbox probe of
   the objective: 118/200 flips (v1: 0/200), 1.9× baseline accuracy
   (`benchmarks/unarchitectured-v1/pretrain-probe-2026-08-28.json`).
   Remaining: the A100 runs (selfcheck first — the CUDA path is
   untested in the sandbox), then dual-elo student distillation +
   UNARCHV1 packaging, then `tools/analyse_rating_conditioning.py`
   (the 0/200 sweep must invert) and a paired-game SPRT before
   `UnarchitecturedHint` turns on.
2. **Runtime:** lazy SMP threads, pondering, tablebases, adaptation
   tuning. NPU dispatch stays experimental-unimplemented (CPU-only
   inference; the case for why: `docs/npu-viability-285h.md`).
3. **Reviewer:** full-strength UCI engine + PGN review CLI (move
   classification, accuracy %).

## Development verification tools

- `python tools/uci_smoke.py <engine>` — 9-step UCI protocol conformance test.
- `python tools/selfplay.py <engine> [games] [movetime]` — self-play sanity run.
- `python tools/rust_bracket_check.py` — balanced-bracket lint for the Rust sources.
- `python tools/pentanomial_sprt.py` — the SPRT decision tool (Fishtest mathematics).
- `python tools/unarchitectured_v1_runtime_readiness.py` — capability-manifest readiness report.
- `python tools/pretrain_move_dataset.py --labels … --out …` — build move-prediction pretrain shards (bridge + leakage guard).
- `python tools/pretrain_move_predictor.py --data …` — sandbox probe of the pretrain objective with the 0/200 conditioning sweep.
- `python tools/pretrain_v5_data.py build --pgn ... --out ...` — CPU stage: PGN -> v5 dual-elo shards (full + `--quality-filter` trusted-only) + validation.
- `python tools/pretrain_v1_a100.py selfcheck` — GPU stage: tiny dual-elo oracle + real loader + conditioning sweep (CPU or CUDA; first command on the A100 box).
- `scripts/pretrain-pipeline/cpu_stage.sh` / `gpu_stage.sh` — the two stages end-to-end, split by machine (see its README).
- `scripts/build-and-test.sh` — full gate set (build + tests + UCI smoke + matetrack).

## Docs index

- **Findings (measured, honest):** `rating-conditioning-finding.md`,
  `gab-capacity-finding.md`,
  `unarchitectured-v1-why-the-hint-costs-elo.md`,
  `policy-prior-calibration.md`, `unarchitectured-v1-theme-breakdown.md`,
  `int8-activation-calibration-finding.md`,
  `parameter-calibration-audit.md`, `nnue-architecture-audit.md`.
- **Research notes:** `research-notes-maia-levels-reverse-engineering.md`
  (the level-ladder mechanism — the retrain design spec),
  `research-notes-moe-mcts-2401.16852.md`, `research-notes-moe-2507.11181.md`,
  `research-notes-vrzina-engine-thesis.md`,
  `research-survey-arxiv-2026-08-24.md`,
  `stockfish-empirical-data-notes.md`, `fishtest-and-quantization-notes.md`,
  `llm-uci-matrix-assessment.md`, `move-prediction-pretrain-plan.md`
  (the two-stage retrain design + sandbox probe).
- **Performance:** `performance-survey-2026-08-24.md`,
  `performance-round-1-implementation.md`,
  `performance-ceiling-and-gpu-viability.md`,
  `memory-hierarchy-notes-285h.md`, `npu-viability-285h.md`,
  `tuning-core-ultra-9-285h-and-low-end.md`.
- **Process:** `dev-environment.md` (toolchain, incl. the sandbox's
  PyPI-distributed Rust toolchain), `workspace-reset-recovery.md`,
  `engine-commit-feed-review.md`, `history-answers-reconciliation.md`,
  `full-scale-bug-audit-2026-08-21.md`,
  `unarchitectured-v1-integration-trial.md`,
  `unarchitectured-v1-runtime-optimization.md`,
  `unarchitectured-v1-calibration.md`, `unarchitectured-v1.md`.
