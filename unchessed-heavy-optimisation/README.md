# Unchessed AI

A UCI chess engine family built from scratch in Rust: full-strength
eval/search plus the **Unarchitectured Metal** human-like policy layer, and a
growing, fully-validated training-data foundation for *level-conditioned*
human play (real humans across every rating band plus real Maia-3
self-play at random UCI elo limits). Legacy `v1` names remain only for
compatibility and historical experiment provenance.

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
| **Unarchitectured Metal** (policy prior) | ✅ canonical package and runtime lineage, Python/Rust parity-validated, int8-weights/int16-activations AVX2 backend, calibrated artifact at `artifacts/unarchitectured-metal-final.unmetal`. **Default-off as a UCI hint** — the current package remains gated by deployment calibration, integrated safety, and paired-game SPRT evidence |
| Adaptive adapter | ✅ live opponent-Elo model, MATCH/PUNISH/CLINCH/DEFEND personas, engine-tell detection, human-plausible move selection |
| Opening book | ✅ ~45 embedded main lines with ECO names + troll tier + external Polyglot `.bin` support |
| Training data | ✅ four committed, move-legality-validated corpora: `data/training/` (71,961 games, rating-banded), `data/training-elo/` (35,812 games, every 100-elo band 100-3200), `data/selfplay/` (real Maia-3 at random UCI elo), `data/archive/` (123,385 games, 1834-2022, era/theme curated) |
| Toolchain | ✅ zero external Rust deps; Python tooling from `tools/requirements-dev.txt`; sandbox build recipe in `docs/dev-environment.md` |

## AcceleratedDetection: shipped research implementation

The `manus/research-facilities` branch contains the current `unarchitectured-metal` opponent-detection research release. `AcceleratedDetection` is **default-off** and preserves the legacy detector when disabled. When enabled, promotion to `Mode::Full` is strict and resilient-channel-only: a suspected engine must satisfy sequential evidence fusion, resilient good-mass, catastrophic-error, clean-streak, and bounded-volatility guards. Clock tells and stable-fusion evidence remain diagnostic and cannot bypass the resilient proof obligation. This policy was designed to reduce false positives against human-like Maia-3 behavior while retaining sensitivity to strong classical engines.

The release also includes a probe-economics breakthrough for long games. Opponent move quality is measured by a real search before the engine's own move search, so the probe is expensive. While the detector is still forming a verdict, the probe uses depth 14 / 400,000 nodes and retains the depth-12 / 250,000-node fallback when necessary. After the public engine verdict has held for ten completed observations, the next probe downgrades to depth 9 / 60,000 nodes and suppresses the fallback search. The current observation is evaluated using the pre-observation saturation state, so a fresh clock tell cannot make its own evidence cheap. A later non-suspect observation resets saturation and restores high-fidelity probing. Telemetry exposes `suspect_streak`, `observation_saturated`, `probe_high_fidelity`, and `probe_saturated` so the behavior is directly auditable.

The implementation is intentionally dependency-free and interpretable: it combines bounded sequential evidence fusion, harmonic-mean agreement, leaky CUSUM-style accumulators, resilient evidence mass, held-verdict hysteresis, and clock-safe probe budgeting. The branch includes deterministic regression tests for accelerated detection, Maia-safe reason purity, catastrophic erratic rejection, held-verdict saturation, fresh clock-tell non-saturation, and accelerated-path compatibility.

### Real validation and reproduction

The final telemetry-enabled Stockfish 19 UCI smoke test used the committed `artifacts/unarchitectured-metal-final.unmetal` package, one thread, 64 MiB hash, real 60-second clocks, fixed openings, `Adaptive=true`, `OwnBook=false`, `AdapterTelemetry=true`, and `UCI_Opponent=- - human UnknownOpponent`. All four games produced 23 observations and zero low-time skips. Standard first-Full confirmation averaged 30 plies; accelerated confirmation averaged 25 plies; both accelerated confirmations used `legacy_accelerated_resilient`. One accelerated game recorded three saturated observations. An additional 80-ply real run recorded nine saturated observations in the standard arm. These are targeted real-UCI smoke tests, not claims of universal throughput or playing-strength improvement.

Recreate the experiments with [`docs/reproduce_resilient_detection_experiments.md`](docs/reproduce_resilient_detection_experiments.md), run the series-scoped driver at [`tools/run_unarchitectured_metal_asymmetric_latency.py`](tools/run_unarchitectured_metal_asymmetric_latency.py), and analyze telemetry with [`tools/analyse_unarchitectured_metal_asymmetric_latency.py`](tools/analyse_unarchitectured_metal_asymmetric_latency.py). The IEEE-style research paper is [`ieee-paper/accelerated_detection_asymmetric_latency.pdf`](ieee-paper/accelerated_detection_asymmetric_latency.pdf), with source at [`ieee-paper/accelerated_detection_asymmetric_latency.tex`](ieee-paper/accelerated_detection_asymmetric_latency.tex).

**Honest goals note:** beating full-strength Stockfish is not a realistic
outcome for any hand-built engine — Stockfish is 15+ years of distributed
testing. The achievable target this architecture is built for: play
convincingly at *any* human level from ~600 Elo to master+ and adapt
between them automatically. The eval/search ceiling keeps rising with
NNUE work; the human-play side is a level-conditioned retrain away — the
data and the design spec are in place (below), and the retrained net
still has to pass a paired-game SPRT before the Metal hint ever turns
on. The legacy `UnarchitecturedHint` spelling remains a compatibility
alias in the current UCI surface.

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
  Unarchitectured Metal runtime (loader, SIMD forward, hint integration).
- `unchessed-adapter/`, `unchessed-reviewer/` — the two binary crates
  (thin `main.rs` wrappers around `unchessed-core`).
- `unchessed-datagen/` — training-data generation for the neural
  components.
- `unchessed-nnue.bin` — the search evaluator (UNCHNNUE format,
  auto-loaded next to the exe; `EvalFile` to point elsewhere).
- `artifacts/unarchitectured-metal-final.unmetal` — the canonical
  Unarchitectured Metal package (`UNMETAL1` format; the loader also accepts
  historical `UNARCHV1` packages). The legacy `UnarchitecturedFile` option
  remains accepted as a compatibility alias.
- `config/` — canonical Unarchitectured Metal architecture, student/oracle,
  training, and safety configs, plus retained compatibility manifests and
  historical pretraining records where filenames still contain `v1`.
- `data/` — the four committed training corpora (below).
- `tools/` — the entire Python pipeline: data curation/validation,
  labeling, calibration, analysis, training, SPRT, and the cloud
  self-play generator (`tools/maia3_cloud_selfplay/`).
- `benchmarks/unarchitectured-metal/` — canonical host-specific
  instrumentation JSONs for runtime, calibration, integration, rating
  conditioning, and theme breakdown. Older `benchmarks/unarchitectured-v1/`
  references in historical notes are retained as provenance labels.
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
  `benchmarks/unarchitectured-metal/`).

**Why this exists** — `docs/research-notes-maia-levels-reverse-engineering.md`
reverse-engineered all three Maia generations from source: the strength
ladder is one unified model conditioned on **two discrete/continuous
skill inputs (self + opponent)**, not a scalar rating. That is the
design spec for retraining our policy net (our scalar rating input was
measured inert — below), and the dual-elo labels above are exactly what
that retrain trains on.

## Unarchitectured Metal (canonical; v1 is legacy nomenclature)

The canonical current architecture is a human-like **policy prior** —
64 board tokens, d512 transformer with GAB (Generalized Attention
Bases), a legal-move decoder, and policy/value/regret/concept heads —
shipped as the `UNMETAL1` package in
[`artifacts/unarchitectured-metal-final.unmetal`](artifacts/unarchitectured-metal-final.unmetal),
auto-loaded by the engine when its hint path is explicitly enabled. int8
package weights with dynamic int16 activations (i32 accumulation), AVX2/FMA
SIMD backend; the full 8-layer/256-wide forward was optimized from 208.61 ms
to 15.45 ms (alternating-round measurement) on the two-visible-CPU sandbox
(see `docs/unarchitectured-metal-runtime-optimization.md` and
`benchmarks/unarchitectured-metal/`). Python cross-check parity gates and
drift gates are frozen and pass in `cargo test`. The historical `UNARCHV1`
header and old `Unarchitectured*` option spellings are compatibility formats,
not the current product identity.

**What it is, honestly:** the net is structurally sound, validated, and
loadable — but as a *hint* it is **default-off** (`UnarchitecturedHint`
= false) for three measured reasons:

- [`docs/unarchitectured-metal-calibration.md`](docs/unarchitectured-metal-calibration.md)
  and the committed calibration reports — deployment calibration and
  runtime safety remain explicit gates.
- [`docs/gab-capacity-finding.md`](docs/gab-capacity-finding.md) — the
  GAB component remains a documented retraining target.
- [`docs/unarchitectured-metal-why-the-hint-costs-elo.md`](docs/unarchitectured-metal-why-the-hint-costs-elo.md)
  — the current hint path remains default-off until the measured quality,
  latency, and paired-game gates are cleared. Older `unarchitectured-v1`
  filenames in research notes are historical aliases for this same lineage.

The capability manifest
(`tools/unarchitectured_metal_runtime_readiness.py` and
`config/unarchitectured_metal_safety.json`) keeps production enablement gated
until the blockers list (provenance-disjoint deployment calibration,
deployment-CPU measurements, broad integrated depth/NPS plus tactical safety,
and isolated paired-game SPRT) is proven. See also
`docs/policy-prior-calibration.md`,
`docs/unarchitectured-metal-theme-breakdown.md`,
`docs/unarchitectured-metal-calibration.md`, and
`docs/unarchitectured-metal-integration-trial.md`.

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
| `PersonaSmooth` | false | EMA+dwell persona filter; **off until Adaptive-on SPRT** |
| `EngineDetectV2` | false | high-level Elo-detector retune; **off until Adaptive-on SPRT** |
| `AcceleratedDetection` | false | strict resilient-only engine promotion with sequential evidence fusion and Maia-safe calibration |
| `AdapterTelemetry` | false | opt-in machine-readable detector/persona telemetry, including saturation and probe profiles |
| `UnarchitecturedHint` | false | current binary-advertised spelling for the canonical Metal policy-prior hint; stays off until the calibration + SPRT gates pass |
| `UnarchitecturedFile` | — | current binary-advertised spelling for an explicit `UNMETAL1` package; default artifact is `artifacts/unarchitectured-metal-final.unmetal`; loader accepts historical `UNARCHV1` |
| `UnarchitecturedMinTime` | 30000 | minimum remaining clock (ms) before the Metal candidate may submit and wait up to 100 ms for a shallow exact-position hint |
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
   opponent's clock usage). With `AcceleratedDetection=true`, a suspected
   engine must pass the resilient sequential-evidence proof before it can
   promote Full; clock and stable-fusion channels cannot bypass that gate.
   Once the public verdict has held for ten observations, the opponent probe
   switches to the cheaper saturated profile so search time is not wasted on
   evidence that can no longer change the mode. Erratic play (brilliancies
   mixed with blunders) widens the model's uncertainty instead of narrowing
   it — the sandbagger pattern.
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
   (  `benchmarks/unarchitectured-metal/pretrain-probe-2026-08-28.json`).
   Remaining: the A100 runs (selfcheck first — the CUDA path is
   untested in the sandbox), then dual-elo student distillation +
   UNMETAL1 packaging, then `tools/analyse_rating_conditioning.py`
   (the 0/200 sweep must invert) and a paired-game SPRT before
   the Metal hint turns on.
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
- `python tools/unarchitectured_metal_runtime_readiness.py` — canonical Metal capability-manifest readiness report.
- `python tools/pretrain_move_dataset.py --labels … --out …` — build move-prediction pretrain shards (bridge + leakage guard).
- `python tools/pretrain_move_predictor.py --data …` — sandbox probe of the pretrain objective with the 0/200 conditioning sweep.
- `python tools/pretrain_v5_data.py build --pgn ... --out ...` — CPU stage: PGN -> v5 dual-elo shards (full + `--quality-filter` trusted-only) + validation.
- `python tools/pretrain_v1_a100.py selfcheck` — retained historical pretraining selfcheck; new work should use the canonical Metal training/export entry points.
- `scripts/pretrain-pipeline/cpu_stage.sh` / `gpu_stage.sh` — the two stages end-to-end, split by machine (see its README).
- `scripts/build-and-test.sh` — full gate set (build + tests + UCI smoke + matetrack).

## Docs index

- **Current UCI gates:** `round-16-uci-gates.md` (`PersonaSmooth` / `EngineDetectV2` default **false**; Metal hint remains gated; `AcceleratedDetection` is independently validated and default-off).
- **Live Elo detector (high-level misfires):** `elo-detector-high-level-integration.md` (Maia≠FULL, ceiling/clock tells retuned).
- **Persona stability / SPRT correlation:** `persona-stability-and-sprt-correlation.md` (EMA+dwell; 57% fewer mode flips in sim).
- **IEEE-style report (cloud speed/quality, persona-on):** `ieee-cloud-nnue-speed-quality.pdf` (LaTeX source `ieee-cloud-nnue-speed-quality.tex`).
- **IEEE-style report (val-MAE floors vs persona):** `ieee-low-cp-val-mae-and-persona.md`
  (simulation + committed SPRTs; sub-20 cp is NO-GO on current HCE labels;
  Adaptive stays on).
- **Findings (measured, honest):** `rating-conditioning-finding.md`,
  `gab-capacity-finding.md`,
  `unarchitectured-metal-why-the-hint-costs-elo.md`,
  `policy-prior-calibration.md`, `unarchitectured-metal-theme-breakdown.md`,
  `int8-activation-calibration-finding.md`,
  `parameter-calibration-audit.md`, `nnue-architecture-audit.md`,
  `nnue-v4-retrain-data-scaling-finding.md`,
  `nnue-v4-training-recipe.md`.
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
  `unarchitectured-metal-integration-trial.md`,
  `unarchitectured-metal-runtime-optimization.md`,
  `unarchitectured-metal-calibration.md`, `unarchitectured-metal.md`,
  `unarchitectured-metal-migration.md`.
