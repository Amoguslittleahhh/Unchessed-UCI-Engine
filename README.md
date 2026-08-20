# Unchessed AI

A UCI chess engine family built from scratch in Rust:

- **Unchessed Game Adapter** (`unchessed-adapter.exe`) — an adaptive engine that
  estimates its opponent's strength *live from their moves* (fully offline, no
  rating lookups) and shapes its play to match, punish, or clinch.
- **Unchessed Game Reviewer** (`unchessed-reviewer.exe`) — a full-strength
  analysis engine: same `unchessed-core` eval/search as the adapter, but no
  persona/opponent-modeling layer — always plays at its raw strength.

Both binaries share the same `unchessed-core` crate (eval, search, movegen)
and are always rebuilt/redeployed together after any change; they differ only
in the UCI options and default behavior their own `main.rs` exposes.

## Research paper

The complete IEEE-LaTeX-styled engineering and research guide is available as
[`papers/ieee-research-guide/unchessed-research-guide.pdf`](papers/ieee-research-guide/unchessed-research-guide.pdf),
with its LaTeX source, renamed IEEEtran derivative, build script, evidence
hashes, and SHA-256 manifest in the same directory.

## Status: barebones milestone (this build)

| Component | State |
|---|---|
| Bitboard movegen | ✅ perft-verified (startpos d6 = 119,060,324; Kiwipete d5 = 193,690,690, exact) |
| Search | ✅ iterative deepening alpha-beta, quiescence, TT, null-move, LMR, killers/history, MultiPV, time management; research options for IIR, history gravity, countermoves, razoring, and LMP are implemented default-off pending match gates |
| Eval | ✅ NNUE v4 (HalfKAv2_hm: 22,528 inputs, 256 accumulator, 5,767,937 parameters), SPRT-validated +26.1 Elo over v1; f32 files load as int16 with AVX-512BW/AVX2/scalar dispatch and ply-indexed incremental accumulators; HCE remains the fallback/explanation layer |
| UCI protocol | ✅ worker-thread loop with strict FEN/state validation, exact shared node limits, `searchmoves`, `mate`, `go infinite`, `ponder`/`ponderhit`, deadline charging, and adversarial smoke coverage |
| Time management | ✅ clock-aware: deep searches on a full clock, urgency tiers as time drains (near-instant on the increment in panic mode), situation-scaled budgets (sharp/wide positions get more), easy-move early stop, score-drop extensions; verified flag-free in 10s+0.1s blitz |
| Opening book | ✅ 3,810 CC0 named historical lines covering all 500 ECO codes, plus 45 curated mainlines, 15 risk-graded troll lines, offbeat historical variety, and external Polyglot `.bin` support |
| Adapter brain | ✅ live opponent-Elo model, MATCH/PUNISH/CLINCH/DEFEND personas, human-plausible move selection, `UCI_Opponent` engine identification |
| Human policy net | ⚠️ Pure-Rust Maia-style v2 inference and training pipeline are implemented, but no trained `unchessed-maia.bin` ships in this repository; production therefore uses heuristic priors unless a sidecar is supplied |

**Honest goals note:** beating full-strength Stockfish is not a realistic outcome
for any hand-built engine — Stockfish is 15+ years of distributed testing. The
achievable target, which this architecture is built for: play convincingly at
*any* level from ~600 Elo to master+ strength and adapt between them
automatically. NNUE + search work later raises the ceiling.

## Building

```
cargo build --release
```

Produces `target\release\unchessed-adapter.exe`. The executable itself has no
shared-library dependency, but production NNUE/policy behavior requires its
model sidecars. Use `tools/package_release.py` to create a checksum manifest
and `--require-policy` when heuristic fallback is unacceptable.

Tests (perft suite, search, book, model): `cargo test`. Deep perft:
`cargo test --release -- --ignored`.

## Repository layout

- `unchessed-core/` — shared eval, search, movegen, UCI protocol logic used
  by both binaries.
- `unchessed-adapter/`, `unchessed-reviewer/` — the two binary crates
  (thin `main.rs` wrappers around `unchessed-core`).
- `unchessed-datagen/` — training-data generation tooling.
- `tools/` — auxiliary scripts (NNUE trainer, etc.).
- `scripts/exhibition/` — the game-runner scripts used for exhibition
  matches against reference engines (RubiChess).
- `scripts/sprt-history/` — smoke-test/SPRT-launch scripts committed as a
  record of specific past eval/search experiments (mix of passed and
  reverted techniques — check project memory or `scripts/sprt-history/`
  filenames for which).

SPRT gate logs/PGNs and exhibition-series results themselves are not
committed to git (they're large and regenerated per-run) — see the WSL
build environment's `~/unchessed-ai/results/README.md` for that data.

## En Croissant setup

The engine has been registered in En Croissant's engine list automatically
(`%APPDATA%\org.encroissant.app\engines\engines.json`). If you need to do it
manually: **Engines → Add new → Local** and point it at
`target\release\unchessed-adapter.exe`.

- **Play vs it:** Board → New game → pick *Unchessed Game Adapter* as one side.
- **Watch it think:** open the engine log panel — every adapter decision is
  narrated via `info string [Unchessed] ...` lines (opponent estimate, persona
  switches, book/troll choices).

> If Windows **Smart App Control** blocks a freshly rebuilt exe (error 4551),
> that's an OS policy on unsigned new binaries — the currently built release
> exe passed it. If it ever triggers after a rebuild, re-run the build once
> (reputation re-check) or run the test suite via WSL.

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 128 | transposition table MB |
| `MultiPV` | 1 | analysis lines shown |
| `Adaptive` | true | the whole adapter brain; off = always best move |
| `UCI_LimitStrength` / `UCI_Elo` | false / 2400 | hard cap on playing strength |
| `Contempt` | 25 | drive to win drawish games (fuels CLINCH mode) |
| `Troll` | Auto | `Off` / `Auto` (model-gated) / `On` (forced clowning) |
| `OwnBook` | true | use the opening book in games |
| `BookFile` | — | path to any Polyglot `.bin` book (e.g. built from Lichess masters) |
| `BookDepth` | 40 | maximum plies available; effective depth is scaled down for weaker opponents to match human book-exit behaviour |
| `PolicyFile` | auto | path to a policy weights file; default: `unchessed-maia.bin` next to the exe |
| `UCI_Opponent` | — | standard GUI-supplied opponent info; seeds identity/strength priors |
| `RandomSeed` | 0 | 0 uses runtime entropy; non-zero makes persona/book choices reproducible for tests |

## How the adapter thinks

1. **Pre-game:** if the GUI sends `UCI_Opponent`, identity and current playing
   strength are tracked separately. Known/computer opponents are anti-troll
   locked even when deliberately limited to a low declared Elo; an omitted
   engine rating uses the known-engine prior. Human declarations are treated
   only as broad priors and are verified from play. The descriptor persists
   across `ucinewgame` while per-game evidence resets.
2. **Live model:** every opponent move is compared against the engine's own
   analysis; centipawn loss (weighted by position difficulty, book moves
   discounted) updates a 3,551-bucket posterior with one bucket per integer Elo
   from 100 through 3650. Decisions use credible bounds—not false one-Elo
   certainty—and the model keeps tracking as play changes.
3. **Personas** (selection only — the search underneath remains full):
   **MATCH** draws a heavy-tailed intended error from a human ACPL curve and
   selects the closest common-depth root candidate, weighted by human priors;
   **PUNISH** snaps to forcing best moves the moment they blunder
   (plays found mates immediately, prefers simplifying captures when far
   ahead); **CLINCH** picks venomous, trap-laden lines in drawish late games
   (narrow-safe-path metric, keeps queens on, and wires *contempt into the
   search itself* so drawn lines score negative while chasing a win — but
   neutral again when DEFENDing, because then a draw is a rescue); **DEFEND**
   digs in when worse. Transitions have hysteresis (enter/exit thresholds)
   so the engine commits to a plan instead of flapping, and every persona
   change is logged: `persona MATCH -> PUNISH (eval 990 cp, opponent ~1011)`.
4. **Engine-tell detection**: strength is load-bearing; timing is not. The
   model tracks lag-1 autocorrelation of log clock-fraction, but regular timing
   can only lower the evidence threshold for an opponent already playing at
   the measurement ceiling. Fast premoves or weak regular play can never flag
   a human by themselves. Erratic play widens uncertainty, and positive engine
   classification is game-latched to prevent mode flapping. An independent,
   account-disjoint public-data validation did **not** validate timing as a
   standalone classifier (account AUC 0.413, 95% CI 0.260–0.575), so this safety
   restriction remains mandatory.
5. **UCI_Elo semantics**: with `UCI_LimitStrength` on, fixed strength has
   absolute precedence and the engine plays *at* `UCI_Elo` (100–2600),
   matching standard UCI behavior. Style personas cannot bypass the fixed
   strength selector.
6. **Book:** popularity-weighted theory with ECO names; a separately-tagged
   troll tier (Bongcloud, Scholar's mate attempts, Stafford, Fried Liver, …)
   gated by the live Elo model — big game detected → mainlines only, and a
   bail-out guard eval-checks the position before continuing any troll line
   (`troll line refuted — back to real chess`).

## The human policy net

When a `PolicyFile` sidecar is supplied, MATCH move priors use Maia-style
policy networks: one net per rating bucket (<1300, 1300–1599, 1600–1899,
1900+), trained to predict *what a human of that rating actually plays* on
non-bullet rated games from
the [Lichess open database](https://database.lichess.org) (CC0). At play time
the engine blends the two buckets nearest its current target rating, so "play
like a 1450" uses genuinely 1450-flavoured move preferences, not random noise
on top of engine moves. If the weights file is missing, the engine falls back
to the built-in heuristic priors and says so in the log.

The v2 nets see the full special-rule state (castling rights + en-passant
square as explicit inputs; en-passant and promotion samples oversampled in
training). Validation top-1 accuracy predicting real human moves, per bucket:

| Bucket | overall | castling | en passant | promotion |
|---|---|---|---|---|
| <1300 | 29.2% | 53.4% | 18.8% | 76.7% |
| 1300–1599 | 31.0% | 60.7% | 44.7% | 79.3% |
| 1600–1899 | 32.8% | 74.1% | 61.4% | 79.5% |
| 1900+ | 33.5% | 81.5% | 67.4% | 75.8% |

(That en-passant accuracy *rising with rating* is real human behavior — weaker
players genuinely miss or decline en passant, and the per-bucket nets
reproduce that.)

Pipeline (reproducible):

```
cargo run --release -p unchessed-datagen -- samples 4000000 0.5 games1.pgn ...
python tools/train_policy.py samples unchessed-maia.bin 3 256 4000000
# put unchessed-maia.bin next to unchessed-adapter.exe
```

Debug helper: type `policy 1200` (or any rating) into the engine's stdin to
see the net's top human moves for the current position.

## The NNUE evaluator

`unchessed-nnue.bin` (auto-loaded next to the executable, or selected with
`EvalFile`) is the v4 HalfKAv2_hm network: 22,528 inputs × 256 accumulator,
5,767,937 parameters, and an exact f32 file size of 23,071,768 bytes. At load
time the feature transformer and bias are validated, quantized with scale 511
to int16 (~11.5 MB runtime table), overflow-bounded, and dispatched once to
AVX-512BW, AVX2, or scalar accumulation. Active features and accumulators are
stack-resident, removing the former two heap allocations per evaluation. The
small scalar output head remains f32 to avoid reassociation drift.

The first flat-768 NNUE was SPRT-validated at +107.1 ± 27.0 Elo over HCE; v4
was subsequently validated at +26.1 ± 12.4 Elo over that network. Search now carries ply-indexed NNUE state: ordinary moves add/subtract only
changed feature rows, while a perspective refreshes when its own king changes
HalfKA bucket/orientation. Special-move and depth-3 tree tests require exact
agreement with full refresh. An isolated same-tree depth-10 run measured
1.215x geometric-mean NPS with exact nodes/scores and 12/12 move agreement.
Run `python tools/inspect_nnue.py
unchessed-nnue.bin` to verify the shipped architecture and parameter count.

```
python tools/train_nnue.py selfcheck              # format/gradient sanity check
python tools/train_nnue.py unchessed-nnue.bin 15 shard0.bin shard1.bin ...
```

## Roadmap (next rounds)

- **Reviewer**: full-strength UCI engine + PGN review CLI (move classification,
  accuracy %).
- **NNUE**: SPRT the measured int16 score drift and incremental path, then
  retrain on stronger distilled labels and evaluate accumulator-layout tuning.
- Deeper policy nets (conv/resnet) once GPU training is available; more data,
  more buckets.
- Lazy SMP threads, pondering, tablebases, adaptation tuning.

## Development verification tools

- `python tools/uci_smoke.py <engine>` — 9-step UCI protocol conformance test.
- `python tools/uci_edge_smoke.py <engine>` — malformed-FEN, node-limit, searchmoves, infinite, and ponder regressions.
- `python tools/persona_smoke.py <adapter>` — black-box identity persistence, fixed-strength, timing, and anti-troll regressions.
- `python tools/selfplay.py <engine> [games] [movetime]` — self-play sanity run.
- `python tools/build_balanced_manifest.py --config config/elo_sampling.json --output balanced.jsonl <pgn...>` — build a player-capped manifest with one source-rating cell per integer Elo from 100–3650, without copying PGNs.
- `python tools/check_opening_coverage.py` — verify 3,810 named lines and all 500 ECO codes.
- `python tools/inspect_nnue.py unchessed-nnue.bin` — verify file format, dimensions, parameter count, finite values, and runtime quantized size.
- `python tools/bench_research.py <reviewer> [--baseline <reviewer>]` — self-consistent fixed-depth throughput/quality benchmark that refuses the adaptive binary.
- `python tools/timing_classifier_validation.py validate --config config/timing_validation.json --records data/timing-validation/records.jsonl --manifest data/timing-validation/source-manifest.json --json data/timing-validation/report.json --markdown data/timing-validation/result.md --check` — reproduce the account-disjoint timing-signal validation and verify the committed negative result.
- `python tools/service_timing_bench.py ...` — aggregate real Lichess, Chess.com, and FICS timing exports without writing usernames, game IDs, or moves.
- `python tools/uci_epd_suite.py --engine <reviewer> --epd <licensed-suite.epd> --movetime 10000 --json result.json` — run public or user-licensed coordinate-move EPD suites with pinned UCI settings and checksums.
- `python tools/train_nnue_xt_a100.py selfcheck` / `python tools/train_chessformer_a100.py selfcheck --no-compile` — validate the Hydra v1 A100 XT-NNUE and Chessformer candidates before long jobs.
- `python tools/train_nnue_xt_v3_a100.py selfcheck --config config/a100_hydra_v3_training.json` — validate Aegis v3's three-stage position/direct/full XT trainer with direct, x-ray, and pawn/king topology groups. A full run requires separate train, calibration, and final-validation shards.
- `python tools/train_chessformer_v4_a100.py selfcheck --config config/a100_hydra_v4_training.json --no-compile` — exercise v4's nested 2/128, 4/192, and 8/256 legal-only policy, evidential WDL, private history adapters, and per-action regret heads.
- `cargo run --release -p unchessed-datagen -- policy-v4 ...` — produce schema-headed `UNCHD4R0` human-policy shards containing all promotion-aware legal actions and privacy-keyed game/player pseudonyms.
- `python tools/aegis_v4_data.py inspect ...` / `audit-split ...` — reject malformed v4 legal sets, target/regret inconsistencies, and player/game leakage before training.
- `python tools/summarize_engine_gauntlet.py --candidate Unchessed --provenance benchmarks/real-engines/provenance.json --pgn benchmarks/real-engines/games/*.pgn --json benchmarks/real-engines/report.json --markdown benchmarks/real-engines/result.md --check` — verify the committed 48-game Ethereal/Berserk/Stockfish gauntlet.
- `python tools/package_release.py --target-dir target/release --output release [--require-policy]` — bundle binaries/models with SHA-256 manifest and optional policy hard requirement.
- `docs/opponent-detection-and-balanced-data.md` — opponent identity/type/strength architecture, safe behavior policy, balanced-data design, and rollout gates.
- `docs/timing-classifier-validation.md` — CC0 source provenance, pseudonymous extraction, account-level statistics, failed gates, and safe production decision.
- `docs/commercial-and-service-validation.md` — measured service coverage, unavailable commercial APIs, and lawful proprietary-suite workflow.
- `docs/real-engine-testing.md` — controlled real-engine match conditions, results, provenance rules, and current-asset limitations.
- `docs/nnue-xt-chessformer-hybrid.md` — compact SFNNv10-inspired threat-residual NNUE and persona-routed Chessformer/alpha-beta design, implementation contract, microbenchmark, and gates.
- `docs/a100-training-guide.md` — BF16/TF32 A100 setup, data contracts, checkpointing, memory tuning, holdouts, and promotion gates.
- `docs/unchessed-hydra-mathematics.md` — complete equations for the unified XT-NNUE/Chessformer architecture, joint losses, alpha-beta integration, quantization, risk routing, and promotion gates.
- `docs/unchessed-hydra-v2-mathematics.md` — next-level Aegis design with x-ray/pawn hypergraphs, uncertainty-gated XT evaluation, elastic Chessformer exits, evidential WDL, concept transport, and gradient-conflict control.
- `docs/unchessed-hydra-v3-mathematics.md` — implementation-focused Aegis v3: three-stage XT, dual aleatoric/epistemic uncertainty, conformal bounds, exact x-ray/topology extractors, private temporal policy adapters, a frozen 160-byte data ABI, calibrated exits, and alpha-beta vetoes.
- `docs/unchessed-hydra-v4-mathematics.md` — promotion-aware legal-set prediction, shared nested exits, evidential WDL, per-action regret distributions, proof-aware candidate ordering with mandatory full legal fallback, and an exact hypergraph-delta oracle.
- `docs/hydra-apex-v5-180core-a100.md` — 29M-878M training-only legal-action Oracles, 4-360-vCPU exact UCI labelling, NUMA/affinity scheduling, adaptive GPU VRAM probing, and compact student distillation.
- `python tools/verda_cpu_profile.py resolve ...` / `tools/v5_180core_datagen.py` / `tools/v5_uci_teacher_worker.py` — resumable guide/regret generation across every advertised Verda CPU-node size.
- `python tools/verda_v5_preflight.py --role cpu|gpu ...` — verify Verda affinity/NUMA, 1-8 supported GPUs, BF16/FP16 PyTorch, RAM, and NVMe before paid jobs.
- `python tools/verda_gpu_profile.py resolve ...` — select a 29M-878M training-only Oracle for V100, A100, L40S/Ada/A6000, H100/H200, RTX PRO 6000, or B200/B300/GB300 nodes.
- `python tools/train_hydra_oracle_v5_a100.py selfcheck ...` — validate the Apex oracle and use `torchrun`-backed `train-oracle`/`distill-student --auto-batch` on 1-8 GPUs.
- `python tools/aegis_v3_data.py inspect ...` / `audit-split ...` — validate `UNCHD3R0` shards and reject game/player leakage before A100 training.
- `docs/opening-book-coverage.md` — historical source, tier safety, coverage, weighting, and external-corpus guidance.
