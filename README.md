# Unchessed AI

A Rust UCI chess-engine family with opponent adaptation, a production alpha-beta
core, and a research-grade neural training stack.

- **Unchessed Game Adapter** — estimates its opponent from play and chooses a
  guarded MATCH, PUNISH, CLINCH, DEFEND, or full-strength response.
- **Unchessed Game Reviewer** — uses the same move generation, evaluation, and
  search without persona weakening.

Both binaries are built from `unchessed-core`; neither depends on a network
service at runtime.

## Current status

| Component | Status |
|---|---|
| Legal move generation | Perft verified: start position depth 6 = 119,060,324; Kiwipete depth 5 = 193,690,690 |
| Search | Iterative deepening alpha-beta, qsearch, TT, null move, LMR, killers/history, MultiPV, exact shared node limits, `searchmoves`, mate limits, infinite/stop, and basic ponder support |
| Production evaluation | Shipped HalfKAv2_hm NNUE plus HCE fallback |
| NNUE execution | Int16 feature transformer, AVX-512BW/AVX2/scalar dispatch, stack-resident active features, incremental per-ply accumulators |
| Opening coverage | 3,810 CC0 historical lines, all 500 ECO codes, curated main/troll overlays, Polyglot support |
| Opponent model | 3,551 integer-Elo posterior buckets from 100 through 3650 with credible bounds |
| Human policy sidecar | Inference/training code exists; **no trained `unchessed-maia.bin` ships** |
| Canonical neural architecture | **Unarchitectured v1**; untrained, exporter/runtime incomplete, default-off |
| Production neural routing | Alpha-beta only until every model/runtime/game gate passes |

## Canonical architecture: Unarchitectured v1

**Unarchitectured v1 is the only canonical architecture name.** Hydra v1-v5 and
the short-lived Apex v1 name are experimental lineage retained for reproducible
ablation history.

Canonical identity:

```text
Registry ID:   unarchitectured-v1
Runtime magic: UNARCHV1
Architecture:  config/unarchitectured_v1.json
Training:      config/unarchitectured_v1_training.json
Safety:        config/unarchitectured_v1_safety.json
```

Primary documentation:

- [`docs/unarchitectured-v1.md`](docs/unarchitectured-v1.md)
- [`config/architecture_registry.json`](config/architecture_registry.json)
- [`benchmarks/unarchitectured-v1/`](benchmarks/unarchitectured-v1/)
- [`docs/full-scale-bug-audit-2026-08-21.md`](docs/full-scale-bug-audit-2026-08-21.md)
- [`docs/unarchitectured-v1-safety-integrity-report.md`](docs/unarchitectured-v1-safety-integrity-report.md)

### Architecture summary

```text
three-stage XT-NNUE
  position-only -> direct relations -> x-ray/pawn topology
                       |
          authoritative alpha-beta search
                       |
     exact promotion-aware legal action set
                       |
  elastic runtime student: 2/128, 4/192, 8/256
                       ^
  29M-878M training-only legal-action Oracle
```

Unarchitectured v1 includes:

- 32,400 direct threat/defence relations;
- 13,824 slider/blocker/behind-target x-ray hyperedges;
- 4,096 hashed pawn/king topology rows;
- exact full-refresh hypergraph delta oracle;
- fast/direct/full uncertainty routing and conformal bounds;
- promotion-aware legal actions for N/B/R/Q underpromotions;
- evidential WDL and per-action regret distributions;
- private human and guide policy adapters;
- 1-8 GPU Oracle training and compact-student distillation; and
- mandatory full-legal alpha-beta fallback.

Calling it v1 does **not** claim it is promoted. A real calibrated checkpoint,
strict package, Python reference, and numerically validated mixed-integer Rust
forward pass now exist; search integration, tactical safety, Elo, and SPRT
remain unpromoted.

### Runtime forward performance

The first optimization round reduced the validated full 8-layer/256-width Rust
forward from 208.61 ms to 14.92 ms on the two-visible-CPU sandbox through
AVX2/FMA, four-token matrix microkernels, cache blocking, and scoped
QKV/FFN/attention parallelism. The retained-int8 round now dynamically
quantizes activations to int16, performs the dominant matrix products as
AVX2 i16×i8→i32 arithmetic, and vectorizes activation quantization. In a
same-process controlled benchmark on this host it reduced one-thread latency
from 21.49 ms to 15.45 ms (1.39x) and two-thread latency from 16.01 ms to
13.01 ms (1.23x) versus the dequantized-f32 backend.

Full, middle, and shallow Python/Rust parity and best-move gates pass. The exits
remain unwired and still require deployment calibration, clock-budget,
tactical-safety, integrated NPS, and paired-game gates. See
[`benchmarks/unarchitectured-v1/runtime-forward-2026-08-22.md`](benchmarks/unarchitectured-v1/runtime-forward-2026-08-22.md)
and [`docs/unarchitectured-v1-runtime-optimization.md`](docs/unarchitectured-v1-runtime-optimization.md).

## Autonomous fail-closed safety

Unarchitectured v1 is designed to stop unsafe paid jobs without a person
watching the terminal.

The launch path applies, in order:

1. engine-runtime/export readiness gate;
2. Rust/config/GPU feature-schema equality gate;
3. dataset size, balance, deduplication, identity, position, and date-order gate;
4. Verda CPU/GPU/RAM/NVMe preflight;
5. reduced model self-check;
6. optimizer-inclusive per-GPU VRAM probe;
7. all-rank numerical, gradient, loss-spike, overfit, and memory-growth guards;
8. external stale-heartbeat process-group watchdog; and
9. atomic checkpoints plus durable incident telemetry.

The paid launcher fails by default because the engine runtime pipeline is not
yet complete. Research-only checkpoints require explicit acknowledgement:

```bash
export ALLOW_RESEARCH_CHECKPOINT_ONLY=1
```

Safety entry points:

```bash
python tools/unarchitectured_v1_runtime_readiness.py --strict
python tools/unarchitectured_v1_architecture_audit.py --strict
python tools/unarchitectured_v1_feature_audit.py --strict
python tools/unarchitectured_v1_dataset_gate.py --help
```

## Efficient data-centre training

### GPU support

Homogeneous Verda nodes with 1, 2, 4, or 8 GPUs are supported.

| GPU profile | Precision | Training-only Oracle | Minimum records |
|---|---:|---:|---:|
| Tesla V100 16GB | FP16 + GradScaler | 29,144,367 | 409,600 |
| A100 40GB / L40S / RTX 6000 Ada / RTX A6000 | BF16 | 58,412,431 | 1,024,000 |
| A100 80GB / H100 / RTX PRO 6000 | BF16 | 230,537,295 | 4,096,000 |
| H200 | BF16 | 501,835,855 | 10,240,000 |
| B200 / B300 / GB300 | BF16 | 878,114,575 | 16,384,000 |

Training uses NCCL DDP, non-final `no_sync()` accumulation, activation
checkpointing, fused AdamW, SDPA, per-rank auto-batching, and BF16 or V100 FP16.
The runtime student remains approximately 4.22M parameters on every profile.

### Faster epochs

Fixed `steps_per_epoch` is forbidden. Every epoch is derived from dataset
cardinality and uses rotated contiguous global-batch shuffling:

- without replacement;
- disjoint DDP rank slices;
- mostly sequential NVMe reads;
- four asynchronous mmap prefetch workers; and
- index memory `O(records/global_batch)` instead of `O(records)` per rank.

CUDA graphs are disabled after the earlier long-run OOM. Validation uses
three-epoch early stopping, finite-metric checks, and reserved-memory growth
limits.

### Verda CPU data generation

Automatic profiles cover 4, 8, 16, 32, 64, 96, 120, 180, and 360 vCPU nodes.
The scheduler reserves a small service allowance and assigns one persistent UCI
teacher per remaining vCPU by default. It supports NUMA affinity, resumable
ranges, exact per-legal-action common-budget labels, hash clearing, and complete
engine/network/data/output provenance.

```bash
export PLAN=/data/unchessed/guide/plan.json
export MANIFEST=/data/unchessed/guide/MANIFEST.json
scripts/training/verda_unarchitectured_v1_datagen.sh
```

Human policy records are generated with the Rust data tool:

```bash
cargo run --release -p unchessed-datagen -- \
  policy-v4 /data/human/train.aegis4 "$PRIVATE_128_BIT_HEX_KEY" \
  5000000 0.25 /data/pgn/*.pgn
```

Use the same private pseudonym key across train/tune/final mining so leakage can
be detected. Never commit the key or raw account identities.

### Full training launcher

```bash
export TRAIN_V5='/nvme/data/train/*.aegis4'
export TUNE_V5='/nvme/data/tune/*.aegis4'
export FINAL_V5='/nvme/data/final/*.aegis4'
export DATA_PROVENANCE=/nvme/data/provenance.json
export OUTPUT_DIR=/nvme/checkpoints/unarchitectured-v1

scripts/training/verda_unarchitectured_v1_train.sh
```

A provenance template is provided at
`config/unarchitectured_v1_data_provenance.example.json`.

## Production NNUE

The shipped `unchessed-nnue.bin` is separate from Unarchitectured v1 research:

```text
format version:       3
scheme:               HalfKAv2_hm
inputs:               22,528
accumulator:          256
parameters:           5,767,937
file size:            23,071,768 bytes
runtime FT storage:   approximately 11.53 MiB int16
```

The v4 network was SPRT-validated at +26.1 ± 12.4 Elo over the prior network.
The incremental mechanism was independently validated on main at +68.6 ± 21.0
Elo in its f32 implementation. This branch combines incremental updates with
int16/SIMD accumulation; that exact combined path retains a separate promotion
gate.

Inspect the shipped asset:

```bash
python tools/inspect_nnue.py unchessed-nnue.bin
```

## Adapter behavior and safety boundaries

- Fixed strength has absolute precedence when `UCI_LimitStrength` is enabled.
- Honest fixed-strength range is `UCI_Elo=100..2600`.
- Opponent posterior support remains `100..3650`; one-point buckets are not
  one-Elo accuracy, so decisions use credible intervals.
- Known engine identity and observed playing strength remain separate.
- Unknown opponents use conservative upper-strength evidence.
- Timing can modulate independently ceiling-level evidence; it cannot classify
  an opponent by itself.
- Auto trolling requires affirmative human evidence and is disabled for known
  engines.
- FULL, PUNISH, DEFEND, engine/GM, uncertain, and missing-model cases use
  authoritative alpha-beta.

Core UCI options:

| Option | Default | Meaning |
|---|---:|---|
| `Hash` | 128 | Transposition-table MiB |
| `MultiPV` | 1 | Number of analysis lines |
| `Adaptive` | true | Enable adapter opponent/persona logic |
| `UCI_LimitStrength` | false | Enable fixed-strength mode |
| `UCI_Elo` | 2400 | Fixed target, clamped to 100–2600 |
| `Contempt` | 25 | Draw aversion used by CLINCH |
| `Troll` | Auto | Off / evidence-gated Auto / forced On |
| `OwnBook` | true | Use embedded/external opening book |
| `BookFile` | — | Optional Polyglot book |
| `BookDepth` | 40 | Maximum available book plies |
| `PolicyFile` | auto | Optional `unchessed-maia.bin` sidecar |
| `UCI_Opponent` | — | Standard opponent descriptor |
| `RandomSeed` | 0 | Runtime entropy; nonzero is deterministic |

## Build and test

```bash
cargo build --workspace --release
cargo test --workspace --release
cargo test --workspace --release -- --ignored --nocapture
cargo clippy --workspace --release -- -D warnings
python3 -m unittest discover -s tools -p 'test_*.py'
```

Latest full local audit:

- 129 normal Rust tests passed;
- 5 deep/ignored Rust tests passed separately;
- 107 Python tests passed, with one A100 dependency-gated skip;
- release build and Clippy `-D warnings` passed;
- UCI smoke 9/9 passed;
- adversarial UCI and persona suites passed;
- 3,810 openings and all 500 ECO codes verified;
- shipped NNUE inspection and release packaging passed.

See [`docs/full-scale-bug-audit-2026-08-21.md`](docs/full-scale-bug-audit-2026-08-21.md).

Black-box checks:

```bash
python tools/uci_smoke.py ./target/release/unchessed-adapter
python tools/uci_edge_smoke.py ./target/release/unchessed-reviewer
python tools/persona_smoke.py ./target/release/unchessed-adapter
```

## Repository layout

```text
unchessed-core/       movegen, search, evaluation, TT, UCI, adaptation
unchessed-adapter/    adaptive UCI binary
unchessed-reviewer/   full-strength UCI binary
unchessed-datagen/    PGN/data-record generator
config/               engine, architecture, safety, and hardware profiles
tools/                validation, training, audit, packaging, and orchestration
scripts/training/     Verda CPU/GPU launchers
benchmarks/           calculated budgets and reproducible measured reports
docs/                 canonical specification and experimental research history
papers/                IEEE-style research guide
```

## Experimental lineage

The following are research history, not product versions:

- Hydra v1
- Hydra Aegis v2
- Hydra Aegis v3
- Hydra Aegis v4
- Hydra Apex v5
- Apex v1 naming candidate

Their configs, reports, and mathematics remain available for reproducibility.
`config/architecture_registry.json` is authoritative for naming.

## Known blockers

The `UNARCHV1` binary tensor container, checkpoint exporter, package inspector,
strict Rust package loader, and tensor reconstruction drift tool now exist.
Engine readiness remains blocked on:

```text
accuracy/calibration gates over a representative deployment-position corpus
safe asynchronous or clock-surplus search integration
runtime mate/only-move safety suite
integrated search NPS and paired-game SPRT
```

The mixed int16-activation/int8-weight Rust matrix backend, AVX2/FMA fallback
kernels, real checkpoint, package loader, and all-exit Python reference gates now
exist. The model is still unwired, and
`config/unarchitectured_v1_runtime_capabilities.json` keeps readiness false
until chess-level runtime safety and game gates are proven.

## Research paper

The full IEEE-LaTeX-styled engineering guide and source are under
[`papers/ieee-research-guide/`](papers/ieee-research-guide/).

## License and release note

The opening corpus is documented as CC0. The repository-wide project license,
tablebase distribution policy, and signing credentials remain owner decisions.
Use `tools/package_release.py` to create checksummed local bundles; use
`--require-policy` when heuristic policy fallback is unacceptable.
