# Unarchitectured v1

## Canonical version policy

**Unarchitectured v1 is the first canonical version of the unified neural/search
architecture.** The names Hydra v1, Hydra Aegis v2-v4, Hydra Apex v5, and the
short-lived Apex v1 naming candidate are experimental lineage labels only. They are retained for reproducibility and
must not be presented as released product versions.

This naming reset does not fabricate maturity. Unarchitectured v1 is currently:

```text
architecture: frozen canonical v1
training pipeline: executed; calibrated checkpoint committed
trained checkpoint: artifacts/unarchitectured-v1-final.unarchv1
checkpoint SHA-256: 5fd9fc3fbf47bd2620c2e832e24c98525b59feeea791abf1c7ae32b9d311b16d
tensor container/exporter/inspector: implemented
Rust package loader and mixed-integer forward: implemented
full/middle/shallow Python/Rust parity: passed at documented tolerances
AVX2/FMA and AVX2 i16×i8→i32 runtime kernels: implemented
retained-int8 matrix forward: implemented; drift-gated against f32
production enablement: default-off
Elo/SPRT evidence: absent
```

## Architecture

Unarchitectured v1 canonically consists of:

- three-stage XT-NNUE with direct, x-ray, and pawn/king topology relations;
- calibrated fast/direct/full uncertainty routing;
- exact hypergraph delta oracle;
- promotion-aware legal action sets;
- elastic 2/128, 4/192, and 8/256 runtime-student exits;
- evidential WDL and per-action regret;
- private human/guide policy adapters;
- a 29M-878M GPU-profile-scaled training-only Oracle;
- Oracle-to-4.22M-student distillation;
- 1-8 GPU NCCL DDP training;
- Verda 4-360-vCPU exact common-budget teacher labelling; and
- mandatory alpha-beta legal fallback and safety vetoes.

The canonical runtime package magic is:

```text
UNARCHV1
```

A real calibrated package now exists and is accepted by the strict loader. It
remains unwired because deployment calibration, clock-budget integration,
chess-level runtime safety, and game gates are incomplete.

## Canonical files

```text
config/unarchitectured_v1.json
config/unarchitectured_v1_training.json
config/unarchitectured_v1_student.json
config/unarchitectured_v1_safety.json
docs/unarchitectured-v1.md
tools/unarchitectured_v1_base_data.py
tools/unarchitectured_v1_data.py
tools/unarchitectured_v1_uci_teacher_worker.py
tools/train_unarchitectured_v1_student_a100.py
tools/train_unarchitectured_v1_a100.py
tools/calibrate_unarchitectured_v1_throughput.py
tools/reference_forward_unarchitectured_v1.py
tools/unarchitectured_v1_position_encoding.py
tools/build_unarchitectured_v1_calibration_corpus.py
tools/calibrate_unarchitectured_v1_policy.py
tools/smoke_unarchitectured_v1_uci.py
tools/unarchitectured_v1_dataset_gate.py
tools/unarchitectured_v1_runtime_readiness.py
tools/unarchitectured_v1_safety.py
tools/unarchitectured_v1_watchdog.py
```

Current data, teacher-label, student, Oracle/distillation, calibration,
reference-forward, and readiness entry points now use canonical Unarchitectured
v1 names. The trainers, throughput calibrator, and Python reference require
NumPy/PyTorch; data, teacher, readiness, safety, and package tools remain
standalone on CPU ingestion hosts. Frozen `UNCHD3R0`/`UNCHD4R0` descriptors
retain predecessor-era text
because those bytes are wire identity. The Rust runtime filename
`aegis_v4_runtime.rs` remains an internal compatibility name, not a second
architecture.

## Experimental lineage

| Experimental label | Contribution carried into Unarchitectured v1 |
|---|---|
| Hydra v1 | compact XT/Chessformer split and parameter accounting |
| Hydra Aegis v2 | x-rays, pawn topology, elastic exits, evidential WDL |
| Hydra Aegis v3 | three-stage uncertainty, conformal bounds, semantic data ABI |
| Hydra Aegis v4 | complete legal sets, promotion identity, regret distributions |
| Hydra Apex v5 | large offline Oracle, distillation, Verda CPU/GPU scaling |

All predecessor configs and benchmark calculations are kept under their
original paths and explicitly marked experimental.

## Training safety inherited into canonical v1

Unarchitectured v1 includes the post-v1-experiment safeguards:

- cardinality-sized, globally shuffled, without-replacement epochs;
- profile-scaled minimum train and validation sets;
- three-epoch early stopping;
- disabled CUDA graphs;
- optimizer-inclusive per-rank VRAM probing;
- reserved-memory growth aborts;
- finite-metric checks;
- train/tune/final split audits;
- tuning-only regret calibration before final holdout; and
- fail-closed engine-runtime readiness.

The paid launcher refuses to run by default while the engine runtime pipeline
is missing. `ALLOW_RESEARCH_CHECKPOINT_ONLY=1` is an explicit acknowledgement
that the output will not be loadable by the engine.

## Autonomous safety without supervision

`config/unarchitectured_v1_safety.json` and
`tools/unarchitectured_v1_safety.py` implement a fail-closed controller. Every
training rank checks finite loss and pre-clip gradient norm. Any rank can abort
the complete DDP job. The controller maintains loss EMA, validation patience,
and a Page-Hinkley-style degradation CUSUM. It automatically chooses among:

```text
continue
preserve checkpoint + early stop
preserve checkpoint + abort all ranks
terminate stale process group + write incident report
refuse GPU launch
```

The trainer writes atomic heartbeats containing phase, epoch, step, loss,
gradient norm, learning rate, memory telemetry, and safety state.
`tools/unarchitectured_v1_watchdog.py` runs outside `torchrun`; if startup or
heartbeat deadlines expire, it terminates the complete process group, captures
GPU telemetry, and writes a durable incident report. No person must watch the
terminal.

Safety is layered before paid compute:

1. runtime/export readiness gate;
2. Rust/GPU feature-schema equality gate;
3. full dataset quality, deduplication, identity-disjointness, and date-order
   gate;
4. Verda hardware/VRAM/NVMe preflight;
5. reduced model self-check;
6. optimizer-inclusive VRAM probe;
7. internal numerical/memory/overfit controller; and
8. external stale-heartbeat watchdog.

## Faster cardinality-sized epochs

Unarchitectured v1 does not create an `int64` permutation of every record on
every DDP rank. It shuffles contiguous global batches, applies an epoch-specific
rotation, and gives each rank a disjoint contiguous slice. This preserves
without-replacement semantics while changing index memory from `O(records)` to
`O(records/global_batch)` and turning most mmap gathers into sequential NVMe
reads. Four asynchronous prefetch workers keep host reads ahead of GPU compute.

Epoch length is derived from actual cardinality. Larger Oracle profiles require
larger datasets, from 409,600 records for V100 to 16,384,000 for Blackwell.
Fixed `steps_per_epoch` is forbidden.

## Data generation and feature extraction safety

The Verda 4-360-vCPU scheduler produces exact common-budget legal-action labels
with executable, network, data, and output hashes. The autonomous dataset gate
then requires:

- minimum human and guide fractions;
- profile-sized record counts;
- low duplicate-position rate;
- exact game, player, and board-state separation across train/tune/final; and
- chronologically ordered provenance manifests.

`tools/unarchitectured_v1_architecture_audit.py` checks the canonical config,
package/runtime contracts, and current script entry points without importing
GPU-only dependencies. Dirty feature optimization may not ship unless it equals
the full-refresh oracle.

## Runtime package bridge

The first engine-integration layer is implemented:

- fixed 64-byte `UNARCHV1` header;
- fixed 200-byte tensor table entries;
- 64-byte payload alignment;
- model UUID;
- dtype, shape, scale, zero-point, flags, and per-section CRC32;
- whole table/payload CRC32;
- deterministic metadata section;
- symmetric int8 matrix export with float32 bias/norm tensors;
- strict Python inspector and tensor reconstruction report; and
- dependency-free Rust parser with bounds, alignment, shape, duplicate-name,
  UTF-8, and CRC validation.

The package bridge now feeds a complete mixed-integer Rust forward pass.
Dominant matrix tensors remain int8; activations are dynamically quantized to
int16, products accumulate into i32, and the tensor and activation scales are
applied once per output. Independent Python reference values remain within the
documented tolerances with identical best moves at every exit. The runtime
chess-safety capability remains false, so the paid launcher remains fail-closed.

The first sandbox round reduced the naive full forward from 208.61ms to
14.92ms. A round-two, same-process controlled benchmark measures the new
retained-int8 backend at 15.45ms versus 21.49ms for the dequantized backend on
one thread (1.39x), and 13.01ms versus 16.01ms on two threads (1.23x). The
latest two-thread exit ladder measured 5.21ms at 4/192 and 2.43ms at 2/128.
See `benchmarks/unarchitectured-v1/runtime-forward-2026-08-22.md`.

## Promotion gates

Unarchitectured v1 becomes usable only after all of the following pass:

1. calibrate all exits and the integer backend on a representative, disjoint
   deployment-position corpus (**done for the policy head in round 6** on 600
   over-the-board positions labelled by Stockfish 17.1, replicated on a further
   300; real but modest signal, and the WDL/regret heads calibrate poorly — see
   `docs/unarchitectured-v1-calibration.md`);
2. measure latency and search impact on the actual deployment CPUs;
3. validate the default-off nonblocking/clock-surplus UCI candidate;
4. confirm player/game/future-disjoint holdout and calibration provenance;
5. measure integrated engine depth and NPS under real clock budgets;
6. pass forced-mate and only-move safety suites;
7. pass separate paired-game gates for every exit/backend; and
8. complete owner-approved packaging/licensing decisions.

Until then, engines, GMs, uncertain opponents, FULL, PUNISH, and DEFEND continue
to use authoritative alpha-beta.

## Detailed engineering history

The retained runtime optimization, fail-closed integration, and teacher-labelled
calibration evidence is in `docs/unarchitectured-v1-runtime-optimization.md`,
`docs/unarchitectured-v1-integration-trial.md`, and
`docs/unarchitectured-v1-calibration.md`. Experimental predecessor names
remain historical lineage only; this document and the canonical configs above
define the current architecture.
