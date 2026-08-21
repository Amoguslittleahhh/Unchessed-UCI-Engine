# Unarchitectured v1

## Canonical version policy

**Unarchitectured v1 is the first canonical version of the unified neural/search
architecture.** The names Hydra v1, Hydra Aegis v2-v4, Hydra Apex v5, and the
short-lived Apex v1 naming candidate are experimental lineage labels only. They are retained for reproducibility and
must not be presented as released product versions.

This naming reset does not fabricate maturity. Unarchitectured v1 is currently:

```text
architecture: frozen canonical v1
training pipeline: implemented but not executed in this repository
trained checkpoint: absent
tensor container/exporter/inspector: implemented
Rust package loader: implemented
scalar and quantized neural forward: absent
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

No package using that magic may be emitted until the exporter, package
inspector, scalar Rust runtime, quantization drift gate, and runtime safety tests
exist.

## Canonical files

```text
config/unarchitectured_v1.json
config/unarchitectured_v1_training.json
config/architecture_registry.json
docs/unarchitectured-v1.md
benchmarks/unarchitectured-v1/
scripts/training/verda_unarchitectured_v1_train.sh
tools/unarchitectured_v1_architecture_report.py
tools/unarchitectured_v1_safety.py
tools/unarchitectured_v1_watchdog.py
tools/unarchitectured_v1_dataset_gate.py
tools/unarchitectured_v1_feature_audit.py
```

Legacy implementation filenames containing `v4` or `v5` remain internal until
a compatibility-preserving code migration is worthwhile. Canonical wrappers
and registry metadata determine product naming; filenames do not promote an
experimental predecessor.

## Experimental lineage

| Experimental label | Contribution carried into Apex v1 |
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

`tools/unarchitectured_v1_feature_audit.py` hashes and cross-checks the canonical
config, Rust direct/x-ray/topology extractor, GPU trainer constants, topology
hash multiplier, and exact-delta requirements before CUDA is allocated. Dirty
feature optimization may not ship unless it equals the full-refresh oracle.

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

This is a package bridge, not neural inference. Capability flags remain false
for scalar forward, quantized forward, exported reference vectors, SIMD, and the
runtime chess-safety suite. The paid launcher therefore remains fail-closed.

## Promotion gates

Unarchitectured v1 becomes usable only after all of the following pass:

1. export a real calibrated checkpoint and pass package/tensor drift inspection;
2. complete scalar Rust inference matching exported reference vectors;
3. SIMD equality tests;
4. float/quantized/Rust drift thresholds;
5. player/game/future-disjoint holdout metrics;
6. conformal calibration on a tuning split and coverage on final holdout;
7. deployment CPU latency and integrated engine NPS;
8. forced-mate and only-move safety suites;
9. separate paired-game gates for XT tiers and policy backends; and
10. owner-approved packaging/licensing decisions.

Until then, engines, GMs, uncertain opponents, FULL, PUNISH, and DEFEND continue
to use authoritative alpha-beta.

## Detailed engineering history

The complete derivations remain available in:

- `docs/unchessed-hydra-mathematics.md`
- `docs/unchessed-hydra-v2-mathematics.md`
- `docs/unchessed-hydra-v3-mathematics.md`
- `docs/unchessed-hydra-v4-mathematics.md`
- `docs/hydra-apex-v5-180core-a100.md`

Those documents are research history. This document and the architecture
registry define the canonical version name.
