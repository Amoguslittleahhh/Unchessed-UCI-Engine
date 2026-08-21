# Unchessed Apex v1

## Canonical version policy

**Apex v1 is the first canonical version of the unified neural/search
architecture.** The names Hydra v1, Hydra Aegis v2-v4, and Hydra Apex v5 are
experimental lineage labels only. They are retained for reproducibility and
must not be presented as released product versions.

This naming reset does not fabricate maturity. Apex v1 is currently:

```text
architecture: frozen canonical v1
training pipeline: implemented but not executed in this repository
trained checkpoint: absent
quantized exporter: absent
Rust neural runtime: absent
production enablement: default-off
Elo/SPRT evidence: absent
```

## Architecture

Apex v1 canonically consists of:

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
UNCHAPX1
```

No package using that magic may be emitted until the exporter, package
inspector, scalar Rust runtime, quantization drift gate, and runtime safety tests
exist.

## Canonical files

```text
config/unchessed_apex_v1.json
config/apex_v1_training.json
config/architecture_registry.json
docs/unchessed-apex-v1.md
benchmarks/apex-v1/
scripts/training/verda_apex_v1_train.sh
tools/apex_v1_architecture_report.py
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

Apex v1 includes the post-v1-experiment safeguards:

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

## Promotion gates

Apex v1 becomes usable only after all of the following pass:

1. quantized package exporter and deterministic inspector;
2. scalar Rust inference matching exported reference vectors;
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
