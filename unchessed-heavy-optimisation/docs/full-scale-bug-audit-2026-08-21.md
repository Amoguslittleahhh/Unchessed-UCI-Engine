# Full-scale bug audit — 2026-08-21

## Executive verdict

| Area | Verdict |
|---|---|
| Existing Rust engine | **PASS locally**: release build, unit/deep tests, Clippy `-D warnings`, UCI edge tests, and persona tests pass |
| Data formats/generation | **PASS locally with fixes**: v3/v4 integrity tests, cross-language records, stable game/player pseudonyms, exact legal actions, and bounded-score rejection |
| V5 training control plane | **Conditionally ready for a staged research pilot**, not a full-cost run |
| V5 GPU numerical path | **UNVERIFIED**: this sandbox has no PyTorch, NumPy, CUDA, NCCL, or Verda GPU |
| V5 engine deployment | **BLOCKED**: exporter, inspector, quantization drift gate, Rust scalar runtime, and runtime safety tests do not exist |
| Strength claim | **NOT AVAILABLE**: no trained Unarchitectured Metal model, integrated NPS, Elo, or SPRT |

V5 cannot be guaranteed to train well before running it. This audit does show
that the two concrete failures that wasted the v1 run—replacement oversampling
from fixed epoch sizing and CUDA-graph memory growth—now have explicit
fail-closed controls.

## Scope

The audit covered 214 Rust, Python, shell, and JSON source/configuration files,
the production Rust workspace, canonical Unarchitectured Metal plus experimental Hydra v1-v5 reports, v3/v4 data ABIs, Verda CPU
and GPU profile resolution, exact UCI teacher labelling, multi-GPU Oracle and
student training control flow, packaging, opening coverage, and black-box UCI
behavior.

No proprietary suite, Verda instance, licensed data, GPU checkpoint, or remote
service credential was available or simulated.

## Executed checks

| Check | Result |
|---|---:|
| `cargo build --workspace --release` | PASS |
| Rust normal workspace tests | **129 passed, 0 failed, 5 ignored** |
| Rust ignored/deep tests, run separately | **3 passed, 0 failed** |
| `cargo clippy --workspace --release -- -D warnings` | PASS, zero warnings |
| `rustfmt --check` on changed Rust | PASS |
| Python unit tests | **107 passed, 0 failed, 1 dependency-gated module skipped** |
| Python bytecode compilation | PASS |
| Shell `bash -n` over training/SPRT scripts | PASS |
| Unarchitectured Metal and experimental Hydra v1-v5 deterministic report checks | PASS |
| UCI smoke | **9/9 passed** |
| Adversarial UCI edge smoke | PASS |
| Persona/identity/fixed-strength smoke | PASS |
| Opening book coverage | **3,810 lines, 500/500 ECO, CC0 metadata PASS** |
| Shipped NNUE inspector | PASS: 5,767,937 parameters, expected file size, finite/nonzero rows |
| Release package construction | PASS |
| JSON parse of repository configurations/reports | PASS |
| Credential/private-key pattern scan | No matches |
| TODO/FIXME/unimplemented marker scan | No matches |

The first black-box invocation failed because `cargo test` had built test
harnesses but not standalone release binaries. After explicit `cargo build
--workspace --release`, all black-box tests passed. This was an audit harness
precondition, not an engine failure.

## Findings fixed during this audit

### F-01 — Critical: v1 fixed epoch sizing caused immediate data reuse

**Old failure:** `steps_per_epoch` was independent of shard cardinality and
sampled with replacement. A small shard could be replayed many times inside one
nominal epoch, explaining validation degradation after epoch one.

**Fix:**

- removed fixed `steps_per_epoch` from Oracle and student configurations;
- epoch steps are `floor(records / effective_global_batch)`;
- one deterministic global permutation is reshaped across DDP ranks;
- records are consumed at most once per epoch;
- rank streams are disjoint;
- dropped tails and records consumed are reported;
- profile-scaled minimum datasets fail before full model allocation.

Minimum training sizes are now:

| Oracle profile | Parameters | Minimum records |
|---|---:|---:|
| V100 compatibility | 29,144,367 | 409,600 |
| 40-48GB base | 58,412,431 | 1,024,000 |
| 80-96GB large | 230,537,295 | 4,096,000 |
| H200 XL | 501,835,855 | 10,240,000 |
| Blackwell XXL | 878,114,575 | 16,384,000 |

These are mechanical lower bounds, not proof that the dataset is statistically
sufficient.

### F-02 — High: v1 CUDA-graph memory growth ended in epoch-29 OOM

**Fix:**

- Inductor CUDA graphs disabled by configuration and environment;
- compile mode reduced from `max-autotune` to `default`;
- real forward/backward/fused-optimizer VRAM probe retained;
- allocated, reserved, peak-allocated, and peak-reserved bytes logged per epoch;
- first two epochs establish a compilation/warm-up baseline;
- later reserved growth above 12% or 512 MiB aborts all ranks;
- latest atomic checkpoint is written before the abort;
- GPU-profile VRAM reserves remain 4-15%.

This prevents the known pattern but cannot rule out every CUDA driver/compiler
memory defect.

### F-03 — High: worsening validation could burn the remaining rental

**Fix:** three-epoch early stopping with a nonzero improvement threshold,
finite-metric checks, and atomic current/best checkpoints. A repeat of the v1
curve should stop near epoch four, not epoch 29.

### F-04 — Critical: training could produce another unusable checkpoint

**Status:** deployment implementation is still absent, but the spend hazard is
fixed. `tools/unarchitectured_metal_runtime_readiness.py --strict` checks for:

- quantized exporter;
- package inspector;
- Rust scalar runtime;
- quantization drift validator; and
- runtime safety tests.

The paid launcher now refuses to start because these files do not exist.
Research-only training requires explicit `ALLOW_RESEARCH_CHECKPOINT_ONLY=1`.

### F-05 — High: student final coverage used an uncalibrated zero quantile

**Fix:** added a dedicated `calibrate-student` stage on the tuning split. The
frozen calibrated checkpoint is then evaluated on the untouched final split.
If calibration is absent, coverage is omitted rather than reported under a
misleading zero-quantile value.

### F-06 — High: UCI lower/upper bounds could be stored as exact regrets

**Fix:** teacher score parsing rejects `lowerbound` and `upperbound` info lines.
A record fails if the completed action search never emits an exact score.

### F-07 — High: game pseudonym included sequence number

**Old risk:** duplicate games mined in a different input order could receive a
different hash, weakening split-leakage detection.

**Fix:** game identity now uses normalized site/date/round/players/result plus
canonical SAN movetext. Whitespace, comments, NAGs, and move-number formatting
are normalized. Player identities are trimmed and lower-cased before keyed
hashing.

### F-08 — Medium: benchmark provenance hash no longer matched source

**Fix:** reran the full Aegis feature benchmark against the current source and
updated its date, source SHA-256, and measured 2,264.2 ns/call result.

### F-09 — Medium: Rust static-analysis debt

**Fix:** removed all current Clippy warnings, including integer-bound style,
derivable defaults, ambiguous RNG naming, checked division, and scoped loop
lint annotations. Clippy now passes with `-D warnings`.

### F-10 — Medium: large Oracles shared the small-model data floor

**Fix:** minimum records and validation sizes now scale from the 29M V100 model
to the 878M Blackwell model.

### F-11 — High: canonical student depended on an experimental v4 config

**Fix:** expanded the canonical runtime-student architecture and added
`config/unarchitectured_metal_student.json`. Training and architecture reports no
longer require an experimental config for student shape or optimizer settings.

### F-12 — High: package readers disagreed on tensor shape/byte validation

**Fix:** Python and Rust now both require exact shape × dtype byte length,
positive finite scale, known flags, alignment, unique names, and checksums.
Checkpoint and package hashing is streamed rather than reading large artifacts
twice into memory.

## Open blockers and residual risks

### O-01 — Critical: mixed-integer runtime exists, safe engine integration does not

The real calibrated package, strict loader, complete mixed-integer forward,
independent Python vectors for every exit, AVX2 f32/integer kernels, complete
backend-drift gates, and best-move parity now exist. Still missing:

```text
representative deployment-position calibration for every exit/backend
nonblocking or clock-surplus search integration
runtime mate/only-move safety suite
integrated NPS and paired-game SPRT
```

The capability manifest keeps readiness false until those semantic capabilities
pass; file existence alone cannot unlock production routing.

### O-02 — High: the committed GPU run was not reproduced in this sandbox

A real calibrated student package is committed and its runtime outputs are
cross-checked. PyTorch and NumPy remain absent locally, so this audit did not
independently rerun:

- BF16/FP16 numerical behavior;
- GradScaler overflow recovery on V100;
- `torch.compile` plus activation checkpointing;
- NCCL DDP on 2/4/8 GPUs;
- `no_sync` accumulation;
- per-rank VRAM probes;
- 29M-878M parameter-count assertions against instantiated models;
- sustained memory stability; and
- Verda NVMe throughput.

### O-03 — High: no real Unarchitectured Metal dataset exists in this environment

There is no generated train/tune/final manifest, no player/game-disjoint audit
of the intended corpus, no measured human/guide mixture, and no actual common-
budget teacher labels from the selected engine/net assets.

### O-04 — High: no model-quality evidence

No Oracle/student holdout metrics, quantization drift, calibration coverage,
integrated engine latency, game Elo, or SPRT exists.

### O-05 — Medium: profile-scale minimums are floors, not sample-complexity proofs

A 16.384M-record Blackwell dataset contains many legal-action labels, but no
claim is made that it is enough for an 878M Oracle. Validation and scaling-law
pilots must decide that.

### O-06 — Medium: full permutations consume host RAM

Without-replacement epochs allocate one `int64` permutation per DDP rank. This
is about 8 bytes per record per process: 100M records on eight ranks could use
roughly 6.4GB aggregate host RAM for indices. Verda preflight reports host RAM,
but a future blockwise bijective shuffle would scale better.

### O-07 — External validation remains owner-operated

Licensed suites, long SPRTs, signing, proprietary data, and large current-engine
gauntlets remain outside this sandbox.

## V5 readiness decision

### Research checkpoint pilot

**Conditional GO**, only with `ALLOW_RESEARCH_CHECKPOINT_ONLY=1`, after Verda
preflight and real train/tune/final manifests pass. Start with one GPU and a
profile-minimum dataset, not 8x Blackwell.

### Full paid training intended to produce engine weights

**NO-GO** until O-01 is implemented. The launcher enforces this by default.

### Existing production engine

**GO for the currently shipped evaluator/search behavior under existing
claims.** This does not promote the Unarchitectured Metal neural path, which remains default-off.

## Required staged rollout

1. Calibrate all exits and the retained-int8 backend on representative,
   disjoint deployment positions.
2. Benchmark full and shallow paths on the actual deployment CPUs.
3. Add a nonblocking or clock-surplus-only root integration behind an option.
4. Run mate/only-move and exact-clock-budget safety suites.
5. Measure integrated NPS/depth before game testing.
6. Run isolated paired-game gates for each exit and routing policy.

## Reproduction commands

```bash
cargo build --workspace --release
cargo test --workspace --release
cargo test --workspace --release -- --ignored --nocapture
cargo clippy --workspace --release -- -D warnings
python3 -m unittest discover -s tools -p 'test_*.py'
python3 -m py_compile tools/*.py
bash -n scripts/training/*.sh scripts/sprt-history/*.sh
python3 tools/unarchitectured_metal_runtime_readiness.py --strict
```

The last command is expected to fail until the runtime pipeline is genuinely
implemented.
