# Hydra Apex v5: offline oracle, 4-360 vCPU data generation, and Verda GPU saturation

> **Experimental lineage document.** This is not a product version. The canonical architecture is [Unchessed Apex v1](unchessed-apex-v1.md).

## 1. The prominent upgrade

The strongest responsible upgrade is not to make the CPU engine run a huge
transformer at every move. It is to train a much stronger **offline oracle** and
distill everything it learns into the compact, alpha-beta-safe runtime model.

Apex v5 therefore has two neural scales:

```text
29M-878M-parameter offline oracle (training GPU only)
    16 x 512 board trunk
    4 x 512 full legal-action decoder
    policy + evidential WDL + score quantiles + regret distribution
                         |
             multi-objective distillation
                         v
4.22M-parameter elastic runtime student
    exits 2/128, 4/192, 8/256
    legal-only policy and regret
    CPU-oriented int8 target
                         |
              alpha-beta remains authoritative
```

This buys substantially more training capacity without permanently charging
engine NPS. It is a hypothesis that must still win holdout and game gates—not a
claimed Elo improvement.

### 1.1 V1 failure postmortem and v5 safeguards

V1 exposed two expensive failure modes: a fixed `steps_per_epoch` that repeatedly
sampled a small shard until it overfit almost immediately, and long-run CUDA
graph memory growth that ended in OOM. V5 now prevents those exact failures:

- epochs are derived from dataset cardinality, never a fixed step count;
- one global permutation is partitioned across DDP ranks, so records are used at
  most once per epoch instead of sampled with replacement;
- profile-scaled minimums fail before the full Oracle is allocated: 409,600
  records for V100 up to 16,384,000 for the 878M Blackwell Oracle;
- validation requires at least 50,000 records and separate shard paths;
- early stopping triggers after three non-improving validation epochs;
- non-finite metrics abort immediately;
- Inductor CUDA graphs are disabled by default;
- allocated/reserved/peak CUDA memory is recorded every epoch;
- reserved-memory growth above a 12%/512MiB envelope aborts after preserving the
  latest atomic checkpoint; and
- optimizer-inclusive microbatch probes retain a GPU-profile safety reserve.

This makes v5 much less likely to waste a full rental on the same bugs. It does
not prove model quality or eliminate every possible driver/compiler OOM.

The second V1 lesson remains unresolved: the repository still lacks a quantized
v5 exporter and Rust inference runtime. `tools/v5_runtime_readiness.py` therefore
makes the paid launcher fail closed. To run architecture research anyway, the
operator must explicitly set `ALLOW_RESEARCH_CHECKPOINT_ONLY=1`; such a
checkpoint is not engine-ready.

## 2. Why the oracle is different from simply enlarging v4

The v4 student compresses a position into 64 square states and scores legal
source/destination pairs. The v5 oracle gives every legal action its own token.
A legal token contains:

- source and destination embeddings;
- explicit N/B/R/Q promotion class;
- source and destination board-trunk states;
- private human/guide adapter state; and
- policy-only Elo, time class, and eight-ply history.

Four decoder layers perform masked legal-action self-attention and
cross-attention to the 64 board tokens. The oracle can therefore compare moves
directly: two queen retreats, four underpromotions, or multiple tactical
captures are represented as a set rather than independent dot products.

The board trunk stays free of player/history context. Its outputs are:

```math
\text{Dirichlet WDL evidence},
```

```math
Q_{0.1},Q_{0.2},\ldots,Q_{0.9}\text{ score quantiles},
```

and a 128-concept distribution. Each legal token predicts policy and
heteroscedastic regret. The oracle is never loaded by the UCI engine.

## 3. Calculated architecture budget

From `config/unchessed_hydra_v5.json`:

| Quantity | Calculated value |
|---|---:|
| Offline oracle | 58,412,431 parameters |
| Oracle BF16 weights | 111.41 MiB |
| FP32 weights+grad+AdamW persistent estimate | 0.87 GiB |
| Maximum-legal-set oracle forward | 11.76 GFLOP |
| Runtime student | 4,222,905 parameters |
| Oracle/student parameter ratio | 13.83x |
| Base CPU teacher topology | 176 workers + 4 service vCPUs on a 180-vCPU node |
| Aggregate teacher hash | 11.25 GiB |
| Base 40-48GB profile VRAM target/reserve | 90% / 10% |

The persistent model/optimizer state is small relative to 80 GB. V5 spends the
remaining memory on a larger measured microbatch and legal-token activations,
not on gratuitously enlarging runtime weights.

## 4. Data hierarchy

Do not treat every record as equal. Use three explicit tiers.

### 4.1 Human tier

Mine a large human corpus with the Rust `policy-v4` generator. These records
provide:

- actual human action;
- exact legal action set;
- rating, time class, history, WDL, and special-rule state; and
- keyed player/game identities for split auditing.

They intentionally contain no invented teacher regrets.

### 4.2 Gold guide tier

The v5 UCI worker searches **every legal action independently with the same node
budget**. With hash clearing enabled before every action:

```math
r(m)=\max_j V(j)-V(m)
```

has a common-budget meaning and is not contaminated by action order in the TT.
The default is 5,000 nodes per action. A position with 30 legal moves therefore
costs 150,000 teacher nodes. This is why the separate high-vCPU CPU node matters.

Use a strong, legally obtained UCI teacher and record its exact executable and
network/table asset SHA-256 values (`teacher.assets` in the CPU datagen config).
Stockfish requires its correct official NNUE files; do not substitute an
unavailable net and claim the resulting labels are Stockfish 18 labels.

### 4.3 Calibration/final tiers

Keep separate:

1. training games/players;
2. model-selection and conformal-calibration games/players; and
3. untouched final games/players.

Use the same private pseudonym key for all three so cross-split players can be
detected. A random record split from one flat shard is invalid.

## 5. Mining human records

On the CPU ingestion machine:

```bash
cargo build --release -p unchessed-datagen

target/release/unchessed-datagen policy-v4 \
  /data/unchessed/human-v4/human-000.aegis4 \
  "$PRIVATE_128_BIT_HEX_KEY" \
  5000000 0.25 /data/pgn/lichess-*.pgn

python3 tools/aegis_v4_data.py inspect \
  /data/unchessed/human-v4/human-000.aegis4
```

The key appears in process arguments in this basic CLI. Run on a trusted host
with shell history disabled or through a protected batch secret mechanism.
Never commit the key or raw player identities.

## 6. Exact UCI teacher worker

One worker can annotate a deterministic range:

```bash
python3 tools/v5_uci_teacher_worker.py \
  --engine /opt/stockfish/stockfish \
  --input /data/unchessed/human-v4/human-000.aegis4 \
  --output /data/unchessed/guide-v5/guide-000.aegis4 \
  --start 0 --count 1000 \
  --threads 1 --hash-mb 64 --nodes-per-action 5000
```

For each task it writes:

- an atomically completed output shard;
- engine/input/output SHA-256;
- exact UCI options;
- range and record count;
- legal actions labelled;
- node budget and hash-clearing policy; and
- elapsed records/actions per second.

A failed or partial output is never accepted as complete by the orchestrator.

## 7. Using Verda 4-360 vCPU nodes without oversubscription

The Verda selector exposes 4, 8, 16, 32, 64, 96, 120, 180, and 360 vCPUs.
`tools/verda_cpu_profile.py` reads the actual affinity mask and reserves a small
control-plane allowance before materializing the scheduler config:

| Node size | Pinned one-thread workers | Service vCPUs |
|---:|---:|---:|
| 4 | 4 | 0 |
| 8 | 7 | 1 |
| 16 | 15 | 1 |
| 32 | 30 | 2 |
| 64 | 60 | 4 |
| 96 | 92 | 4 |
| 120 | 116 | 4 |
| 180 | 176 | 4 |
| 360 | 352 | 8 |

These are Verda **vCPUs**, not a promise of the same number of physical cores.
Consequently, `physical_cores_only` is false; affinity IDs are treated as the
schedulable resource sold by the node. The UI's 4GB RAM label must be confirmed
on the deployed VM with preflight rather than assumed. At the default 64MiB hash
plus 64MiB process estimate, 176 workers reserve about 22GiB and 352 workers
about 44GiB; the orchestrator rejects layouts above 80% of detected RAM.

Independent single-thread searches remain the best initial layout for
per-action labels because they avoid parallel-search scaling loss. It is still
a starting point, not a measured result. On a 180-vCPU pilot compare 176x1 with
88x2 and 44x4; on 360 vCPUs compare 352x1 with 176x2 and 88x4. Keep the layout
with the highest labelled actions/second at identical nodes/action.

The orchestrator:

- reads the process affinity mask rather than assuming CPU IDs;
- optionally selects one logical CPU per physical core;
- keeps each worker's cores within one NUMA node;
- launches the engine only after child affinity is set, encouraging local
  first-touch hash allocation;
- requires UCI `Threads == cores_per_worker`;
- rejects hash/process memory above the configured RAM fraction;
- sets OpenMP/MKL/OpenBLAS/Rayon helper threads to one;
- divides input shards into deterministic, resumable ranges;
- reuses already checksum-verified outputs; and
- stops scheduling after the first worker failure.

Run:

```bash
CONFIG=config/v5_180core_datagen.json \
CPU_PROFILE_CONFIG=config/verda_cpu_profiles.json \
PLAN=/data/unchessed/guide-v5/plan.json \
MANIFEST=/data/unchessed/guide-v5/MANIFEST.json \
  scripts/training/verda_cpu_v5_datagen.sh
```

The launcher detects the selected node size automatically. Do not assume that
100% scheduler occupancy equals maximum labelled actions/second; the reserved
vCPUs prevent orchestration, NVMe completion, and kernel work from starving.

## 8. Storage and transfer

V4 records are 1,088 bytes. One million records occupy about 1.01 GiB before
filesystem overhead. Keep generation outputs on local NVMe; network filesystems
can serialize hundreds of workers on metadata and synchronous writes.

After finalization:

```bash
zstd -T0 -10 --long=27 guide-*.aegis4
sha256sum guide-*.aegis4.zst > TRANSFER.sha256
```

Transfer compressed shards and manifests to local NVMe on the GPU host, verify
checksums, then decompress before memory mapping. The current trainer consumes
fixed uncompressed records for predictable random access.

## 9. GPU VRAM utilization strategy

"Use all VRAM" should mean highest stable throughput, not an OOM at 99.9%.
Apex profiles target 85% on 16GB V100, 90% on 40-48GB devices, 93% on 80-96GB,
95% on H200, and 96% on B200/B300/GB300. The remaining memory covers CUDA
context, allocator fragmentation, compilation workspaces, and batch variance.

The auto-probe performs a real:

```text
forward -> all legal decoder layers -> loss -> backward -> fused AdamW step
```

for candidate microbatches. This includes optimizer-state allocation, which a
forward-only estimate misses. Binary search selects the largest tensor-core
aligned microbatch under the resolved profile ceiling. Every DDP rank probes;
the node uses the minimum safe result. Oracle+student distillation has its own
two-model probe.

Additional throughput choices:

- BF16 autocast on Ampere/Ada/Hopper/Blackwell, with FP16 GradScaler on V100;
- TF32 for residual FP32 matrix multiplications where supported;
- fused AdamW;
- Flash/efficient SDPA selected by PyTorch;
- activation checkpointing across the 16+4 oracle layers;
- fixed 218-action padding to avoid recompilation;
- background memory-map prefetch;
- pinned nonblocking host-to-device transfers in the existing data path;
- gradient accumulation to at least 4,096 effective records;
- `torch.compile(mode="max-autotune")` after checkpoint restore; and
- local NVMe rather than remote random reads.

FP8 is not enabled even on Hopper/Blackwell: this implementation prioritizes one
reproducible BF16 path over an unvalidated Transformer Engine branch. The
resolved Oracle fits on one device in every profile, so multi-GPU uses DDP rather
than FSDP; FSDP would add sharding communication without enabling a larger
checked-in model.

## 10. Verda-specific deployment profile

Both machines are on Verda. GPU instances and the separate 4-360-vCPU
labelling node have different jobs: do not run hundreds of UCI teacher workers
on a GPU VM. The profile resolvers support every family/size in the supplied
Verda selectors, while the CPU node remains dedicated to exact legal-action
labels.
Verda GPU overview: <https://verda.com/gpu-instances>.

Use an NVMe block volume in the same Verda location as each instance. Verda
lists up to 4.0 GB/s for NVMe block volumes versus up to 2.0 GB/s for its shared
filesystem, and its instance documentation requires attached/shared storage to
be in the same location as compute:

- <https://verda.com/products>
- <https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/>

Recommended layout:

```text
CPU instance NVMe: /data/unchessed/human-v4 and /data/unchessed/guide-v5
GPU instance NVMe: /data/unchessed/v5 and /data/checkpoints/hydra-apex-v5
Durable detached volume/object storage: manifests, compressed shards, checkpoints
```

The CPU selector is explicitly denominated in vCPUs. The checked-in 180-vCPU
base profile reserves four and launches 176 one-thread workers; the 360-vCPU
profile reserves eight and launches 352. Actual physical/SMT topology is still
recorded by preflight for throughput analysis, but is not misrepresented as 180
or 360 physical cores.

Run these before provisioning expensive jobs:

```bash
python3 tools/verda_v5_preflight.py --role cpu --data-path /data \
  --expected-logical-cpus 180 --strict --json verda-cpu-preflight.json

python3 tools/verda_v5_preflight.py --role gpu --data-path /data \
  --strict --require-torch --json verda-a100-preflight.json
```

The report records affinity-visible logical/physical CPUs, NUMA groups, RAM,
storage mount/free space, NVIDIA model/VRAM/driver, and PyTorch CUDA/BF16
support. It does not access or store Verda API credentials.

Verda supports detachable NVMe volumes, S3-compatible object storage, startup
scripts, and resumable transfers through its CLI/API. Keep checkpoints and
manifests on durable storage if choosing an interruptible spot instance; for a
long first oracle run, fixed PAYG capacity is operationally safer. CLI overview:
<https://docs.verda.com/cli/>.

### 10.1 GPU families and automatic Oracle scaling

Apex now supports homogeneous 1x, 2x, 4x, and 8x Verda nodes across every GPU
family shown in the Verda selector. `tools/verda_gpu_profile.py` reads
`nvidia-smi`, rejects mixed nodes, and materializes a resolved training config:

| Verda GPU family | Precision | Training-only Oracle parameters | Minimum records |
|---|---:|---:|---:|
| Tesla V100 16GB | FP16 + dynamic loss scaling | 29,144,367 | 409,600 |
| A100 40GB, L40S, RTX 6000 Ada, RTX A6000 | BF16 | 58,412,431 | 1,024,000 |
| A100 80GB, H100 80GB, RTX PRO 6000 96GB | BF16 | 230,537,295 | 4,096,000 |
| H200 141GB | BF16 | 501,835,855 | 10,240,000 |
| B200/B300/GB300 | BF16 | 878,114,575 | 16,384,000 |

The runtime student remains 4.22M parameters in every case. Expensive GPUs buy
a better-capacity teacher and faster exposure to more records, not a larger UCI
runtime model.

Multi-GPU uses one replicated model per GPU with NCCL DistributedDataParallel.
The model fits each listed single GPU profile, so DDP is faster and simpler than
FSDP here. Gradient accumulation uses `no_sync()` on non-final microbatches,
and effective batch calculations include world size. Each rank performs a real
VRAM probe; the minimum safe result is used across the node.

V100 automatically falls back to FP16 with `GradScaler`. Ampere, Ada, Hopper,
and Blackwell profiles use BF16. FP8 remains disabled: implementing a different
Transformer Engine path for only some devices would make cross-profile
reproducibility and quantization comparisons weaker.

More GPUs should noticeably reduce wall-clock time, especially from 1 to 2/4/8
on the 230M+ Oracle, but speedup is not promised to be linear. NCCL all-reduce,
NVMe input, compilation, and validation become limiting factors. More GPUs do
not create Elo by themselves; quality improves only if the saved time is spent
on more disjoint data, a larger resolved Oracle, or better ablations.

Resolve a profile manually:

```bash
python3 tools/verda_gpu_profile.py resolve \
  --profiles config/verda_gpu_profiles.json \
  --base-config config/a100_hydra_v5_training.json \
  --output resolved-gpu-training.json
```

The launcher performs this automatically and starts `torchrun` with every
visible GPU, up to eight. To use a subset, set `CUDA_VISIBLE_DEVICES` before
launch; mixed GPU models or mismatched VRAM are intentionally rejected.

## 11. Full multi-GPU run

```bash
export TRAIN_V5='/nvme/v5/train/*.aegis4'
export TUNE_V5='/nvme/v5/tune/*.aegis4'
export FINAL_V5='/nvme/v5/final/*.aegis4'
export OUTPUT_DIR=/nvme/checkpoints/hydra-apex-v5

scripts/training/verda_hydra_v5_multigpu_train.sh
```

Today this exits at the runtime-readiness gate because export/Rust inference is
not implemented. A deliberate research-only run requires:

```bash
export ALLOW_RESEARCH_CHECKPOINT_ONLY=1
```

The launcher performs, sequentially:

1. runtime/export readiness validation;
2. full record validation;
3. pairwise player/game leakage audits;
4. GPU-family detection and Oracle-profile resolution;
5. CUDA BF16/FP16 capability verification for all visible GPUs;
6. reduced oracle self-check;
7. per-rank real VRAM microbatch probe;
8. 1-8 GPU NCCL/DDP oracle training;
9. untouched oracle final evaluation;
10. separate oracle+student VRAM probe;
11. 1-8 GPU student distillation;
12. regret calibration on the tuning split;
13. untouched calibrated-student final evaluation; and
14. checkpoint/metric SHA-256 generation.

Running oracle and student jobs sequentially gives each phase the full visible
1-8 GPU node. Do not run XT, oracle, and student training concurrently.

## 12. Distillation objective

The runtime student learns more than hard best moves:

```math
L = 0.40L_{oracle-policy}
  + 0.18L_{oracle-WDL}
  + 0.17L_{oracle-regret}
  + 0.10L_{human}
  + 0.08L_{guide}
  + 0.05L_{exit-consistency}
  + 0.02L_{concept-statistics}.
```

The oracle also trains on WDL, score quantiles, per-action regret calibration,
and human/guide private adapters. Future XT distillation may use oracle score
quantiles, but v5 does not claim that integration is already present.

## 13. Production boundary

Even a much better oracle cannot bypass chess safety:

- oracle is absent at runtime;
- unknown opponents, engines, GMs, FULL, PUNISH, and DEFEND use alpha-beta;
- policy candidate sets only change ordering;
- all legal moves retain fallback search;
- history cannot affect board-state value;
- missing models fall back to alpha-beta; and
- every strength-changing path remains default-off until paired-game gates.

## 14. Evidence still required

This repository provides the architecture, exact label worker, 4-360-vCPU
scheduler, multi-GPU trainer/distiller, auto-batch probe, validators, and tests. It
does not claim that the separate systems have run.

Before any strength claim, report:

- actual topology pilot actions/second for one-, two-, and four-thread worker layouts;
- generated record counts and complete manifests;
- per-GPU peak allocated/reserved memory, NCCL scaling, and records/second;
- oracle and every student exit's human/guide/WDL/regret metrics;
- quantization drift;
- integrated CPU latency and NPS; and
- isolated paired-game SPRT results.
