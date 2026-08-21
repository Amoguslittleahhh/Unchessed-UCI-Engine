# A100 80GB training guide

This pipeline is optimized for one NVIDIA A100 80GB. It trains two independent
research checkpoints:

1. `UNCHNNX4_TRAINING_V1`: the 256-wide HalfKAv2 positional accumulator plus a
   32-wide compact threat residual and eight phase heads;
2. `UNCHFORM_TRAINING_V1`: an eight-layer, approximately 4.2M-parameter Chessformer policy with 64
   square tokens, dynamic Geometric Attention Bias, continuous Elo context, and
   a source/destination policy head.

Neither checkpoint is a production sidecar. Export and Rust inference remain
gated on quantized validation. The joint mathematical target, cross-branch
losses, factorized threat export, and runtime package are specified in
`docs/unchessed-hydra-mathematics.md`; the two trainers here first stabilize the
branches independently before joint distillation. The original scripts remain
Hydra v1 trainers. Aegis v3 now has a separate XT trainer,
`tools/train_nnue_xt_v3_a100.py`, which implements direct, x-ray, and pawn/king
feature groups, position/direct/full heads, dual fast uncertainty heads, and a
mandatory train/calibration/final-holdout split. Aegis v4 separately implements
elastic 2/128, 4/192, and 8/256 Chessformer training with legal-only actions,
evidential WDL, private human/guide adapters, and per-action regret. The
canonical Apex v1 architecture (developed under the experimental Hydra Apex v5
label) adds an automatically scaled 29M-878M-parameter training-only Oracle, real
optimizer-inclusive per-GPU VRAM probing, NCCL/DDP support for 1-8 homogeneous
Verda GPUs, and student distillation. The branches are still not an exported
runtime package; see `docs/hydra-apex-v5-180core-a100.md` for the exact boundary.
Apex v1 specifically removes fixed `steps_per_epoch`: globally shuffled records are
consumed without replacement once per cardinality-sized epoch, with minimum
dataset gates and three-epoch early stopping. CUDA graphs are disabled and
reserved-memory growth is fail-closed after the v1 OOM postmortem.

## Why the A100 changes the plan

The A100 supports BF16 Tensor Cores with FP32-like exponent range, so the
transformer can use BF16 autocast without FP16 loss scaling. The default model
widths and batch sizes are multiples of eight, TF32 is enabled for residual FP32
matmuls, AdamW uses fused CUDA kernels, and `torch.compile` is enabled after
checkpoint restore. These are standard Ampere optimizations; actual throughput
must still be measured on the assigned host.

The XT-NNUE model is embedding-bound rather than Tensor-Core-bound. Its large
batch is intended to expose enough independent sparse lookups. Chessformer is
dense and should use the A100 substantially better.

## Install

Use a recent CUDA driver and the CUDA-enabled PyTorch wheel recommended at
<https://pytorch.org/get-started/locally/>. Then:

```bash
python3 -m venv .venv-a100
source .venv-a100/bin/activate
pip install --upgrade pip
pip install -r config/requirements-a100.txt
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
PY
```

## Data contracts

### XT-NNUE

The trainer accepts current fixed 104-byte NNUE records:

```text
12 x u64 mover-normalized bitboards
int16 teacher score from side-to-move
u8 WDL: 0 loss, 1 draw, 2 win
5 zero padding bytes
```

Threat relations are generated on GPU. Slider blockers use a precomputed 64 x
64 between-square table; no expanded threat sidecar is needed.

### Chessformer

The trainer accepts current fixed 104-byte policy records:

```text
12 x u64 mover-normalized bitboards
u16 from | (to << 6)
u16 mover rating
u8 normalized castling rights
u8 en-passant file or 0xff
u8 castle/en-passant/promotion flags
u8 padding
```

The existing record cannot identify the selected underpromotion piece and does
not include board history. Version 2 training therefore predicts from/to policy
only.

### Aegis v3 unified records

That format gap is closed by the frozen `UNCHD3R0` ABI: a 64-byte schema header
and fixed 160-byte records containing promotion identity, WDL, eight prior
moves, pseudonymous game/player hashes, time class, clocks, and optional
common-budget teacher scores. `tools/aegis_v3_data.py` validates shards without
NumPy/PyTorch and audits game/player-disjoint train/validation inputs. The
current Chessformer v1 trainer still consumes 104-byte records and must not be
pointed at v3 shards.

### Aegis v4 legal-set records

`UNCHD4R0` extends the 160-byte semantic prefix to a fixed 1,088-byte record by
adding all 218 promotion-aware legal-action slots and one optional common-budget
regret per action. `tools/train_chessformer_v4_a100.py` consumes only this
format. Human shards can be generated with `unchessed-datagen policy-v4`;
calibration of regret bounds additionally requires separately annotated guide
shards with real common-budget regrets. Never fill those fields with fabricated
or mixed-budget labels.

## Holdouts are mandatory

Do not split one flat file randomly. Mine training and validation shards
separately so:

- games do not overlap;
- policy players/accounts do not overlap;
- validation is from a later month;
- opening families and repeated positions are deduplicated;
- engine families are disjoint for engine-type evaluation.

The scripts require explicit `--train` and `--validation` arguments for this
reason.

## Self-checks

```bash
python tools/train_nnue_xt_a100.py selfcheck
python tools/train_chessformer_a100.py selfcheck --no-compile
python tools/train_nnue_xt_v3_a100.py selfcheck \
  --config config/a100_hydra_v3_training.json
python tools/train_chessformer_v4_a100.py selfcheck \
  --config config/a100_hydra_v4_training.json --no-compile
python tools/train_hydra_oracle_v5_a100.py selfcheck \
  --config config/a100_hydra_v5_training.json --no-compile
```

Run the self-check matching the intended experiment before allocating a long
job. The v3 command exercises all three relation indexers and XT heads. The v4
command exercises every elastic student exit. The v5 command exercises the
board oracle, masked legal-action decoder, WDL, quantiles, regret, concepts, and
private policy context. The first compiled training step can be
slow because Inductor autotunes kernels.

## Full launcher

```bash
export TRAIN_NNUE='/data/nnue/train-*.bin'
export VAL_NNUE='/data/nnue/future-holdout-*.bin'
export TRAIN_POLICY='/data/policy/train-*.bin'
export VAL_POLICY='/data/policy/player-disjoint-holdout-*.bin'
export OUTPUT_DIR=/data/checkpoints/unchessed-a100
scripts/training/a100_hybrid_train.sh
```

For Aegis v3 XT, provide an additional disjoint calibration set:

```bash
export TRAIN_NNUE='/data/nnue/train-*.bin'
export CAL_NNUE='/data/nnue/calibration-*.bin'
export VAL_NNUE='/data/nnue/final-holdout-*.bin'
export OUTPUT_DIR=/data/checkpoints/aegis-v3-xt
scripts/training/a100_hydra_v3_xt_train.sh
```

For Aegis v4 Chessformer:

```bash
export TRAIN_POLICY_V4='/data/v4/train-*.aegis4'
export CAL_POLICY_V4='/data/v4/calibration-*.aegis4'
export VAL_POLICY_V4='/data/v4/final-holdout-*.aegis4'
export OUTPUT_DIR=/data/checkpoints/aegis-v4-chessformer
scripts/training/a100_hydra_v4_chessformer_train.sh
```

For the full Apex v5 oracle-to-student sequence:

```bash
export TRAIN_V5='/nvme/v5/train/*.aegis4'
export TUNE_V5='/nvme/v5/tune/*.aegis4'
export FINAL_V5='/nvme/v5/final/*.aegis4'
export OUTPUT_DIR=/nvme/checkpoints/hydra-apex-v5
scripts/training/verda_apex_v1_train.sh
```

The v5 launcher detects the GPU family, selects the corresponding 29M-878M
Oracle, starts 1-8 `torchrun` ranks, and probes real
forward/backward/fused-optimizer memory on every GPU. Profile ceilings retain
4-15% VRAM for workspaces and fragmentation. All launchers validate records and
split leakage before CUDA. Shell expansion is intentional; avoid spaces in
shard paths. The jobs run
sequentially so each receives the entire GPU.

Resume after interruption:

```bash
python tools/train_nnue_xt_a100.py train \
  --train /data/nnue/train-*.bin \
  --validation /data/nnue/holdout-*.bin \
  --output /data/checkpoints/xt-nnue.pt \
  --resume /data/checkpoints/xt-nnue.pt
```

The Chessformer command has the same `--resume` behavior.

## Memory and throughput tuning

Start from `config/a100_hybrid_training.json`.

- XT-NNUE default batch: 16,384. Its temporary blocker matrix is large; reduce
  to 8,192 if compilation or validation peaks unexpectedly.
- Chessformer default batch: 2,048, matching published CF-6M A100 experiments.
- Increase only after checking `nvidia-smi dmon` and PyTorch peak allocation.
- Keep fixed shapes to avoid recompilation.
- If `torch.compile` regresses because sparse nonzero shapes vary, use
  `--no-compile` for XT-NNUE while retaining it for Chessformer.
- Do not use FP8 on A100; BF16 is the supported stable Tensor-Core path.
- Do not enable gradient checkpointing for these small models unless a much
  larger Chessformer is selected; recomputation is likely slower than useful.

## Accuracy strategy

### XT-NNUE

The model combines teacher score and WDL. It trains eight small material-phase
heads, and the threat table is clipped to a narrower quantization-aware range.
Before export, compare:

- float checkpoint;
- fake/real scale-255 transformer quantization;
- int8 threat weights if calibration permits;
- future-holdout WDL loss and centipawn MAE;
- score drift on the frozen twelve-position suite;
- NPS with full residual refresh versus dirty updates.

### Chessformer

Policy training uses rating conditioning, file-mirror augmentation, label
smoothing, and extra weight for castling, en passant, and promotions. Required
reports are overall and per-special-class top-1/NLL by Elo and time control.

For persona use, train two checkpoints from the same architecture:

- human checkpoint: player move as policy target;
- guide checkpoint: alpha-beta/MCTS teacher policy as target.

Do not blend their weights. The persona router chooses the checkpoint and
alpha-beta verification policy.

## Promotion gates

A model is not production-ready until all of the following exist:

1. frozen input/output format and inspector;
2. player/game/future-disjoint holdout metrics;
3. quantized versus float drift report;
4. Rust scalar reference inference;
5. SIMD equality and latency tests;
6. alpha-beta safety-veto suite;
7. fixed-Elo clock-tier stability;
8. paired game SPRT for each separately enabled backend.

A100 training creates candidates quickly; it does not replace these gates.
