# Move-prediction pretrain pipeline (CPU stage / GPU stage)

The two-stage retrain from `docs/move-prediction-pretrain-plan.md`,
split by machine so each box does exactly what it is good at:

```
CPU.180V.720G box                      A100/H100 box
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ 1. generate.py (this repo)  │        │ 4. gpu_stage.sh              │
│    1M-5M mixed games        │ rsync  │    selfcheck (first!)        │
│    (pgn/ + labels/)         │ ─────► │    stage 1 pretrain (24 ep)  │
│ 2. cpu_stage.sh             │        │    stage 2 finetune (8 ep)   │
│    PGN -> v5 dual-elo shards│        │    per-epoch 0/200 sweep     │
│    (full + trusted-only)    │        │ -> ckpt-stage1/2.pt          │
│ 3. validate (structural)    │        │                              │
└─────────────────────────────┘        └──────────────────────────────┘
```

## What runs where (and why)

| Work | Box | Notes |
|---|---|---|
| Game generation (`tools/maia3_cloud_selfplay/generate.py`) | CPU | 170 workers, ~95-110 h for 5M mixed (measured inputs, see its README); the pilot command nails the real rate |
| v5 shard build + validation (`cpu_stage.sh`) | CPU | pure python-chess, minutes per 1M games; ~1088 B/record |
| Stage-1/2 training (`gpu_stage.sh`) | GPU | 58.5M-param dual-elo oracle; single GPU (DDP is the next iteration); `--micro-batch 128` default, probe with nvidia-smi |
| Selfcheck | GPU (first command) | tiny model on CUDA — catches torch/precision/model/loader problems before hours of GPU time. Also runs on CPU (sandbox-verified) |

**Handoff:** rsync both shard directories from the CPU box to the GPU
box (the GPU stage reads them via mmap; the CPU box does not need
torch and the GPU box does not need the generator).

## Commands

CPU box (after `generate.py` finished and passed its built-in
validation):

```sh
scripts/pretrain-pipeline/cpu_stage.sh /data/mixed-5m /data/pretrain 20000
rsync -a /data/pretrain/pretrain-v5 /data/pretrain/pretrain-v5-trusted <a100>:/data/pretrain/
```

GPU box:

```sh
scripts/pretrain-pipeline/gpu_stage.sh /data/pretrain
```

## Data format (v5, `UNCHD5R0`)

`tools/pretrain_v5_data.py` writes the frozen v4 wire record
(1088 B) with its 48 reserved bytes redefined as
`elo_oppo:u16 + pretrain_quality:u1 + pad:45`, a new magic/version/
schema-SHA. The mover's own elo stays in the existing `rating` field,
so each record carries the **dual-elo pair** the conditioning needs.
Quality: 0 calibrated (maia3), 1 native (stockfish), 2 approximate
(lc0/rubichess ladders — monotone but uncalibrated value), 3 human.
`wdl` is a game-outcome proxy (stage 1 trains policy CE only and does
not use it). Train/val split is by **game** (never by row). The
trusted-only stage-2 set contains games where **every** row is
calibrated/native/human — no approximate rows at all.

## What is verified, and what is not (honest status)

- **Verified in the sandbox (CPU):** the full CPU stage on the
  committed 13k-row self-play set (build + validate + spot-checks,
  0 errors); the GPU trainer's selfcheck; a real-data training smoke
  (small dual-elo oracle, 5 optimizer steps + conditioning sweep on
  v5 shards, loss finite, dual-elo logits differ by elo); 7 + new
  hermetic tests in the repo suite.
- **Not verified here (no CUDA in the sandbox):** the CUDA path of
  the trainer (precision/compile/memory). That is exactly what the
  selfcheck exists for — run it first on the box; expect it to take
  seconds. The multi-GPU DDP path is not implemented (single GPU).
- **Not yet wired (next round):** distillation to a dual-elo student
  + UNARCHV1 packaging + the Rust runtime change for dual-elo inputs.
  The checkpoint format marks `dual_elo: true` and the distill path
  refuses silently-wrong checkpoints. Nothing in this pipeline
  touches the engine binary; the hint stays default-off and the SPRT
  gate is unchanged.

## Sizes (1088 B/record)

| Set | Records | Raw |
|---|---|---|
| 5M mixed games (~65 plies/game) | ~327M | ~356 GB (mmap'd on the GPU box; consider `--shard-records` tuning on a tight volume) |
| 1M mixed games | ~65M | ~71 GB |
| 200-game self-play reference | 13,076 | ~14 MB (sandbox test set) |
