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
| Game generation (`tools/maia3_cloud_selfplay/generate.py`) | CPU | auto workers (175 mixed / 350 maia-only on 360 vCPUs), ~95-110 h for 5M mixed on 180V or ~half that on 360V at the same cost (measured inputs, see its README); the pilot command nails the real rate |
| v5 shard build + validation (`cpu_stage.sh`) | CPU | two-pass parallel: a text-only scan fixes the game-disjoint split, then one worker per PGN file replays + streams final shards; **deterministic byte-for-byte regardless of `--workers`** (tested); ~1088 B/record |
| Stage-1/2 training (`gpu_stage.sh`) | GPU | 58.5M-param dual-elo oracle; **DDP via torchrun (1-8 GPUs, nccl; gloo smoke-tested on CPU)**; global effective batch stays 4096 at every world size, `--micro-batch 256` default, activation checkpointing off (re-enable past micro 512) |
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

GPU box (single GPU):

```sh
scripts/pretrain-pipeline/gpu_stage.sh /data/pretrain
```

Multi-GPU (8x A100 — the trainer is DDP-ready; nccl on CUDA, rank 0
validates + checkpoints, one allreduce per optimizer step; from the
repo checkout on the box, with the shards under /data/pretrain):

```sh
torchrun --nproc_per_node=8 tools/pretrain_v1_a100.py train \
    --stage pretrain \
    --train /data/pretrain/pretrain-v5/train/shard-*.v5 \
    --validation /data/pretrain/pretrain-v5/val/shard-*.v5 \
    --config config/pretrain_v1_training.json \
    --output /data/pretrain/ckpt-stage1.pt
```

(`gpu_stage.sh` runs the single-GPU form of both stages; for
multi-GPU, run the same two stage commands under torchrun as above.
CPU smoke of the DDP path:
`pytest tools/test_pretrain_v5.py::test_ddp_gloo_two_rank_smoke`.)

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

## Throughput round (2026-08-28) — what was optimized

- **CPU stage, parallel v5 build:** `pretrain_v5_data.py build` is a
  two-pass design — pass 1 a text-only scan (no chess parsing) that
  counts the kept games per file and fixes the game-disjoint split,
  pass 2 one worker per PGN file replaying with full legality and
  streaming the final `shard-<file>-<seq>.v5` files directly (no temp
  files, so disk = dataset size). A worker that disagrees with the
  pass-1 count hard-fails the build. Output bytes are **identical for
  any `--workers`** (worker-invariance test in the suite).
- **CPU stage, 360-vCPU generation:** `generate.py` auto-sizes its
  workers (`(vCPUs-10)/2` mixed, `vCPUs-10` maia-only) and pins each
  worker to dedicated cores with one-thread BLAS/OpenMP pools
  (`--cpu-affinity auto`). 360-vCPU: 175 mixed / 350 maia-only
  workers.
- **GPU stage, DDP:** the trainer runs under `torchrun` (nccl on
  CUDA, gloo on CPU). Global effective batch is 4096 at every world
  size (per-rank accumulation adapts), one allreduce per optimizer
  step (accumulation substeps under `no_sync`), rank 0 validates +
  checkpoints + broadcasts the epoch metrics. A 2-rank gloo CPU smoke
  is in the suite.
- **GPU stage, memory:** `micro_batch_initial` 128 -> 256 and
  `activation_checkpointing` off — the 58.5M model at micro 256
  measured ~13 GB on an A100 (2x the R21 micro-128 6.7 GB), far
  inside 80 GB; the optimizer schedule is unchanged because the
  global batch is constant. The per-epoch conditioning sweep is
  chunked (64 positions x 27 elos per forward,
  `sweep_chunk_positions`) — the unchunked 5400-row forward OOM'd a
  CPU smoke box and is needless GPU memory.

## What is verified, and what is not (honest status)

- **Verified in the sandbox (CPU):** the full CPU stage on the
  committed 13k-row self-play set (build + validate + spot-checks,
  0 errors), including the parallel build's worker-invariance
  (workers 1 vs 3 -> byte-identical shards); the GPU trainer's
  selfcheck; a real-data training smoke (small dual-elo oracle,
  5 optimizer steps + conditioning sweep on v5 shards, loss finite,
  dual-elo logits differ by elo); a **2-rank gloo DDP smoke**
  (2 epochs, disjoint strided data, rank-0 checkpoint, exactly one
  epoch line per epoch); 9 + 4 + 12 hermetic tests in the repo suite
  for this pipeline.
- **Not verified here (no CUDA in the sandbox):** the CUDA path of
  the trainer (precision/compile/memory) and the **nccl** multi-GPU
  path (the DDP code is the same for nccl/gloo; the smoke uses gloo
  on CPU). That is exactly what the selfcheck exists for — run it
  first on the box; expect it to take seconds. A bare python launch
  (no torchrun) stays a single-process run, so the old single-GPU
  command line is unchanged.
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
