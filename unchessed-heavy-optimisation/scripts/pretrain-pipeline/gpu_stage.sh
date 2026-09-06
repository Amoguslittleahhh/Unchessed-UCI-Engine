#!/usr/bin/env bash
# ============================================================================
# PRETRAIN PIPELINE — GPU STAGE (runs on the A100/H100 box)
#
# Trains the dual-elo Unarchitectured Metal oracle on the v5 shards built
# by cpu_stage.sh:
#   stage 1 (pretrain): legal-only policy cross-entropy, all rows,
#                       approximate rows down-weighted (0.5x)
#   stage 2 (finetune): same objective, trusted-only shards, lower LR,
#                       resumed from the stage-1 best checkpoint
#
# Every epoch the validator runs the 0/200 conditioning sweep on
# held-out positions (mover elo 600->3200): watch
# sweep_flips=N/200 in the log — the canonical v1 was 0/200; a working
# pretrain shows substantial flips with high-elo play more concentrated
# (top1prob@3200 > top1prob@600).
#
# Multi-GPU: this script runs the single-GPU form. For 8x A100, run
# the same two stage commands under
#   torchrun --nproc_per_node=8 tools/pretrain_v1_a100.py train ...
# (nccl; the global effective batch stays 4096 at every world size —
# see scripts/pretrain-pipeline/README.md).
#
# Usage:
#   scripts/pretrain-pipeline/gpu_stage.sh <out_root> [micro_batch] [gpus]
#
#   <out_root>    directory holding pretrain-v5/ + pretrain-v5-trusted/
#                 (rsynced from the CPU box)
#   [micro_batch] per-GPU micro-batch (default 256; probe with
#                 nvidia-smi if OOM)
#   [gpus]        number of GPUs to use for torch (default: all)
#
# Output:
#   <out_root>/ckpt-stage1.pt   (+ .best)
#   <out_root>/ckpt-stage2.pt   (+ .best)
#   heartbeats: <out_root>/*.heartbeat.json
#
# FIRST COMMAND ON A FRESH BOX IS SELFCHK — it exercises the CUDA path
# (torch build, precision, model, loader) with a tiny model before
# committing hours of GPU time:
# ============================================================================
set -euo pipefail

OUT_ROOT="${1:?usage: gpu_stage.sh <out_root> [micro_batch] [gpus]}"
MICRO_BATCH="${2:-256}"
GPUS="${3:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/venv/bin/python"
CFG="$REPO_ROOT/config/pretrain_v1_training.json"

[ -d "$OUT_ROOT/pretrain-v5" ] || { echo "missing $OUT_ROOT/pretrain-v5 (rsync from the CPU box)" >&2; exit 1; }
[ -d "$OUT_ROOT/pretrain-v5-trusted" ] || { echo "missing $OUT_ROOT/pretrain-v5-trusted" >&2; exit 1; }

if [ ! -x "$PY" ]; then
  echo "[gpu] creating venv with torch..."
  (cd "$REPO_ROOT" && python3 -m venv venv
   venv/bin/pip install -q -r tools/requirements-dev.txt
   venv/bin/pip install -q torch)
fi

export OMP_NUM_THREADS=8

echo "[gpu] selfcheck (tiny model, CUDA if available)..."
"$PY" "$REPO_ROOT/tools/pretrain_v1_a100.py" selfcheck --config "$CFG"

STAGE1_SHARDS=( "$OUT_ROOT"/pretrain-v5/train/shard-*.v5 )
STAGE1_VAL=( "$OUT_ROOT"/pretrain-v5/val/shard-*.v5 )
STAGE2_SHARDS=( "$OUT_ROOT"/pretrain-v5-trusted/train/shard-*.v5 )
STAGE2_VAL=( "$OUT_ROOT"/pretrain-v5-trusted/val/shard-*.v5 )

if [ -n "$GPUS" ]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))
fi

echo "[gpu] stage 1: pretrain (all rows, ${#STAGE1_SHARDS[@]} train shards)"
"$PY" "$REPO_ROOT/tools/pretrain_v1_a100.py" train --stage pretrain \
    --train "${STAGE1_SHARDS[@]}" \
    --validation "${STAGE1_VAL[@]}" \
    --config "$CFG" \
    --micro-batch "$MICRO_BATCH" \
    --output "$OUT_ROOT/ckpt-stage1.pt"

echo "[gpu] stage 2: finetune (trusted-only, ${#STAGE2_SHARDS[@]} train shards)"
"$PY" "$REPO_ROOT/tools/pretrain_v1_a100.py" train --stage finetune \
    --train "${STAGE2_SHARDS[@]}" \
    --validation "${STAGE2_VAL[@]}" \
    --config "$CFG" \
    --micro-batch "$MICRO_BATCH" \
    --lr 5e-5 --epochs 8 \
    --resume "$OUT_ROOT/ckpt-stage1.pt.best" \
    --output "$OUT_ROOT/ckpt-stage2.pt"

echo "[gpu] done. Checkpoints:"
ls -la "$OUT_ROOT"/ckpt-stage*.pt*
echo "[gpu] next (not yet wired): dual-elo student distillation + UNARCHV1 packaging + conditioning diagnostics + SPRT."
