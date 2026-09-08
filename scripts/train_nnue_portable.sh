#!/usr/bin/env bash
# Portable resource launcher for the original Unchessed NNUE trainer.
# Usage: scripts/train_nnue_portable.sh OUT.bin EPOCHS SHARD [SHARD...]
# Override DEVICE, BATCH_SIZE, THREADS, and AMP explicitly when needed.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUT.bin EPOCHS SHARD [SHARD...]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$1
EPOCHS=$2
shift 2

THREADS=${THREADS:-$(nproc)}
# Use all logical CPU threads for host-side decoding and BLAS unless the caller
# deliberately caps THREADS. This is throughput-oriented, not a safe setting
# for thermally constrained laptops or shared machines.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$THREADS}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$THREADS}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-$THREADS}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# PyTorch exposes CUDA for NVIDIA CUDA and AMD ROCm builds. DEVICE can be set
# explicitly to cuda, cuda:0, mps, or cpu. Intel/oneAPI and other backends
# should use a PyTorch build exposing the desired device and pass DEVICE.
if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE=cuda
  elif [[ "${PYTORCH_ENABLE_MPS_FALLBACK:-0}" == "1" ]]; then
    DEVICE=mps
  else
    DEVICE=cpu
  fi
fi
export DEVICE

# CUDA/ROCm throughput knobs. TF32 is enabled by the trainer runtime where
# supported; these variables avoid hidden allocator fragmentation on long runs.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export HIP_FORCE_DEV_KERNARG=${HIP_FORCE_DEV_KERNARG:-1}

# The trainer already selects the validated recipe. These defaults saturate an
# A100 80 GB while remaining overrideable for smaller GPUs or CPU hosts.
export BATCH_SIZE=${BATCH_SIZE:-131072}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
export EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.1}

# Never silently claim a GPU is active: print the selected resource policy and
# let train_nnue.py's runtime preflight report actual PyTorch availability.
echo "device=$DEVICE threads=$THREADS batch_size=$BATCH_SIZE" >&2
echo "gpu power/clock controls are intentionally not changed; use the host scheduler/vendor tools within safe thermal and policy limits" >&2

cd "$ROOT"
exec python3 tools/train_nnue.py "$OUT" "$EPOCHS" "$@"
