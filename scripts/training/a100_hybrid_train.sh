#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

: "${TRAIN_NNUE:?space-separated NNUE training shards required}"
: "${VAL_NNUE:?space-separated NNUE validation shards required}"
: "${TRAIN_POLICY:?space-separated policy training shards required}"
: "${VAL_POLICY:?space-separated policy validation shards required}"

OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/a100-hybrid}
CONFIG=${CONFIG:-config/a100_hybrid_training.json}
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/torch-cache"

export DEVICE=${DEVICE:-cuda:0}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/torch-cache}

nvidia-smi | tee "$OUTPUT_DIR/nvidia-smi.txt"
python3 - <<'PY'
import json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "capability": torch.cuda.get_device_capability(0),
}, indent=2))
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.is_bf16_supported(), "A100 BF16 support was not detected"
PY

python3 tools/train_nnue_xt_a100.py selfcheck --config "$CONFIG"
python3 tools/train_chessformer_a100.py selfcheck --config "$CONFIG" --no-compile

# shellcheck disable=SC2086
python3 tools/train_nnue_xt_a100.py train \
  --config "$CONFIG" \
  --train $TRAIN_NNUE \
  --validation $VAL_NNUE \
  --output "$OUTPUT_DIR/xt-nnue.pt" \
  2>&1 | tee "$OUTPUT_DIR/xt-nnue.log"

# shellcheck disable=SC2086
python3 tools/train_chessformer_a100.py train \
  --config "$CONFIG" \
  --train $TRAIN_POLICY \
  --validation $VAL_POLICY \
  --output "$OUTPUT_DIR/chessformer.pt" \
  2>&1 | tee "$OUTPUT_DIR/chessformer.log"

sha256sum "$OUTPUT_DIR"/*.pt* > "$OUTPUT_DIR/SHA256SUMS"
echo "A100 training complete: $OUTPUT_DIR"
