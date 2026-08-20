#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

: "${TRAIN_NNUE:?space-separated NNUE training shards required}"
: "${CAL_NNUE:?space-separated disjoint NNUE calibration shards required}"
: "${VAL_NNUE:?space-separated final NNUE validation shards required}"

OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/aegis-v3-xt}
CONFIG=${CONFIG:-config/a100_hydra_v3_training.json}
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/torch-cache"

export DEVICE=${DEVICE:-cuda:0}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/torch-cache}

nvidia-smi | tee "$OUTPUT_DIR/nvidia-smi.txt"
python3 - <<'PY' | tee "$OUTPUT_DIR/runtime.json"
import json, torch
facts = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
}
print(json.dumps(facts, indent=2))
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.is_bf16_supported(), "A100 BF16 support was not detected"
PY

python3 tools/train_nnue_xt_v3_a100.py selfcheck --config "$CONFIG"

# Shell expansion is intentional: each variable may name multiple shards.
# shellcheck disable=SC2086
python3 tools/train_nnue_xt_v3_a100.py train \
  --config "$CONFIG" \
  --train $TRAIN_NNUE \
  --calibration $CAL_NNUE \
  --validation $VAL_NNUE \
  --output "$OUTPUT_DIR/xt-nnue-v3.pt" \
  2>&1 | tee "$OUTPUT_DIR/xt-nnue-v3.log"

sha256sum "$OUTPUT_DIR"/*.pt* > "$OUTPUT_DIR/SHA256SUMS"
echo "Aegis v3 XT training complete: $OUTPUT_DIR"
