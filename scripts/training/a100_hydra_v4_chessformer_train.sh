#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

: "${TRAIN_POLICY_V4:?space-separated UNCHD4R0 training shards required}"
: "${CAL_POLICY_V4:?space-separated player/game-disjoint calibration shards required}"
: "${VAL_POLICY_V4:?space-separated final holdout shards required}"

OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/aegis-v4-chessformer}
CONFIG=${CONFIG:-config/a100_hydra_v4_training.json}
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/torch-cache"

export DEVICE=${DEVICE:-cuda:0}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/torch-cache}

# Shell expansion is intentional; shard paths must not contain spaces.
# shellcheck disable=SC2086
python3 tools/aegis_v4_data.py inspect $TRAIN_POLICY_V4 $CAL_POLICY_V4 $VAL_POLICY_V4 \
  --json "$OUTPUT_DIR/data-inspection.json"
# Pairwise audits ensure all three partitions are disjoint.
# shellcheck disable=SC2086
python3 tools/aegis_v4_data.py audit-split --train $TRAIN_POLICY_V4 --validation $CAL_POLICY_V4
# shellcheck disable=SC2086
python3 tools/aegis_v4_data.py audit-split --train $TRAIN_POLICY_V4 $CAL_POLICY_V4 --validation $VAL_POLICY_V4

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

python3 tools/train_chessformer_v4_a100.py selfcheck --config "$CONFIG" --no-compile

# shellcheck disable=SC2086
python3 tools/train_chessformer_v4_a100.py train \
  --config "$CONFIG" \
  --train $TRAIN_POLICY_V4 \
  --calibration $CAL_POLICY_V4 \
  --validation $VAL_POLICY_V4 \
  --output "$OUTPUT_DIR/chessformer-v4.pt" \
  2>&1 | tee "$OUTPUT_DIR/chessformer-v4.log"

sha256sum "$OUTPUT_DIR"/*.pt* > "$OUTPUT_DIR/SHA256SUMS"
echo "Aegis v4 Chessformer training complete: $OUTPUT_DIR"
