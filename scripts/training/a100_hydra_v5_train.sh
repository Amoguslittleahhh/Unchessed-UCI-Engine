#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

: "${TRAIN_V5:?space-separated mixed human/guide UNCHD4R0 training shards required}"
: "${TUNE_V5:?space-separated player/game-disjoint tuning shards required}"
: "${FINAL_V5:?space-separated untouched final holdout shards required}"

OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/hydra-apex-v5}
BASE_CONFIG=${CONFIG:-config/a100_hydra_v5_training.json}
PROFILE_CONFIG=${PROFILE_CONFIG:-config/verda_gpu_profiles.json}
RESOLVED_CONFIG="$OUTPUT_DIR/resolved-gpu-training.json"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/torch-cache"

GPU_COUNT=${GPU_COUNT:-$(nvidia-smi -L | wc -l)}
if (( GPU_COUNT < 1 || GPU_COUNT > 8 )); then
  echo "GPU_COUNT must be in 1..8, detected $GPU_COUNT" >&2
  exit 1
fi
python3 tools/verda_gpu_profile.py resolve \
  --profiles "$PROFILE_CONFIG" --base-config "$BASE_CONFIG" \
  --output "$RESOLVED_CONFIG" >/dev/null
CONFIG="$RESOLVED_CONFIG"

export DEVICE=${DEVICE:-cuda:0}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/torch-cache}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

VERDA_DATA_PATH=${VERDA_DATA_PATH:-/data}
python3 tools/verda_v5_preflight.py --role gpu \
  --data-path "$VERDA_DATA_PATH" --expected-gpus "$GPU_COUNT" \
  --strict --require-torch --json "$OUTPUT_DIR/verda-gpu-preflight.json"

# Validate every record and enforce pairwise player/game disjointness before GPU use.
# shellcheck disable=SC2086
python3 tools/aegis_v4_data.py inspect $TRAIN_V5 $TUNE_V5 $FINAL_V5 \
  --json "$OUTPUT_DIR/data-inspection.json"
# shellcheck disable=SC2086
python3 tools/aegis_v4_data.py audit-split --train $TRAIN_V5 --validation $TUNE_V5
# shellcheck disable=SC2086
python3 tools/aegis_v4_data.py audit-split --train $TRAIN_V5 $TUNE_V5 --validation $FINAL_V5

nvidia-smi | tee "$OUTPUT_DIR/nvidia-smi-before.txt"
python3 - "$CONFIG" <<'PY' | tee "$OUTPUT_DIR/runtime.json"
import json, sys, torch
config = json.load(open(sys.argv[1], encoding="utf-8"))
facts = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "profile": config["hardware"],
    "gpus": [
        {
            "index": index,
            "device": torch.cuda.get_device_name(index),
            "total_vram": torch.cuda.get_device_properties(index).total_memory,
            "capability": torch.cuda.get_device_capability(index),
        }
        for index in range(torch.cuda.device_count())
    ],
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
}
print(json.dumps(facts, indent=2))
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == config["hardware"]["gpu_count"]
PY

python3 tools/train_hydra_oracle_v5_a100.py selfcheck \
  --config "$CONFIG" --no-compile

# The per-rank probe includes forward, backward, and fused AdamW state; the
# resolved GPU profile keeps a 4-15% workspace/fragmentation reserve.
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node="$GPU_COUNT" \
  tools/train_hydra_oracle_v5_a100.py train-oracle \
  --config "$CONFIG" \
  --train $TRAIN_V5 \
  --validation $TUNE_V5 \
  --auto-batch \
  --output "$OUTPUT_DIR/oracle-v5.pt" \
  2>&1 | tee "$OUTPUT_DIR/oracle-v5.log"

# shellcheck disable=SC2086
python3 tools/train_hydra_oracle_v5_a100.py evaluate-oracle \
  --config "$CONFIG" \
  --oracle "$OUTPUT_DIR/oracle-v5.pt.best" \
  --validation $FINAL_V5 \
  --metrics-json "$OUTPUT_DIR/oracle-final-holdout.json"

# Distill all oracle distributions into the compact, teacher-free runtime student.
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node="$GPU_COUNT" \
  tools/train_hydra_oracle_v5_a100.py distill-student \
  --config "$CONFIG" \
  --oracle "$OUTPUT_DIR/oracle-v5.pt.best" \
  --train $TRAIN_V5 \
  --validation $TUNE_V5 \
  --auto-batch \
  --output "$OUTPUT_DIR/student-v5.pt" \
  2>&1 | tee "$OUTPUT_DIR/student-v5.log"

# shellcheck disable=SC2086
python3 tools/train_hydra_oracle_v5_a100.py evaluate-student \
  --config "$CONFIG" \
  --student "$OUTPUT_DIR/student-v5.pt.best" \
  --validation $FINAL_V5 \
  --metrics-json "$OUTPUT_DIR/student-final-holdout.json"

nvidia-smi | tee "$OUTPUT_DIR/nvidia-smi-after.txt"
sha256sum "$OUTPUT_DIR"/*.pt* "$OUTPUT_DIR"/*holdout.json > "$OUTPUT_DIR/SHA256SUMS"
echo "Hydra Apex v5 $GPU_COUNT-GPU oracle training and student distillation complete: $OUTPUT_DIR"
