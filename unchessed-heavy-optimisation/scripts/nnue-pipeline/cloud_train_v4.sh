#!/bin/bash
# Fail-closed A100 launcher for NNUE v4. Does not start billing-relevant
# training unless GO_CLOUD=I_ACCEPT_SPRT_GATES. Adaptive/persona stays ON.
# UnarchitecturedHint stays OFF. See docs/ieee-cloud-nnue-speed-quality.pdf.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

: "${GO_CLOUD:?set GO_CLOUD=I_ACCEPT_SPRT_GATES to start a billed run}"
export GO_CLOUD
export REQUIRE_CLOUD_GO=1
export PERSONA_ACTIVE="${PERSONA_ACTIVE:-1}"
export UNARCH_HINT="${UNARCH_HINT:-0}"
export DEVICE="${DEVICE:-cuda}"
export BATCH_SIZE="${BATCH_SIZE:-131072}"
export ALLOW_TF32="${ALLOW_TF32:-1}"
export USE_AMP="${USE_AMP:-1}"
export FUSED_ADAM="${FUSED_ADAM:-1}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.1}"
export EPOCH_CAP="${EPOCH_CAP:-15}"

OUT="${1:?usage: cloud_train_v4.sh <out.bin> <shard1.bin> [...]}"
shift
: "${1:?need at least one shard}"

PYTHON="${PYTHON:-python3}"
export METRICS_JSONL="${METRICS_JSONL:-${OUT}.metrics.jsonl}"

echo "cloud_train_v4: device=$DEVICE batch=$BATCH_SIZE amp=$USE_AMP tf32=$ALLOW_TF32 fused_adam=$FUSED_ADAM persona=$PERSONA_ACTIVE hint=$UNARCH_HINT cap=$EPOCH_CAP"
"$PYTHON" tools/train_nnue.py "$OUT" "$EPOCH_CAP" "$@"
echo "done. Adaptive remains default-on. Run SPRT with Adaptive=true before promoting $OUT."
