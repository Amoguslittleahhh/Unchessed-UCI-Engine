#!/bin/bash
# Defended NNUE v4 training recipe (docs/nnue-v4-training-recipe.md).
# Does NOT launch cloud spend. Encodes the local production-matched run:
# 15-epoch cap, early-stop patience 3, best-checkpoint export, CPU batch
# 65536. Pass the shards that actually exist on this machine (12 original
# = 108M on the reviewer box; whatever is on disk otherwise).
#
# Usage: train_recipe.sh <out.bin> <shard1.bin> [shard2.bin ...]
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:?usage: train_recipe.sh <out.bin> <shard1.bin> [...]}"
shift
: "${1:?need at least one shard}"

export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.1}"
export BATCH_SIZE="${BATCH_SIZE:-65536}"
export DEVICE="${DEVICE:-cpu}"

PYTHON="${PYTHON:-python3}"
echo "recipe: out=$OUT cap=15 patience=$EARLY_STOP_PATIENCE min_delta=$EARLY_STOP_MIN_DELTA batch=$BATCH_SIZE device=$DEVICE shards=$# python=$PYTHON"
"$PYTHON" tools/train_nnue.py "$OUT" 15 "$@"
