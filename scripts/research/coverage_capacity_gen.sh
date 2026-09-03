#!/bin/bash
# Minimal coverage-vs-capacity diagnostic (docs/reinforcement/13-nnue-ceiling.md),
# scoped down to the highest-priority axis: coverage (raw vs piece-count-
# balanced sampling) at fixed capacity, since Manus's own doc says not to
# spend on a capacity sweep until coverage is understood first.
#
# Train pool and held-out pool are file-disjoint (different PGN files),
# so held-out games are never seen during train-pool generation.
set -u
BIN="$(pwd)/target/release/unchessed-datagen.exe"
OUT_DIR="${1:?usage: coverage_capacity_gen.sh <out_dir>}"
mkdir -p "$OUT_DIR/train_pool" "$OUT_DIR/heldout"

TRAIN_FILES=$(find data/training data/training-elo data/selfplay -name "*.pgn" \
  ! -path "*lichess-2022-10-05/elo-2000-2300.pgn" \
  ! -path "*players/Nakamura.pgn" | sort)
HELDOUT_FILES="data/training/lichess-2022-10-05/elo-2000-2300.pgn data/training/players/Nakamura.pgn"

export UNCHESSED_NNUE_MIN_BASE_SECS=0
N_WORKERS=16
PER_WORKER_CAP=15000

echo "[$(date '+%H:%M:%S')] generating train pool (16 workers, cap $PER_WORKER_CAP/worker)..."
PIDS=()
for ((i=0; i<N_WORKERS; i++)); do
  "$BIN" nnue "$OUT_DIR/train_pool/w$i.bin" "$i" "$N_WORKERS" "$PER_WORKER_CAP" $TRAIN_FILES \
    > "$OUT_DIR/train_pool/w$i.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[$(date '+%H:%M:%S')] train pool done: $(du -sh "$OUT_DIR/train_pool" | cut -f1)"

echo "[$(date '+%H:%M:%S')] generating held-out set (game-disjoint files: $HELDOUT_FILES)..."
"$BIN" nnue "$OUT_DIR/heldout/heldout.bin" 0 1 30000 $HELDOUT_FILES \
  > "$OUT_DIR/heldout/gen.log" 2>&1
echo "[$(date '+%H:%M:%S')] held-out done: $(du -sh "$OUT_DIR/heldout/heldout.bin" | cut -f1)"
