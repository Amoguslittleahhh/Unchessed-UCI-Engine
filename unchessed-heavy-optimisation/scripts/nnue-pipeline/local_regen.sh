#!/bin/bash
# Local (non-cloud) regen of the NNUE quiet-filtered training set from the
# already-committed corpus (data/training, data/training-elo, data/selfplay).
# Round-12 finding: none of these carry TimeControl headers, so the
# fail-closed NNUE_MIN_BASE_SECS gate rejects everything at its default --
# UNCHESSED_NNUE_MIN_BASE_SECS=0 is the documented, deliberate opt-out for
# exactly this corpus (docs/nnue-round-12-results.md).
set -u
BIN="$(pwd)/target/release/unchessed-datagen.exe"
OUT_DIR="${1:?usage: local_regen.sh <out_dir> [n_workers]}"
N_WORKERS="${2:-16}"
PER_WORKER_CAP=5000000

mkdir -p "$OUT_DIR"

PGNS=$(find data/training data/training-elo data/selfplay -name "*.pgn" | sort)
N_FILES=$(echo "$PGNS" | wc -l)
echo "[$(date '+%H:%M:%S')] $N_FILES PGN files, $N_WORKERS workers, min_base_secs=0"

export UNCHESSED_NNUE_MIN_BASE_SECS=0
export UNCHESSED_QUIET_HISTOGRAM=1

PIDS=()
for ((i=0; i<N_WORKERS; i++)); do
  "$BIN" nnue "$OUT_DIR/w$i.bin" "$i" "$N_WORKERS" "$PER_WORKER_CAP" $PGNS \
    > "$OUT_DIR/w$i.log" 2>&1 &
  PIDS+=($!)
done

echo "[$(date '+%H:%M:%S')] launched ${#PIDS[@]} workers, waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL+1))
done
echo "[$(date '+%H:%M:%S')] done, $FAIL worker(s) failed"

TOTAL_BYTES=$(cat "$OUT_DIR"/w*.bin 2>/dev/null | wc -c)
echo "total bytes across shards: $TOTAL_BYTES"
tail -n 2 "$OUT_DIR"/w0.log
