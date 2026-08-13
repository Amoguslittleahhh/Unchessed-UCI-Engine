#!/bin/bash
# Deliberately NOT using `set -e` -- this runs unattended overnight, and a
# single bad month (e.g. a truncated download if wifi drops) must not abort
# the whole pipeline. Every risky step below checks its own exit code.
BIN=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-datagen
FRESH=/home/amogusontheterminal/unchessed-ai/data/maia-data/fresh
NNUE_DIR=/home/amogusontheterminal/unchessed-ai/data/maia-data/nnue
N_WORKERS=28
# ~14M new positions/month target (not 20M) -- combined with the existing
# 108M positions, this keeps total records under ~178M, which keeps
# train_nnue.py's peak RAM (data array + two shuffle-index-array copies,
# ~record_count*120 bytes) under ~20GB, safely inside WSL's 23GB cap. The
# earlier memory-doubling bug already taught us this project's history of
# reaching for round numbers without doing this arithmetic -- don't repeat it.
PER_WORKER_CAP=500000

MONTHS="2026-07 2026-06 2026-05 2026-04 2026-03"
SUCCEEDED=0

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# --- Step 1: wait for all downloads to finish ---
log "Waiting for downloads to finish..."
while pgrep -f "curl.*lichess_db_standard_rated_2026" > /dev/null; do
  sleep 30
done
log "All curl processes exited."
ls -la "$FRESH"

# --- Step 2: for each month, verify integrity -> decompress -> label -> clean up ---
for M in $MONTHS; do
  ZST="$FRESH/lichess_db_standard_rated_${M}.pgn.zst"
  PGN="$FRESH/lichess_db_standard_rated_${M}.pgn"
  OUTDIR="$FRESH/labeled_${M}"

  if [ ! -f "$ZST" ]; then
    log "SKIP $M: $ZST not found (download never started?)"
    continue
  fi

  # A curl process exiting is NOT proof the download completed -- wifi
  # dropping mid-transfer looks identical (the process just exits early).
  # -t verifies the compressed stream's checksums without writing output;
  # a truncated/corrupt file fails this cleanly instead of silently
  # producing a truncated .pgn (or crashing the whole script via set -e).
  log "Verifying integrity of $M ..."
  if ! pzstd -t "$ZST" > "$FRESH/verify_${M}.log" 2>&1; then
    log "SKIP $M: integrity check FAILED (likely an incomplete download -- wifi dropped mid-transfer). See $FRESH/verify_${M}.log. Leaving $ZST in place for a manual retry."
    continue
  fi
  log "$M integrity OK."

  log "Decompressing $M ..."
  if ! pzstd -f -d "$ZST" -o "$PGN" 2>> "$FRESH/verify_${M}.log"; then
    log "SKIP $M: decompression failed despite passing integrity check (unexpected). See $FRESH/verify_${M}.log."
    rm -f "$PGN"
    continue
  fi
  log "Decompressed $M: $(stat -c%s "$PGN" 2>/dev/null) bytes"

  log "Labeling $M with $N_WORKERS workers, cap $PER_WORKER_CAP/worker ..."
  mkdir -p "$OUTDIR"
  PIDS=()
  for ((i=0; i<N_WORKERS; i++)); do
    "$BIN" nnue "$OUTDIR/w$i.bin" "$i" "$N_WORKERS" "$PER_WORKER_CAP" "$PGN" > "$OUTDIR/w$i.log" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done

  MISSING=0
  for ((i=0; i<N_WORKERS; i++)); do
    [ -f "$OUTDIR/w$i.bin" ] || MISSING=$((MISSING+1))
  done
  if [ "$MISSING" -gt 0 ]; then
    log "WARNING $M: $MISSING/$N_WORKERS worker output files missing (a worker likely crashed) -- concatenating whatever succeeded rather than losing the whole month."
  fi

  cat "$OUTDIR"/w*.bin > "$NNUE_DIR/shard_${M}.bin" 2>/dev/null
  WROTE=$(stat -c%s "$NNUE_DIR/shard_${M}.bin" 2>/dev/null || echo 0)
  if [ "$WROTE" -lt 1000 ]; then
    log "SKIP $M: output shard is empty/near-empty ($WROTE bytes) -- something went wrong, not counting this month."
    rm -f "$NNUE_DIR/shard_${M}.bin"
  else
    log "Wrote $NNUE_DIR/shard_${M}.bin: $WROTE bytes"
    SUCCEEDED=$((SUCCEEDED+1))
  fi

  log "Cleaning up decompressed $PGN and worker outputs to free space ..."
  rm -f "$PGN"
  rm -rf "$OUTDIR"
  rm -f "$ZST"
done

log "Month processing done: $SUCCEEDED/5 months succeeded."
log "Shards now in $NNUE_DIR:"
ls -la "$NNUE_DIR"/*.bin

if [ "$SUCCEEDED" -eq 0 ]; then
  log "No new months succeeded -- nothing new to train on. Stopping without retraining (the existing unchessed-nnue.bin from the last run is untouched)."
  exit 1
fi

# --- Step 3: retrain NNUE on the combined (old + new) dataset ---
SHARDS=$(ls "$NNUE_DIR"/*.bin)
TOTAL_BYTES=$(cat $SHARDS | wc -c)
TOTAL_RECORDS=$((TOTAL_BYTES / 104))
# Safety ceiling: train_nnue.py's peak RAM is roughly record_count*120 bytes
# (the data array itself at 104 bytes/record, plus two int64 shuffle-index-
# array copies at 8 bytes/record each). 200M records * 120 bytes ~= 22.4GB,
# leaving headroom under WSL's 23GB cap. This is exactly the class of bug
# (unchecked combined dataset size -> near-OOM/swap-thrashing) already hit
# once tonight -- don't repeat it a second time with nobody around to catch it.
SAFE_MAX_RECORDS=200000000
log "Combined dataset: $TOTAL_RECORDS records ($TOTAL_BYTES bytes) across $(echo $SHARDS | wc -w) shard files."
if [ "$TOTAL_RECORDS" -gt "$SAFE_MAX_RECORDS" ]; then
  log "ABORT: $TOTAL_RECORDS records exceeds the $SAFE_MAX_RECORDS safety ceiling -- training on this would risk OOM/swap-thrashing with nobody around to intervene. Not starting training. The labeled shards are still on disk in $NNUE_DIR; re-run just the training step manually with a subset of shards, or after raising WSL's memory cap."
  exit 1
fi

log "Starting NNUE retrain on combined dataset ..."
cd /home/amogusontheterminal/unchessed-kingsafety-src
OUT=/home/amogusontheterminal/unchessed-ai/results/nnue_training/unchessed-nnue-v2.bin
LOG=/home/amogusontheterminal/unchessed-ai/results/nnue_training/train_v2.log

env OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 DEVICE=cpu BATCH_SIZE=65536 \
  ~/unchessed-ai/data/maia-venv/bin/python3 tools/train_nnue.py "$OUT" 15 $SHARDS \
  > "$LOG" 2>&1
TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -eq 0 ]; then
  log "Training finished successfully. Output: $OUT"
else
  log "Training FAILED (exit code $TRAIN_EXIT). Check $LOG for the error -- the old unchessed-nnue.bin is untouched, nothing was overwritten."
fi
tail -20 "$LOG"
