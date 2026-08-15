#!/bin/bash
# Cloud variant of full_pipeline.sh, sized for the rented A100 80GB box
# (22 CPU / 120GB host RAM / 80GB VRAM) instead of the local 14-core/24GB-cap
# WSL box the original script was tuned for. Same core design as the
# original: download -> per-month integrity-check -> decompress -> label ->
# delete the decompressed PGN and worker outputs before moving to the next
# month, so uncompressed Lichess data (which is 5-10x the compressed size)
# is NEVER all resident on disk at once. This is what makes "get data from
# online and train on it directly" safe without needing to permanently store
# the full multi-hundred-GB uncompressed corpus -- the discard-per-month
# pattern already does that; nothing here needs to change for that part.
#
# Adjust BIN/FRESH/NNUE_DIR/VENV below to match the actual paths on the
# provisioned box before running. Deliberately NOT using `set -e` -- this
# runs unattended, and a single bad month (e.g. a truncated download) must
# not abort the whole pipeline. Every risky step checks its own exit code.
BIN=~/unchessed-kingsafety-src/target/release/unchessed-datagen
FRESH=~/unchessed-ai/data/maia-data/fresh
NNUE_DIR=~/unchessed-ai/data/maia-data/nnue
VENV=~/unchessed-ai/data/maia-venv
TRAIN_SRC=~/unchessed-kingsafety-src

# 2x-core oversubscription (44 workers / 22 cores) matched the fastest
# config found in the local worker_bench.sh sweep (28 workers / 14 cores,
# same 2x ratio) -- CPU-bound labeling, not memory-bound, so this should
# transfer directly to the bigger box.
N_WORKERS=44

# With 120GB host RAM (vs. the original 24GB WSL cap this pipeline's limits
# were tuned around), the RAM-safety reasoning that produced
# PER_WORKER_CAP=500000 / SAFE_MAX_RECORDS=200,000,000 no longer binds the
# same way. train_nnue.py's peak host RAM during shard loading is roughly
# record_count*112 bytes (104-byte record + 8-byte shuffle-index copy;
# GPU-resident mode drops the second shuffle-index copy off the host side
# once the transfer to DEVICE completes, but size for the pre-transfer peak
# to be safe). 500M records * 112 bytes ~= 56GB, comfortably under 120GB
# with room for the OS/Python/venv overhead. Raised ~2.5x vs. the original
# 200M ceiling; not raised further than that in one step since this hasn't
# been run at this scale before -- re-tune upward once a real run confirms
# actual peak RSS at this size.
PER_WORKER_CAP=1200000
SAFE_MAX_RECORDS=500000000

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

  # A curl process exiting is NOT proof the download completed -- a dropped
  # connection looks identical (the process just exits early). -t verifies
  # the compressed stream's checksums without writing output; a truncated/
  # corrupt file fails this cleanly instead of silently producing a
  # truncated .pgn (or crashing the whole script via set -e).
  log "Verifying integrity of $M ..."
  if ! pzstd -t "$ZST" > "$FRESH/verify_${M}.log" 2>&1; then
    log "SKIP $M: integrity check FAILED (likely an incomplete download). See $FRESH/verify_${M}.log. Leaving $ZST in place for a manual retry."
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
  log "No new months succeeded -- nothing new to train on. Stopping without retraining."
  exit 1
fi

# --- Step 3: retrain NNUE v3 (HalfKA, GPU-resident) on the combined dataset ---
SHARDS=$(ls "$NNUE_DIR"/*.bin)
TOTAL_BYTES=$(cat $SHARDS | wc -c)
TOTAL_RECORDS=$((TOTAL_BYTES / 104))
log "Combined dataset: $TOTAL_RECORDS records ($TOTAL_BYTES bytes) across $(echo $SHARDS | wc -w) shard files."
if [ "$TOTAL_RECORDS" -gt "$SAFE_MAX_RECORDS" ]; then
  log "ABORT: $TOTAL_RECORDS records exceeds the $SAFE_MAX_RECORDS safety ceiling -- training on this would risk OOM/swap-thrashing with nobody around to intervene. Not starting training. The labeled shards are still on disk in $NNUE_DIR; re-run just the training step manually with a subset of shards, or after re-tuning this ceiling against the box's actual RAM."
  exit 1
fi

log "Starting NNUE v3 retrain (HalfKA features, GPU-resident) on combined dataset ..."
cd "$TRAIN_SRC"
OUT=~/unchessed-ai/results/nnue_training/unchessed-nnue-v3.bin
LOG=~/unchessed-ai/results/nnue_training/train_v3.log

# DEVICE=cuda triggers train_nnue.py's GPU-resident path (whole dataset
# shipped to VRAM once). BATCH_SIZE raised well above the CPU-tuned 65536 --
# 80GB VRAM has ample room and larger batches improve GPU utilization for
# this small a model; re-tune down if it OOMs on VRAM (unlikely at this
# model size, but the box hasn't been measured yet).
env DEVICE=cuda BATCH_SIZE=131072 \
  "$VENV/bin/python3" tools/train_nnue.py "$OUT" 15 $SHARDS \
  > "$LOG" 2>&1
TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -eq 0 ]; then
  log "Training finished successfully. Output: $OUT"
else
  log "Training FAILED (exit code $TRAIN_EXIT). Check $LOG for the error -- nothing was overwritten."
fi
tail -20 "$LOG"
