#!/bin/bash
# Labeling-only variant of full_pipeline_cloud.sh, for a CPU-heavy box (180
# cores / 720GB RAM) with NO GPU. Does download -> label -> shard, same
# per-month integrity-check/decompress/cleanup discipline as the combined
# script, but deliberately stops there -- no training step, no Python/torch
# setup at all, since this box's whole job is CPU-bound labeling.
#
# Why this exists as a separate script rather than reusing
# full_pipeline_cloud.sh: labeling and training never overlap in time (the
# GPU sits at 0% the entire labeling phase, confirmed on the first real
# run), so pairing them on one GPU-bundled box means paying GPU-idle time
# for hours of CPU-only work. Split: CPU-heavy box for labeling (this
# script), separate GPU box for training only (train_nnue.py directly, once
# all shards are collected). See memory note
# unchessed-ai-cloud-nnue-two-phase for the full reasoning.
#
# Deliberately NOT using `set -e` -- a single bad month must not abort the
# whole run. Every risky step checks its own exit code instead.
BIN=~/unchessed-kingsafety-src/target/release/unchessed-datagen
FRESH=~/unchessed-ai/data/maia-data/fresh
NNUE_DIR=~/unchessed-ai/data/maia-data/nnue
LICHESS_BASE="https://database.lichess.org/standard"

NOTIFY_TOPIC="${NOTIFY_TOPIC:-}"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

notify() {
  local msg="$1"
  if [ -z "$NOTIFY_TOPIC" ]; then
    log "notify (skipped, NOTIFY_TOPIC not set): $msg"
    return
  fi
  curl -fsS -d "$msg" "https://ntfy.sh/$NOTIFY_TOPIC" > /dev/null 2>&1 \
    || log "notify: curl to ntfy.sh failed -- message was: $msg"
}

mkdir -p "$FRESH" "$NNUE_DIR"

# 2x-core oversubscription (360 workers / 180 cores) -- same ratio that
# benchmarked fastest locally (28/14) and on the A100 box (44/22). CPU-bound
# labeling, not memory-bound, so this should transfer directly.
N_WORKERS=360

# Full original cap restored (vs. the 250k fallback used when this was
# still running on the 22-core A100 under time pressure) -- the 180-core
# box is fast enough that the full cap no longer costs meaningful extra
# time, so there's no reason to settle for the smaller dataset.
PER_WORKER_CAP=1200000

# Month 1 (2026-07) already labeled on the A100 box and saved locally --
# only the remaining 4 months run here, no re-download needed for that one.
MONTHS="2026-06 2026-05 2026-04 2026-03"
SUCCEEDED=0

declare -A DL_PID
log "Starting all $(echo $MONTHS | wc -w) downloads in parallel..."
for M in $MONTHS; do
  ZST="$FRESH/lichess_db_standard_rated_${M}.pgn.zst"
  if [ -f "$ZST" ]; then
    log "$M: $ZST already present, skipping fetch."
    continue
  fi
  curl -fsSL "$LICHESS_BASE/lichess_db_standard_rated_${M}.pgn.zst" -o "$ZST" \
    > "$FRESH/curl_${M}.log" 2>&1 &
  DL_PID[$M]=$!
  log "Started download for $M (pid ${DL_PID[$M]})"
done

for M in $MONTHS; do
  ZST="$FRESH/lichess_db_standard_rated_${M}.pgn.zst"
  PGN="$FRESH/lichess_db_standard_rated_${M}.pgn"
  OUTDIR="$FRESH/labeled_${M}"

  if [ -n "${DL_PID[$M]:-}" ]; then
    log "Waiting for $M's download (pid ${DL_PID[$M]}) -- other months keep downloading in the background..."
    if ! wait "${DL_PID[$M]}"; then
      log "SKIP $M: download failed. See $FRESH/curl_${M}.log."
      continue
    fi
  fi

  if [ ! -f "$ZST" ]; then
    log "SKIP $M: $ZST not found after download attempt."
    continue
  fi

  log "Verifying integrity of $M ..."
  if ! pzstd -t "$ZST" > "$FRESH/verify_${M}.log" 2>&1; then
    log "SKIP $M: integrity check FAILED. See $FRESH/verify_${M}.log. Leaving $ZST in place for a manual retry."
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

log "Month processing done: $SUCCEEDED/$(echo $MONTHS | wc -w) months succeeded."
log "Shards now in $NNUE_DIR:"
ls -la "$NNUE_DIR"/*.bin

TOTAL_BYTES=$(cat "$NNUE_DIR"/*.bin 2>/dev/null | wc -c)
TOTAL_RECORDS=$((TOTAL_BYTES / 104))
log "Total records across all shards on this box: $TOTAL_RECORDS"

DELETE_CMD="verda vm delete <this-instance-id> --yes"
if [ "$SUCCEEDED" -gt 0 ]; then
  notify "Labeling done: $SUCCEEDED/$(echo $MONTHS | wc -w) months. Copy shards from $NNUE_DIR, then delete this box: $DELETE_CMD"
else
  notify "Labeling finished with 0/$(echo $MONTHS | wc -w) months succeeding -- check the log, something went wrong."
fi
log "No training happens on this box by design. Copy the shards in $NNUE_DIR to wherever training will run, then delete this instance yourself -- it is NOT auto-deleted."
