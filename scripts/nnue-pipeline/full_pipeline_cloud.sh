#!/bin/bash
# Cloud variant of full_pipeline.sh, sized for the rented A100 80GB box
# (22 CPU / 120GB host RAM / 80GB VRAM) instead of the local 14-core/24GB-cap
# WSL box the original script was tuned for. Same core design as the
# original: download -> per-month integrity-check -> decompress -> label ->
# delete the decompressed PGN and worker outputs before moving to the next
# month, so uncompressed Lichess data (which is 5-10x the compressed size)
# is NEVER all resident on disk at once.
#
# Speed/cost changes vs. the original (billing is per-hour, so wall-clock
# time IS money here):
#   1. Downloads ALL months in parallel (not serially), and labeling for a
#      month starts as soon as THAT month's download+verify finishes --
#      months still downloading keep going in the background while already-
#      ready months are being labeled on the CPU. Network-bound and CPU-bound
#      work overlap instead of happening in two fully separate phases.
#   2. `trap ... EXIT` DELETES the instance via the Verda CLI the moment the
#      script ends, on ANY exit path (success, training failure, or an early
#      abort). This is NOT a plain OS shutdown -- Verda's own docs
#      (docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete)
#      say explicitly "Shutdown instances continue to charge your account."
#      Only `verda vm delete` (or `hibernate`) actually stops compute
#      billing; `shutdown -h now` from inside the OS does not, on this
#      provider. Requires VERDA_CLIENT_ID/VERDA_CLIENT_SECRET (Verda's
#      documented env vars for CLI auth) and VERDA_INSTANCE_ID exported
#      before running this script -- see the instructions printed below if
#      they're missing. Remove the trap line if you want the box to stay up
#      afterward (e.g. to run SPRT/inference tests before tearing it down).
#   CAVEAT, not fully resolved: Verda's docs also say deleting an instance
#   does NOT delete its attached storage by default ("By default, no
#   storage is selected for deletion. All storage not marked for deletion
#   will continue to charge your account.") -- the CLI's exact flag for
#   also deleting the boot volume wasn't confirmed from the docs page
#   fetched during this session. After a run, verify via the Verda
#   dashboard that no orphaned volume is still billing, and delete it
#   manually there if so.
#
# Adjust BIN/FRESH/NNUE_DIR/VENV/TRAIN_SRC below to match the actual paths
# on the provisioned box before running. Deliberately NOT using `set -e` --
# a single bad month (e.g. a truncated download) must not abort the whole
# pipeline. Every risky step checks its own exit code instead.
BIN=~/unchessed-kingsafety-src/target/release/unchessed-datagen
FRESH=~/unchessed-ai/data/maia-data/fresh
NNUE_DIR=~/unchessed-ai/data/maia-data/nnue
VENV=~/unchessed-ai/data/maia-venv
TRAIN_SRC=~/unchessed-kingsafety-src
LICHESS_BASE="https://database.lichess.org/standard"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Stops billing the instant this script finishes, no matter how it exits --
# see the CAVEAT above about attached storage possibly surviving the delete.
teardown() {
  local exit_code=$?
  log "Pipeline finished (exit $exit_code)."
  if [ -z "${VERDA_INSTANCE_ID:-}" ] || ! command -v verda > /dev/null 2>&1; then
    log "!!! COULD NOT AUTO-DELETE: VERDA_INSTANCE_ID not set or 'verda' CLI not installed."
    log "!!! Plain OS shutdown does NOT stop billing on Verda (confirmed via their docs)."
    log "!!! YOU MUST MANUALLY DELETE THIS INSTANCE from the Verda dashboard/CLI now, or it keeps charging."
    return
  fi
  log "Deleting instance $VERDA_INSTANCE_ID via Verda CLI to stop billing..."
  if verda vm delete "$VERDA_INSTANCE_ID" --yes; then
    log "Instance delete requested. Verify in the Verda dashboard that no orphaned storage volume is still billing (see CAVEAT above)."
  else
    log "!!! 'verda vm delete' FAILED. YOU MUST MANUALLY DELETE THIS INSTANCE from the Verda dashboard now, or it keeps charging."
  fi
}
trap teardown EXIT

mkdir -p "$FRESH" "$NNUE_DIR"

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

# --- Step 1: start every month's download in parallel immediately ---
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

# --- Step 2: for each month, wait for ITS OWN download, then verify ->
#     decompress -> label -> clean up, while the other months' downloads
#     keep running in the background. ---
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

  # A curl exiting 0 is a good sign but not airtight proof of a complete
  # file (e.g. a connection reset mid-transfer that curl still reports as
  # success in some edge cases) -- -t verifies the compressed stream's
  # checksums without writing output; a truncated/corrupt file fails this
  # cleanly instead of silently producing a truncated .pgn.
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
  log "ABORT: $TOTAL_RECORDS records exceeds the $SAFE_MAX_RECORDS safety ceiling. Not starting training. The labeled shards are still on disk in $NNUE_DIR."
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
  log "Training FAILED (exit code $TRAIN_EXIT). Check $LOG for the error."
fi
tail -20 "$LOG"

# No auto git-commit here by design -- left for manual review/commit
# instead. Free disk of the consumed INPUT data only (downloads,
# decompressed PGNs, labeled shards) -- deliberately leaves
# ~/unchessed-ai/results/nnue_training (the trained weights + log) alone,
# since that's the one thing worth keeping around to inspect/copy off
# before the instance auto-deletes below.
log "Cleaning up consumed input data before teardown (trained weights in $OUT are left in place)..."
rm -rf "$FRESH" "$NNUE_DIR"
log "Pipeline done. Instance will be deleted now (see teardown trap above) -- copy $OUT off the box first if you haven't already."
