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
#   2. Deletion is MANUAL, by design. The script sends a push notification
#      (via ntfy.sh, see NOTIFY_TOPIC below) when the run finishes or fails,
#      with the exact `verda vm delete ...` command to run yourself -- it
#      does NOT auto-delete the instance, so you keep full control over
#      when the box actually goes away (e.g. to copy weights off first, or
#      run SPRT/inference tests before tearing it down). This means an idle
#      instance keeps billing until you act on the notification -- that's
#      the deliberate tradeoff for not risking data loss while you're away.
#      Reminder either way: per Verda's own docs
#      (docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete),
#      "Shutdown instances continue to charge your account" -- only
#      `verda vm delete` (or `hibernate`) actually stops compute billing;
#      `shutdown -h now` from inside the OS does not, on this provider.
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

# Push notification so you know the run is done even if you're away from
# the terminal -- ntfy.sh needs no account/API key, just a private topic
# name only you know (pick something long/random, e.g. a UUID) and
# subscribe to it beforehand via the ntfy app or https://ntfy.sh/<topic> in
# a browser. Set NOTIFY_TOPIC before running this script; if unset,
# notifications are silently skipped (logged, not fatal).
NOTIFY_TOPIC="${NOTIFY_TOPIC:-}"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

notify() {
  local msg="$1"
  if [ -z "$NOTIFY_TOPIC" ]; then
    log "notify (skipped, NOTIFY_TOPIC not set): $msg"
    return
  fi
  curl -fsS -d "$msg" "https://ntfy.sh/$NOTIFY_TOPIC" > /dev/null 2>&1 \
    || log "notify: curl to ntfy.sh failed (no internet, or ntfy.sh down?) -- message was: $msg"
}

# No auto-delete. Instance deletion is manual by design -- this script only
# notifies you and prints the exact command to run yourself when ready.
# Remember: per Verda's own docs, a plain OS `shutdown` does NOT stop
# billing on this provider -- only `verda vm delete` (or `hibernate`) does,
# and even that leaves attached storage billing separately unless you also
# choose to delete it (see CAVEAT above).

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

DELETE_CMD="verda vm delete ${VERDA_INSTANCE_ID:-<instance-id>} --yes"
if [ "$TRAIN_EXIT" -eq 0 ]; then
  log "Training finished successfully. Output: $OUT"
  notify "NNUE training finished OK. Instance is still running -- copy $OUT off, then delete it yourself: $DELETE_CMD"
else
  log "Training FAILED (exit code $TRAIN_EXIT). Check $LOG for the error."
  notify "NNUE training FAILED (exit $TRAIN_EXIT). Instance is still running -- inspect $LOG, then delete it yourself: $DELETE_CMD"
fi
tail -20 "$LOG"

# --- Step 4: pre-SPRT smoke test ---
# 100 games @ 1+0.1s against the currently-deployed network, reject if score
# < 40%. This is the cheap reject-only filter added after v3's -70.3 Elo
# regression took ~756 full-SPRT games to catch -- see
# scripts/research/remaining_research_topics.md item #92 for the detection-
# floor validation (it correctly failed v3 at 28.5% in the same 100 games).
# A PASS here does not mean "good," only "not obviously terrible" -- still
# run a full SPRT (scripts/sprt-history/sprt_nnue_v4.sh is the pattern) to
# confirm any real Elo gain before promoting. Requires cutechess-cli to be
# present on this box; if this pipeline is run on a fresh rental without it
# set up yet, the smoke test is skipped rather than failing the whole run.
if [ "$TRAIN_EXIT" -eq 0 ]; then
  log "Running pre-SPRT smoke test on $OUT ..."
  SMOKE_SCRIPT="$TRAIN_SRC/scripts/nnue-pipeline/smoke_test_nnue.sh"
  if [ -x "$SMOKE_SCRIPT" ]; then
    "$SMOKE_SCRIPT" "$OUT" "" "v3_cloud"
    SMOKE_EXIT=$?
    if [ "$SMOKE_EXIT" -eq 0 ]; then
      log "Smoke test PASSED -- $OUT is not obviously bad."
      notify "Smoke test PASSED for $OUT. Not obviously bad -- run a full SPRT before promoting."
    else
      log "Smoke test FAILED -- $OUT looks like a regression. Do NOT run a full SPRT on this without investigating first."
      notify "Smoke test FAILED for $OUT -- looks like a regression, investigate before running SPRT."
    fi
  else
    log "Smoke test script not found or not executable at $SMOKE_SCRIPT (cutechess-cli/book may not be set up on this box yet) -- skipping. Run it manually before any SPRT gate."
  fi
else
  log "Skipping smoke test since training failed."
fi

# No auto git-commit, no auto cleanup, no auto-delete -- all left for
# manual handling. The instance keeps running (and billing) until you
# delete it yourself.
log "Data and weights left on disk ($FRESH, $NNUE_DIR, $OUT) for manual retrieval/cleanup."
log "Instance is NOT auto-deleted. When you're done, delete it yourself: $DELETE_CMD"
log "Reminder: a plain OS 'shutdown' does NOT stop billing on Verda -- only 'verda vm delete' (or hibernate) does, and attached storage may keep billing separately unless also deleted (see CAVEAT above)."
