#!/bin/bash
# One-shot watcher: polls for month 1 (2026-07)'s shard to appear, then kills
# the running full_pipeline_cloud.sh, lowers PER_WORKER_CAP for the
# remaining months (they're already downloaded, no re-fetch needed), and
# relaunches for just those months. Exists to avoid wasting the ~2.6h+
# already sunk into month 1's labeling by killing it mid-way, while still
# cutting the remaining months down hard to hit a time/budget deadline.
SCRIPT=~/unchessed-kingsafety-src/scripts/nnue-pipeline/full_pipeline_cloud.sh
SHARD=~/unchessed-ai/data/maia-data/nnue/shard_2026-07.bin
LOG=~/pipeline_run_phase2.log

echo "[watcher] waiting for $SHARD to appear..."
while [ ! -f "$SHARD" ]; do
  sleep 15
done
echo "[watcher] month 1 done, killing current pipeline and relaunching for remaining months with a lower cap."

pkill -f full_pipeline_cloud.sh
pkill -f unchessed-datagen
sleep 2

sed -i 's/^PER_WORKER_CAP=.*/PER_WORKER_CAP=250000/' "$SCRIPT"
sed -i 's/^MONTHS=.*/MONTHS="2026-06 2026-05 2026-04 2026-03"/' "$SCRIPT"

export VERDA_INSTANCE_ID=e75751b4-9551-404c-a0a9-221af2d96f94
export NOTIFY_TOPIC=unchessed-nnue-1800cee527745b93
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
disown
echo "[watcher] relaunched, logging to $LOG"
