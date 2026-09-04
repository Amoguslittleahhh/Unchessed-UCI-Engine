#!/bin/bash
# Usage: worker_bench.sh <n_workers> <total_positions_target> <outdir>
set -e
N=$1
TOTAL=$2
OUTDIR=$3
BIN=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-datagen
PGN=/home/amogusontheterminal/unchessed-ai/data/maia-data/l2019-06.pgn
PER_WORKER=$((TOTAL / N))

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.bin "$OUTDIR"/*.log

START=$(date +%s.%N)
PIDS=()
for ((i=0; i<N; i++)); do
  "$BIN" nnue "$OUTDIR/w$i.bin" "$i" "$N" "$PER_WORKER" "$PGN" > "$OUTDIR/w$i.log" 2>&1 &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
END=$(date +%s.%N)

TOTAL_BYTES=$(cat "$OUTDIR"/*.bin | wc -c)
TOTAL_RECORDS=$((TOTAL_BYTES / 104))
ELAPSED=$(echo "$END - $START" | bc)
RATE=$(echo "$TOTAL_RECORDS / $ELAPSED" | bc)

echo "workers=$N target_total=$TOTAL actual_records=$TOTAL_RECORDS elapsed=${ELAPSED}s rate=${RATE}/s"
