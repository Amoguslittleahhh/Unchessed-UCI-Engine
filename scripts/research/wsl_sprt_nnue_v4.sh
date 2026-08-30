#!/bin/bash
# SPRT: current default NNUE (v3, unchessed-nnue.bin) vs. the newly trained
# v4 net (8 piece-count output buckets, round-12 quiet-filtered dataset).
set -u
cd ~/unchessed-sprt-build
BIN=$(pwd)/target/release/unchessed-adapter
BASE_NET=$(pwd)/unchessed-nnue.bin
NEW_NET=$(pwd)/nnue-round12-test/nnue_v4_round12.bin
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
OUT=~/unchessed-sprt-build/nnue-round12-test

"$CUTECHESS" \
  -engine name=Base-v3 cmd="$BIN" option.EvalFile="$BASE_NET" option.Threads=1 option.OwnBook=false \
  -engine name=New-v4 cmd="$BIN" option.EvalFile="$NEW_NET" option.Threads=1 option.OwnBook=false \
  -each proto=uci tc=10+0.1 \
  -openings file="$BOOK" format=pgn order=random \
  -games 2 -repeat -rounds 5000 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -concurrency 7 -recover \
  -pgnout "$OUT/sprt_games.pgn" \
  > "$OUT/sprt_log.txt" 2>&1
echo "SPRT process exited $?"
tail -30 "$OUT/sprt_log.txt"
