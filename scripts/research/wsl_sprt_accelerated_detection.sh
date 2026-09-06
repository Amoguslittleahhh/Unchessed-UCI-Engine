#!/bin/bash
# Real SPRT for the AcceleratedDetection option (commit b5e5b32): same
# main binary both sides, one instance running the shipped default
# (AcceleratedDetection=false) against the same binary with the option
# turned on. Tests whether confirming a strong opponent sooner (see
# scripts/research/manus_opponent_detection_lag_finding.md and the IEEE
# paper at manus/research-facilities commit 11683a8) has a measurable
# real-game effect.
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
ENGINE=~/unchessed-sprt-build/target/release/unchessed-adapter
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_accelerated_detection.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_accelerated_detection.log

"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Baseline option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.AcceleratedDetection=false option.EvalFile="$NNUE" \
  -engine cmd="$ENGINE" name=Accelerated option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.AcceleratedDetection=true option.EvalFile="$NNUE" \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 500 -games 2 \
  -concurrency 7 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -recover \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
echo "SPRT process exited $?"
tail -20 "$LOG"
