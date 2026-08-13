#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
NNUE_FILE=/home/amogusontheterminal/unchessed-ai/results/nnue_training/unchessed-nnue-v1.bin
OUT=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_nnue_v1.pgn
LOG=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_nnue_v1.log

# Same binary both sides -- only EvalFile differs (NNUE v1 vs default HCE,
# same isolation pattern as every eval-term gate before it). Adaptive=false
# so the raw eval is what's being compared, not persona-layer behavior.
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=NNUEv1 option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE_FILE" \
  -engine cmd="$ENGINE" name=Baseline option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
