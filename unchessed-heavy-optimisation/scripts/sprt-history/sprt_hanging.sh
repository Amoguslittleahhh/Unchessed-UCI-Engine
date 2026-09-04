#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_hanging.pgn
LOG=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_hanging.log

# Same binary both sides (all other terms at their SPRT-validated defaults)
# -- only HangingPct differs, isolating the new hanging-piece term's own
# contribution, same pattern used for KnightOutpostPct/RookPct.
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Hanging option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.HangingPct=100 \
  -engine cmd="$ENGINE" name=Baseline option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.HangingPct=0 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
