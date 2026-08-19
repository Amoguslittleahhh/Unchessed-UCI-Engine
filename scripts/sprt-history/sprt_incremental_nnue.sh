#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
AFTER=/home/amogusontheterminal/sprt-incremental-after/target/release/unchessed-adapter
BEFORE=/home/amogusontheterminal/sprt-incremental-before/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_incremental_nnue.pgn
LOG=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_incremental_nnue.log

# Same real v4 weights both sides (copied next to each binary already) --
# the only difference is the code itself: incremental accumulator updates
# (After, commit 00d0941) vs full recompute every eval (Before, commit
# fde7fce, 00d0941's direct parent). Adaptive=false isolates raw search+
# eval strength, not persona-layer behavior. At real game time controls,
# the ~1.6-1.9x depth-12/13 speedup measured directly should translate
# into a real Elo gain from reaching greater depth in the same clock --
# this SPRT run is what actually confirms that rather than assuming it.
"$CUTECHESS" \
  -engine cmd="$AFTER" name=After option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -engine cmd="$BEFORE" name=Before option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
