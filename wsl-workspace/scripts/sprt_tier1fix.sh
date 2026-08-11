#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
FIXED=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
BASE=/home/amogusontheterminal/unchessed-tier1fix-baseline-src/target/release/unchessed-adapter
LOG=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_tier1fix.log
PGN=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_tier1fix.pgn
nohup "$CUTECHESS" -engine cmd="$FIXED" name=Tier1Fix option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -engine cmd="$BASE" name=Baseline option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 -concurrency 13 \
  -sprt elo0=-3 elo1=1 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$PGN" > "$LOG" 2>&1 &
disown
