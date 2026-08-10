#!/bin/bash
CUTECHESS=/home/amogusontheterminal/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=A option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=64 option.MobilityPct=100 \
  -engine cmd="$ENGINE" name=B option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=64 option.MobilityPct=100 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 3 -games 2 -concurrency 2 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800
