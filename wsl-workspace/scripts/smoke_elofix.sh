#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix/release/unchessed-adapter
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Weak option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=500 option.OwnBook=false option.Hash=64 \
  -engine cmd="$ENGINE" name=Full option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=false option.OwnBook=false option.Hash=64 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -repeat -rounds 3 -games 2 -concurrency 2 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=900
