#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
RUBI=/home/amogusontheterminal/unchessed-ai/data/rubichess_bin/RubiChess-20240817/linux/RubiChess-20240817_x86-64-avx2
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=Adapter2000 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=2000 option.OwnBook=false option.Hash=64 \
  -engine cmd="$RUBI" name=Rubi option.Threads=1 option.Hash=64 option.LimitNps=2500 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 4 -games 2 -concurrency 4 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000
