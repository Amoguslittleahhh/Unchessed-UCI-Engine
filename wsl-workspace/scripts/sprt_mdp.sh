#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
MDP_ENGINE=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-mdp/release/unchessed-adapter
BASE_ENGINE=/home/amogusontheterminal/unchessed-ai/builds/unchessed-magic-src/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_mdp_vs_magic.pgn
LOG=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_mdp_vs_magic.log

"$CUTECHESS" \
  -engine cmd="$MDP_ENGINE" name=MDP option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -engine cmd="$BASE_ENGINE" name=Magic option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
