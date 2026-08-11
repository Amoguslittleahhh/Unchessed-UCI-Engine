#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
SF=/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_scaling
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=Adapter500 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=500 option.OwnBook=false option.Hash=64 \
  -engine cmd="$SF" name=SF1800 option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=1800 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 15 -games 2 -concurrency 1 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/elo_500_retry.pgn" \
  > "$OUTDIR/log_500_retry.txt" 2>&1
echo "500 retry done"
grep 'Score of' "$OUTDIR/log_500_retry.txt" | tail -1
