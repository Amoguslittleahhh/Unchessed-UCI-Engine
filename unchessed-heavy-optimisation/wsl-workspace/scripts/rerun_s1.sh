#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
SF=/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/feature_matrix
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=AdaptVsWeak option.Threads=1 option.Adaptive=true option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SF1400 option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 8 -games 2 -concurrency 2 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s1_retry.pgn" \
  > "$OUTDIR/log_s1_retry.txt" 2>&1
echo "s1 retry done"
