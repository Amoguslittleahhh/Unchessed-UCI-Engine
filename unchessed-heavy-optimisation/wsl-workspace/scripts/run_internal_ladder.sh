#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix2/release/unchessed-adapter
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_internal
mkdir -p "$OUTDIR"

"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=A500 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=500 option.OwnBook=false option.Hash=64 \
  -engine cmd="$ADAPTER" name=A3000 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=3000 option.OwnBook=false option.Hash=64 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 20 -games 2 -concurrency 4 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/500_vs_3000.pgn" \
  > "$OUTDIR/log_500v3000.txt" 2>&1
echo "500v3000 done"

"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=A500 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=500 option.OwnBook=false option.Hash=64 \
  -engine cmd="$ADAPTER" name=A1800 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=1800 option.OwnBook=false option.Hash=64 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 20 -games 2 -concurrency 4 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/500_vs_1800.pgn" \
  > "$OUTDIR/log_500v1800.txt" 2>&1
echo "500v1800 done"

"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=A1800 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=1800 option.OwnBook=false option.Hash=64 \
  -engine cmd="$ADAPTER" name=A3000 option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=3000 option.OwnBook=false option.Hash=64 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 20 -games 2 -concurrency 4 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/1800_vs_3000.pgn" \
  > "$OUTDIR/log_1800v3000.txt" 2>&1
echo "1800v3000 done"
echo "ALL DONE"
