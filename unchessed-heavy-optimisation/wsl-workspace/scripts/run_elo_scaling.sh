#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
SF=/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_scaling
mkdir -p "$OUTDIR"

# Fixed mid-strength reference so low targets lose badly and high targets
# win/draw more, giving a real spread across the whole tested range instead
# of saturating at 0% or 100% (which is what happened when the reference
# was too weak or too strong in earlier tests).
LEVELS="500 1000 1500 2000 2500 3000"
for ELO in $LEVELS; do
  echo "=== ELO $ELO vs SF@1800 ==="
  "$CUTECHESS" \
    -engine cmd="$ADAPTER" name="Adapter$ELO" option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=$ELO option.OwnBook=false option.Hash=64 \
    -engine cmd="$SF" name=SF1800 option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=1800 \
    -each proto=uci tc=5+0.05 \
    -openings file="$BOOK" format=pgn order=random plies=10 \
    -rounds 15 -games 2 -concurrency 4 \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=4 score=1000 \
    -pgnout "$OUTDIR/elo_$ELO.pgn" \
    > "$OUTDIR/log_$ELO.txt" 2>&1
  echo "  done"
done
echo "ALL DONE"
