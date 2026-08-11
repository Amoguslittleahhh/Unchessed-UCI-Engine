#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix2/release/unchessed-adapter
SF=/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_points
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.pgn

# fewer levels, many games each -- statistical power over breadth
LEVELS="500 1200 1800 2400 3000"
GAMES_PER_LEVEL=16

for ELO in $LEVELS; do
  echo "=== ELO $ELO ($GAMES_PER_LEVEL games) ==="
  "$CUTECHESS" \
    -engine cmd="$ADAPTER" name="Adapter$ELO" option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=$ELO option.OwnBook=false option.Hash=64 \
    -engine cmd="$SF" name=Stockfish option.Threads=1 option.Hash=64 \
    -each proto=uci tc=5+0.05 \
    -openings file="$BOOK" format=pgn order=random plies=10 \
    -rounds $((GAMES_PER_LEVEL / 2)) -games 2 -concurrency 4 \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=4 score=1000 \
    -pgnout "$OUTDIR/elo_$ELO.pgn" \
    > "$OUTDIR/log_$ELO.txt" 2>&1
  echo "  done"
done
echo "ALL DONE"
