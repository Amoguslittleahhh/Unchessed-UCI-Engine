#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix/release/unchessed-adapter
SF=/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_ladder_v2
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.pgn

# 64 levels, evenly spaced 500..3200 inclusive
python3 -c "
lo, hi, n = 500, 3200, 64
for i in range(n):
    print(round(lo + i*(hi-lo)/(n-1)))
" > "$OUTDIR/levels.txt"

# Sequential, low concurrency -- an SPRT gate (concurrency=13) is already
# running on this box; running 64 parallel matches on top would oversubscribe
# the CPU and corrupt BOTH this test's and the SPRT's real-time-based results.
while read -r ELO; do
  echo "=== ELO $ELO ==="
  "$CUTECHESS" \
    -engine cmd="$ADAPTER" name="Adapter$ELO" option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=$ELO option.OwnBook=false option.Hash=64 \
    -engine cmd="$SF" name=Stockfish option.Threads=1 option.Hash=64 \
    -each proto=uci tc=5+0.05 \
    -openings file="$BOOK" format=pgn order=random plies=10 \
    -rounds 1 -games 1 -concurrency 1 \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=4 score=1000 \
    -pgnout "$OUTDIR/elo_$ELO.pgn" \
    > "$OUTDIR/log_$ELO.txt" 2>&1
  echo "  done"
done < "$OUTDIR/levels.txt"
echo "ALL DONE"
