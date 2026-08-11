#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
RUBI=/home/amogusontheterminal/unchessed-ai/data/rubichess_bin/RubiChess-20240817/linux/RubiChess-20240817_x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_scaling_rubi_v2
mkdir -p "$OUTDIR"

LEVELS="500 1000 2000 4000 8000 16000 32000"
for NPS in $LEVELS; do
  echo "=== RubiChess LimitNps=$NPS vs Adapter(UCI_Elo=2400) ==="
  "$CUTECHESS" \
    -engine cmd="$ADAPTER" name="Adapter2400" option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=2400 option.OwnBook=false option.Hash=64 \
    -engine cmd="$RUBI" name="Rubi$NPS" option.Threads=1 option.Hash=64 option.LimitNps=$NPS \
    -each proto=uci tc=5+0.05 \
    -openings file="$BOOK" format=pgn order=random plies=10 \
    -rounds 20 -games 2 -concurrency 8 \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=4 score=1000 \
    -pgnout "$OUTDIR/nps_$NPS.pgn" \
    > "$OUTDIR/log_$NPS.txt" 2>&1
  grep "Score of" "$OUTDIR/log_$NPS.txt" | tail -1
done
echo "CALIBRATION DONE"
