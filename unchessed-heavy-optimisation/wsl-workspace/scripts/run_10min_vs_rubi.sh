#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
RUBI=/home/amogusontheterminal/unchessed-ai/data/rubichess_bin/RubiChess-20240817/linux/RubiChess-20240817_x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/tenmin_game
mkdir -p "$OUTDIR"
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=UnchessedAdapter option.Threads=1 option.Adaptive=true option.OwnBook=true option.Hash=128 \
  -engine cmd="$RUBI" name=RubiChessFull option.Threads=1 option.Hash=128 \
  -each proto=uci tc=600+0 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 1 -games 1 -concurrency 1 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=6 score=1000 \
  -pgnout "$OUTDIR/game.pgn" \
  > "$OUTDIR/full_log.txt" 2>&1
echo "GAME DONE"
grep -E "^Finished|^Score" "$OUTDIR/full_log.txt"
