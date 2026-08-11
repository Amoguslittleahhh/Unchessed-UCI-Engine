#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/elo_internal_v3
mkdir -p "$OUTDIR"

run_match() {
  local NAME1=$1 ELO1=$2 NAME2=$3 ELO2=$4 TAG=$5
  "$CUTECHESS" \
    -engine cmd="$ADAPTER" name="$NAME1" option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=$ELO1 option.OwnBook=false option.Hash=64 \
    -engine cmd="$ADAPTER" name="$NAME2" option.Threads=1 option.Adaptive=false option.UCI_LimitStrength=true option.UCI_Elo=$ELO2 option.OwnBook=false option.Hash=64 \
    -each proto=uci tc=5+0.05 \
    -openings file="$BOOK" format=pgn order=random plies=10 \
    -repeat -rounds 60 -games 2 -concurrency 13 \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=4 score=1000 \
    -pgnout "$OUTDIR/${TAG}.pgn" \
    > "$OUTDIR/log_${TAG}.txt" 2>&1
  echo "$TAG done"
}

run_match A500 500 A3000 3000 500v3000
run_match A500 500 A1800 1800 500v1800
run_match A1800 1800 A3000 3000 1800v3000
echo "ALL DONE"
