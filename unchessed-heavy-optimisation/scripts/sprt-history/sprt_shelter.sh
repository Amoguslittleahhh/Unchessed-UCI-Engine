#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_shelter.pgn
LOG=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_shelter.log

# Same binary both sides (already has PassedPawnMgPct/EgPct/MobilityPct/
# RookPct=100 as defaults, all already SPRT-validated) -- only ShelterPct
# differs, isolating the pawn shelter/storm term's own contribution.
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Shelter option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.ShelterPct=100 \
  -engine cmd="$ENGINE" name=Baseline option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.ShelterPct=0 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
