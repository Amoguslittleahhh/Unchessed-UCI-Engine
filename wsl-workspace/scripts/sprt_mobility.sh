#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-ai/builds/unchessed-kingsafety-src/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_mobility_vs_passedpawn.pgn
LOG=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_mobility_vs_passedpawn.log

# Same binary both sides (already has PassedPawnMgPct/EgPct=100 as its
# default, the already-SPRT-validated passed-pawn baseline) -- only
# MobilityPct differs, isolating mobility's own contribution instead of
# conflating it with passed pawns' already-proven gain.
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Mobility option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.MobilityPct=100 \
  -engine cmd="$ENGINE" name=Baseline option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 option.MobilityPct=0 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
