#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter

# 2-game smoke that the UCI options are accepted before a full SPRT.
"$CUTECHESS" \
  -engine cmd="$ENGINE" name=A option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=64 option.PersonaSmooth=true option.EngineDetectV2=true \
  -engine cmd="$ENGINE" name=B option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=64 option.PersonaSmooth=false option.EngineDetectV2=false \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=8 \
  -repeat -rounds 1 -games 2 \
  -concurrency 1
