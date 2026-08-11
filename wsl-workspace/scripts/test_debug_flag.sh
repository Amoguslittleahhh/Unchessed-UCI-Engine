#!/bin/bash
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=A option.Threads=1 option.Adaptive=true debug \
  -engine cmd="$ADAPTER" name=B option.Threads=1 option.Adaptive=true debug \
  -each proto=uci tc=1+0.01 \
  -openings file="$BOOK" format=pgn order=random plies=6 \
  -rounds 1 -games 1 -concurrency 1 > /tmp/debugtest.log 2>&1
wc -l /tmp/debugtest.log
head -30 /tmp/debugtest.log
