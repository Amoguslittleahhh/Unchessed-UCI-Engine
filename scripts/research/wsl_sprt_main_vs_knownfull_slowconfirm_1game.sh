#!/bin/bash
# Single-game, no-increment slow-time-control spot check of the same
# known_full isolation candidate as wsl_sprt_main_vs_knownfull_isolated.sh.
# NOT a statistically meaningful sample on its own (n=1, LOS is
# uninformative) -- this only checks that the fast-time-control result's
# direction doesn't reverse at a slower control before spending real
# compute on a proper slow confirmation SPRT.
#
# Result: Main (White) lost, KnownFullOnly won -- same direction as the
# fast result, no reversal, but still just one game.
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
MAIN_ENGINE=~/unchessed-sprt-build/target/release/unchessed-adapter
KNOWNFULL_ENGINE=${1:-~/unchessed-known-full-isolation/target/release/unchessed-adapter}
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_knownfull_slowconfirm_1game.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_knownfull_slowconfirm_1game.log

"$CUTECHESS" \
  -engine cmd="$MAIN_ENGINE" name=Main option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE" \
  -engine cmd="$KNOWNFULL_ENGINE" name=KnownFullOnly option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE" \
  -each proto=uci tc=20 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -rounds 1 -games 1 \
  -concurrency 1 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
echo "match process exited $?"
cat "$LOG"
