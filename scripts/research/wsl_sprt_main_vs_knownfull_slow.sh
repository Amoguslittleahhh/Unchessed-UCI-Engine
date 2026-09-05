#!/bin/bash
# Real slow-time-control confirmation SPRT for the known_full isolation
# candidate, following up wsl_sprt_main_vs_knownfull_isolated.sh (fast,
# tc=5+0.05, -223.2 +/- 44.6 Elo for unmodified main) and the single-game
# no-increment spot check (wsl_sprt_main_vs_knownfull_slowconfirm_1game.sh,
# which only confirmed the direction didn't reverse, not a real sample).
# This is the proper slower confirmation control the project's standard
# fast+slow gate practice calls for before treating the fast result as
# final.
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
MAIN_ENGINE=~/unchessed-sprt-build/target/release/unchessed-adapter
KNOWNFULL_ENGINE=${1:-~/unchessed-known-full-isolation/target/release/unchessed-adapter}
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_knownfull_slow.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_knownfull_slow.log

"$CUTECHESS" \
  -engine cmd="$MAIN_ENGINE" name=Main option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE" \
  -engine cmd="$KNOWNFULL_ENGINE" name=KnownFullOnly option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE" \
  -each proto=uci tc=20+0.2 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 500 -games 2 \
  -concurrency 7 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -recover \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
echo "SPRT process exited $?"
tail -20 "$LOG"
