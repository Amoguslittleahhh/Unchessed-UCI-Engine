#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
NEW=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
BASE=/home/amogusontheterminal/unchessed-punishlatch-baseline-src/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_punish_latch.pgn
LOG=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_punish_latch.log

# Two separate binaries (decide_mode()'s PUNISH big-lead-trigger threshold
# only exists as a hardcoded constant, not a UCI-tunable EvalParams field,
# so unlike every prior eval/search-term gate this can't be isolated via a
# same-binary option toggle). Adaptive=true on both sides is required --
# decide_mode() returns early (Mode::Full or Mode::Match) whenever
# cfg.adaptive is false, so the changed code path is never reached
# otherwise. This is the first SPRT gate in this project targeting persona
# logic rather than a pure eval/search term.
"$CUTECHESS" \
  -engine cmd="$NEW" name=PunishLatch100 option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 \
  -engine cmd="$BASE" name=Baseline option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
