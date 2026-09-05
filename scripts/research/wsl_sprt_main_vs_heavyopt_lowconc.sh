#!/bin/bash
# Control rerun of the main-vs-heavyopt gate without the telemetry
# tee-wrapper, and with -concurrency matched to the host's core count
# (7 game-pairs * 2 threads = 14 cores, no oversubscription) instead of
# the original -concurrency 13 on 14 cores (up to 26 engine processes
# plus 26 tee/stdbuf wrapper processes -- real oversubscription at a
# fast time control). This was run to rule out CPU-contention and
# wrapper-overhead as an alternative explanation for the -244 Elo
# result in wsl_sprt_main_vs_heavyopt_fast.sh before trusting it.
#
# Result: -238.1 +/- 50.7 Elo (200 games), statistically indistinguishable
# from the original -244.4 +/- 48.0 -- confirms the effect is real, not
# a measurement artifact of concurrency or the wrapper.
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
MAIN_ENGINE=~/unchessed-sprt-build/target/release/unchessed-adapter
HEAVYOPT_ENGINE=~/unchessed-heavy-opt-build/unchessed-heavy-optimisation/target/release/unchessed-adapter
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_heavyopt_lowconc.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_heavyopt_lowconc.log

"$CUTECHESS" \
  -engine cmd="$MAIN_ENGINE" name=Main option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.PersonaSmooth=false option.EngineDetectV2=false option.EvalFile="$NNUE" \
  -engine cmd="$HEAVYOPT_ENGINE" name=HeavyOpt option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.PersonaSmooth=false option.EngineDetectV2=false option.EvalFile="$NNUE" \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 100 -games 2 \
  -concurrency 7 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
echo "match process exited $?"
tail -30 "$LOG"
