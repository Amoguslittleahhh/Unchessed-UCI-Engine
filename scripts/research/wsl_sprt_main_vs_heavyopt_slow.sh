#!/bin/bash
# Slower confirmation control for the main-vs-heavyopt gate (see
# wsl_sprt_main_vs_heavyopt_fast.sh for the full rationale). Only run
# this after the fast screen finishes -- per Manus's staged design,
# a confirmation control at a different time control is meant to catch
# a hardware-optimization effect that is time-control-dependent, not to
# replace the fast screen.
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
MAIN_WRAP=~/unchessed-ai/data/main_telemetry_wrapper.sh
HEAVYOPT_WRAP=~/unchessed-ai/data/heavyopt_telemetry_wrapper.sh
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_heavyopt_slow.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_heavyopt_slow.log

# See wsl_sprt_main_vs_heavyopt_fast.sh for why -debug is not used here,
# and for why PersonaSmooth/EngineDetectV2 must be pinned explicitly
# (heavy-optimisation's binary defaults PersonaSmooth to true, main's
# defaults it to false -- an unpinned run silently compares that
# instead of the intended hardware/build difference).
"$CUTECHESS" \
  -engine cmd="$MAIN_WRAP" name=Main option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.AdapterTelemetry=true option.PersonaSmooth=false option.EngineDetectV2=false option.EvalFile="$NNUE" \
  -engine cmd="$HEAVYOPT_WRAP" name=HeavyOpt option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.AdapterTelemetry=true option.PersonaSmooth=false option.EngineDetectV2=false option.EvalFile="$NNUE" \
  -each proto=uci tc=20+0.2 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 5000 -games 2 \
  -concurrency 13 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -recover \
  -pgnout "$OUT" \
  > "$LOG" 2>&1
echo "SPRT process exited $?"
tail -30 "$LOG"
