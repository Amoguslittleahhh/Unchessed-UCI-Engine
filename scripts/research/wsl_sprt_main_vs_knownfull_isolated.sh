#!/bin/bash
# Decisive isolation SPRT: unmodified main vs. main + exactly one change
# (manus/research-facilities's known_full MultiPV-narrowing, built by
# build_known_full_isolation.sh). This is the test that confirmed the
# ~240 Elo gap between main and unchessed-heavy-optimisation is
# essentially entirely explained by this one change, not by anything
# else on that branch (Cargo profile, aegis_v4_runtime.rs, etc.).
#
# Result: 233 games, Elo difference -223.2 +/- 44.6 for unmodified main,
# LOS 0.0%, SPRT bound hit ("H0 accepted" -- cutechess's H0/H1 labels
# are relative to the first-named engine, Main here, so this means Main
# was not proven better, which is trivially true since it lost
# decisively). Cross-validated independently by
# manus/research-facilities on different hardware: +147.2 +/- 160.4 Elo
# (20 games, wide interval but same direction).
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
MAIN_ENGINE=~/unchessed-sprt-build/target/release/unchessed-adapter
KNOWNFULL_ENGINE=${1:-~/unchessed-known-full-isolation/target/release/unchessed-adapter}
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_knownfull_isolated.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_knownfull_isolated.log

"$CUTECHESS" \
  -engine cmd="$MAIN_ENGINE" name=Main option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE" \
  -engine cmd="$KNOWNFULL_ENGINE" name=KnownFullOnly option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.EvalFile="$NNUE" \
  -each proto=uci tc=5+0.05 \
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
