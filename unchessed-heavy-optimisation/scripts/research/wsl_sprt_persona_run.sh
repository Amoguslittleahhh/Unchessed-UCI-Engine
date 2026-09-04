#!/bin/bash
# Real cutechess SPRT for PersonaSmooth/EngineDetectV2, on this reviewer's
# hardware (WSL) -- the gate arena's sandbox cannot run (no cutechess
# there). Same shape as scripts/sprt-history/sprt_persona_smooth_detect.sh,
# ENGINE path adapted to this checkout (~/unchessed-sprt-build, built at
# main after round 19).
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=~/unchessed-sprt-build/target/release/unchessed-adapter
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_persona_smooth_detect.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_persona_smooth_detect.log

"$CUTECHESS" \
  -engine cmd="$ENGINE" name=PersonaV2 option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.PersonaSmooth=true option.EngineDetectV2=true \
  -engine cmd="$ENGINE" name=Baseline option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.PersonaSmooth=false option.EngineDetectV2=false \
  -each proto=uci tc=5+0.05 \
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
