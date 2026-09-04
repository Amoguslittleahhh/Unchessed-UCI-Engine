#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
OUT=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_persona_smooth_detect.pgn
LOG=/home/amogusontheterminal/unchessed-ai/results/adapter/sprt_gates/sprt_persona_smooth_detect.log

# Same binary both sides. Adaptive=true is required: decide_mode() returns
# Full/Match immediately when cfg.adaptive is false, so PersonaSmooth and
# EngineDetectV2 never run. Defaults of both options are false (old path);
# this SPRT is the only gate before those defaults may flip. Shape matches
# sprt_punish_latch.sh (tc=5+0.05, elo0=0 elo1=5, Adaptive=true both).
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
  -pgnout "$OUT" \
  > "$LOG" 2>&1
