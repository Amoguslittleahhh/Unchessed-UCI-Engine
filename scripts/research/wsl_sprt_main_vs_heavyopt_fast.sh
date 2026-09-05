#!/bin/bash
# Real cutechess SPRT: main vs manus/research-facilities's
# unchessed-heavy-optimisation copy, per Manus's predeclared spec
# (relayed 2026-09-05). Fast screening stage.
#
# - Adaptive=true both sides (screens persona/adaptive-path behaviour,
#   not just raw search strength)
# - OwnBook=false both sides
# - AdapterTelemetry=true both sides -- feeds
#   tools/analyse_manus_gate_buckets.py for the mode/low-time/transition
#   buckets Manus asked for
# - identical EvalFile: both engines point at the literal same file,
#   not just a hash-matched copy
# - -repeat pairs every opening with colors reversed
# - elo0=0 elo1=5 alpha=beta=0.05, matching this project's existing
#   SPRT-gate convention (scripts/sprt-history/*.sh)
set -e
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
NNUE=~/unchessed-sprt-build/unchessed-nnue.bin
MAIN_WRAP=~/unchessed-ai/data/main_telemetry_wrapper.sh
HEAVYOPT_WRAP=~/unchessed-ai/data/heavyopt_telemetry_wrapper.sh
OUT=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_heavyopt_fast.pgn
LOG=~/unchessed-ai/results/adapter/sprt_gates/sprt_main_vs_heavyopt_fast.log

# cutechess-cli 1.5.1's bare -debug flag fails immediately in this build
# ("Empty value for option -debug", reproduced with a minimal 2-engine
# repro -- not something specific to this match config). Telemetry is
# captured instead by routing each engine through a tee-wrapper
# (data/main_telemetry_wrapper.sh / heavyopt_telemetry_wrapper.sh) that
# duplicates its stdout to a per-process log file under
# results/adapter/sprt_gates/telemetry_{main,heavyopt}/.
#
# PersonaSmooth and EngineDetectV2 are pinned explicitly on BOTH sides.
# The heavy-optimisation binary's Default impl silently ships
# persona_smooth: true (unchessed-core/src/uci.rs:141) while main's is
# false (uci.rs:117) and neither is a promoted default per
# docs/persona-sprt-gate.md. Leaving these unset caused a first attempt
# at this run to compare "heavy-optimisation + PersonaSmooth=true" vs
# "main + PersonaSmooth=false" instead of isolating hardware/build
# differences -- that run was killed after ~36 games once the divergent
# telemetry surfaced it.
"$CUTECHESS" \
  -engine cmd="$MAIN_WRAP" name=Main option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.AdapterTelemetry=true option.PersonaSmooth=false option.EngineDetectV2=false option.EvalFile="$NNUE" \
  -engine cmd="$HEAVYOPT_WRAP" name=HeavyOpt option.Threads=1 option.Adaptive=true option.OwnBook=false option.Hash=256 option.AdapterTelemetry=true option.PersonaSmooth=false option.EngineDetectV2=false option.EvalFile="$NNUE" \
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
