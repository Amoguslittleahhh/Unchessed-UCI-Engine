#!/bin/bash
# SPRT: current default NNUE (v3, unchessed-nnue.bin) vs. the v4 net trained
# on the same 27M real self-play records as wsl_sprt_nnue_real27m.sh, but
# with round 13's defended recipe (batch 65536, 15-epoch cap, early-stop
# patience 3, best-checkpoint export -- best was epoch 2 at 49.3cp val-mae,
# vs. the old recipe's last-epoch export at 54.3cp). Isolates the recipe
# fix's own effect before any cloud spend on more data.
set -u
cd ~/unchessed-sprt-build
BIN=$(pwd)/target/release/unchessed-adapter
BASE_NET=$(pwd)/unchessed-nnue.bin
NEW_NET=$(pwd)/nnue-round12-test/nnue_v4_27m_recipe.bin
CUTECHESS=~/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=~/unchessed-ai/data/maia-data/sprt_book.pgn
OUT=~/unchessed-sprt-build/nnue-round12-test

"$CUTECHESS" \
  -engine name=Base-v3 cmd="$BIN" option.EvalFile="$BASE_NET" option.Threads=1 option.OwnBook=false \
  -engine name=New-v4-27m-recipe cmd="$BIN" option.EvalFile="$NEW_NET" option.Threads=1 option.OwnBook=false \
  -each proto=uci tc=10+0.1 \
  -openings file="$BOOK" format=pgn order=random \
  -games 2 -repeat -rounds 5000 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -concurrency 7 -recover \
  -pgnout "$OUT/sprt_games_27m_recipe.pgn" \
  > "$OUT/sprt_log_27m_recipe.txt" 2>&1
echo "SPRT process exited $?"
tail -30 "$OUT/sprt_log_27m_recipe.txt"
