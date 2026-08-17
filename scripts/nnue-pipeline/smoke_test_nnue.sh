#!/bin/bash
# Pre-SPRT smoke test for a candidate NNUE network.
#
# Backlog item: research/remaining_research_topics.md item #92, adopted
# from the IEEE research doc's Appendix C recommendation. v3's -70.3 Elo
# regression took ~756 SPRT games to catch; this runs a much cheaper
# fixed-game-count match first (100 games @ 1+0.1s, reject if score < 40%)
# so an obviously-bad architecture never reaches a full SPRT run at all.
#
# This is a REJECT-ONLY filter, not a substitute for SPRT: a pass here
# means "not obviously terrible," not "good." Only a full SPRT gate
# (see scripts/sprt-history/sprt_nnue_v4.sh for the pattern) confirms a
# real Elo gain.
#
# Usage:
#   ./smoke_test_nnue.sh <candidate.bin> [baseline.bin] [label]
#
# candidate.bin  - the new network being screened
# baseline.bin   - what it's compared against (default: the currently
#                  deployed unchessed-nnue.bin next to the adapter binary)
# label          - short name used in output file naming (default: "candidate")

set -e

CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ENGINE=/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter
DEFAULT_BASELINE=/home/amogusontheterminal/unchessed-kingsafety-src/unchessed-nnue.bin
RESULTS_DIR=/home/amogusontheterminal/unchessed-ai/results/adapter/smoke_tests

PASS_THRESHOLD=40.0

CANDIDATE="$1"
BASELINE="${2:-$DEFAULT_BASELINE}"
LABEL="${3:-candidate}"

if [ -z "$CANDIDATE" ]; then
  echo "Usage: $0 <candidate.bin> [baseline.bin] [label]"
  exit 2
fi
if [ ! -f "$CANDIDATE" ]; then
  echo "Candidate NNUE file not found: $CANDIDATE"
  exit 2
fi
if [ ! -f "$BASELINE" ]; then
  echo "Baseline NNUE file not found: $BASELINE"
  exit 2
fi

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/smoke_${LABEL}.pgn"
LOG="$RESULTS_DIR/smoke_${LABEL}.log"

echo "Smoke test: $LABEL"
echo "  candidate: $CANDIDATE"
echo "  baseline:  $BASELINE"
echo "  100 games @ 1+0.1s, pass threshold ${PASS_THRESHOLD}%"
echo ""

"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Candidate option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=64 option.EvalFile="$CANDIDATE" \
  -engine cmd="$ENGINE" name=Baseline  option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=64 option.EvalFile="$BASELINE" \
  -each proto=uci tc=1+0.1 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds 50 -games 2 \
  -concurrency 13 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  > "$LOG" 2>&1

# cutechess-cli prints a running "Score of Candidate vs Baseline: W - L - D  [pct] N"
# line after every game; the last one in the log is the final result.
FINAL_LINE=$(grep "Score of Candidate vs Baseline" "$LOG" | tail -1)
echo "$FINAL_LINE"

if [ -z "$FINAL_LINE" ]; then
  echo "Could not parse a final score line from $LOG -- check the log directly."
  exit 2
fi

# Extract the bracketed win-rate fraction, e.g. "[0.386]" -> 38.6
FRACTION=$(echo "$FINAL_LINE" | grep -oP '\[\K[0-9.]+(?=\])')
if [ -z "$FRACTION" ]; then
  echo "Could not parse win-rate fraction -- check the log directly: $LOG"
  exit 2
fi

SCORE_PCT=$(awk -v f="$FRACTION" 'BEGIN { printf "%.1f", f * 100 }')
echo ""
echo "Candidate score: ${SCORE_PCT}%"

PASSED=$(awk -v s="$SCORE_PCT" -v t="$PASS_THRESHOLD" 'BEGIN { print (s >= t) ? "1" : "0" }')
if [ "$PASSED" = "1" ]; then
  echo "RESULT: PASS (>= ${PASS_THRESHOLD}%) -- not obviously bad, proceed to full SPRT if desired."
  exit 0
else
  echo "RESULT: FAIL (< ${PASS_THRESHOLD}%) -- reject before spending a full SPRT run on this."
  exit 1
fi
