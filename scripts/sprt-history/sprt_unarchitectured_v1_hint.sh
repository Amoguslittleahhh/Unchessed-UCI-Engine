#!/bin/bash
# Isolated paired-game gate for the default-off Unarchitectured v1 root hint.
# No result is claimed by committing this launcher. The owner must provide a
# real opening book, engine binary, model package, and cutechess installation.
set -euo pipefail

: "${ENGINE:?set ENGINE to the candidate unchessed-adapter binary}"
: "${MODEL:?set MODEL to unarchitectured-v1-final.unarchv1}"
: "${BOOK:?set BOOK to the paired-game PGN opening suite}"
: "${OUT:?set OUT to the output PGN path}"
: "${LOG:?set LOG to the output log path}"
CUTECHESS=${CUTECHESS:-cutechess-cli}
CONCURRENCY=${CONCURRENCY:-1}
ROUNDS=${ROUNDS:-5000}

for path in "$ENGINE" "$MODEL" "$BOOK"; do
  test -f "$path" || { echo "missing required file: $path" >&2; exit 2; }
done
command -v "$CUTECHESS" >/dev/null 2>&1 || {
  echo "cutechess-cli unavailable: $CUTECHESS" >&2
  exit 2
}
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

"$CUTECHESS" \
  -engine cmd="$ENGINE" name=Hint \
    option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
    option.UnarchitecturedFile="$MODEL" option.UnarchitecturedHint=true \
    option.UnarchitecturedMinTime=1000 \
  -engine cmd="$ENGINE" name=Baseline \
    option.Threads=1 option.Adaptive=false option.OwnBook=false option.Hash=256 \
    option.UnarchitecturedHint=false \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=16 \
  -repeat -rounds "$ROUNDS" -games 2 \
  -concurrency "$CONCURRENCY" \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=800 \
  -pgnout "$OUT" \
  >"$LOG" 2>&1
