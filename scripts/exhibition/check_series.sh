#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  echo "--- game $i ---"
  grep "RESULT:" ~/unchessed-ai/results/tenmin_3min_series/game$i/moves.txt
  head -c 250 ~/unchessed-ai/results/tenmin_3min_series/game$i/game.pgn | tail -1
  echo
done
