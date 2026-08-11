#!/bin/bash
for ELO in 500 1000 1500 2000 2500 3000; do
  path=~/elo_scaling_rubi/log_${ELO}.txt
  echo "=== ELO $ELO ==="
  grep -E "Score of|disconnect|No result" "$path" | tail -3
done
