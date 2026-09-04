#!/bin/bash
for f in s1 s2 s3 s4 s5 s6; do
  path=~/feature_matrix/log_${f}.txt
  if [ -f "$path" ]; then
    echo "=== $f ==="
    grep -E "Score of|disconnect|No result|Illegal|crashe" "$path" | tail -6
  fi
done
