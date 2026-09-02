#!/bin/bash
# Apply a sidecar of stronger STM cp labels to an existing 104-byte shard.
# Does not run search. Usage:
#   relabel_existing.sh compare old.bin new_scores.i16
#   relabel_existing.sh apply   old.bin new_scores.i16 out.bin
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/tools/nnue_relabel_existing.py" "$@"
