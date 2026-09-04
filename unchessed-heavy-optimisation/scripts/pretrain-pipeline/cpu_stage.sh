#!/usr/bin/env bash
# ============================================================================
# PRETRAIN PIPELINE — CPU STAGE (runs on the CPU.180V.720G box)
#
# Takes the game set produced by tools/maia3_cloud_selfplay/generate.py
# and turns it into v5 dual-elo shards (stage-1 full + stage-2
# trusted-only) that the GPU stage trains on.
#
# The build is two-pass and parallel: pass 1 is a text-only scan that
# fixes the game-disjoint train/val split, pass 2 replays one PGN file
# per worker and streams the final shards directly (no temp files).
# Output bytes are identical regardless of worker count (tested);
# workers default to min(cpus-2, 128), so on CPU.360V.1440G the 5M set
# builds with 128 parallel file-workers.
#
# Usage:
#   scripts/pretrain-pipeline/cpu_stage.sh <games_dir> <out_root> [val_games]
#
#   <games_dir>   generate.py --out directory (has pgn/shard-*.pgn +
#                 labels/ + manifest.json)
#   <out_root>    where pretrain-v5/ and pretrain-v5-trusted/ are written
#   [val_games]   games held out for validation (default 20000)
#
# Output:
#   <out_root>/pretrain-v5/          stage-1 data (all quality rows)
#   <out_root>/pretrain-v5-trusted/  stage-2 data (calibrated+native+human
#                                    games only — no approximate rows)
#   both with manifest.json + validation.json
#
# Then rsync both directories to the A100 box and run gpu_stage.sh.
# ============================================================================
set -euo pipefail

GAMES_DIR="${1:?usage: cpu_stage.sh <games_dir> <out_root> [val_games]}"
OUT_ROOT="${2:?usage: cpu_stage.sh <games_dir> <out_root> [val_games]}"
VAL_GAMES="${3:-20000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/venv/bin/python"

[ -d "$GAMES_DIR/pgn" ] || { echo "not a generator output dir: $GAMES_DIR" >&2; exit 1; }

if [ ! -x "$PY" ]; then
  echo "[cpu] creating venv with python-chess..."
  (cd "$REPO_ROOT" && python3 -m venv venv
   venv/bin/pip install -q -r tools/requirements-dev.txt)
fi

echo "[cpu] stage-1 (all rows): building v5 shards from $GAMES_DIR/pgn/*.pgn"
"$PY" "$REPO_ROOT/tools/pretrain_v5_data.py" build \
    --pgn "$GAMES_DIR"/pgn/shard-*.pgn \
    --out "$OUT_ROOT/pretrain-v5" \
    --val-games "$VAL_GAMES"

echo "[cpu] stage-2 (trusted-only: calibrated+native+human)"
"$PY" "$REPO_ROOT/tools/pretrain_v5_data.py" build \
    --pgn "$GAMES_DIR"/pgn/shard-*.pgn \
    --out "$OUT_ROOT/pretrain-v5-trusted" \
    --quality-filter calibrated,native,human \
    --val-games "$VAL_GAMES"

echo "[cpu] validating both sets"
"$PY" "$REPO_ROOT/tools/pretrain_v5_data.py" validate \
    --dir "$OUT_ROOT/pretrain-v5"
"$PY" "$REPO_ROOT/tools/pretrain_v5_data.py" validate \
    --dir "$OUT_ROOT/pretrain-v5-trusted"

du -sh "$OUT_ROOT/pretrain-v5" "$OUT_ROOT/pretrain-v5-trusted"
echo "[cpu] done. Hand off to the A100 box:"
echo "  rsync -a $OUT_ROOT/pretrain-v5 $OUT_ROOT/pretrain-v5-trusted <a100>:$OUT_ROOT/"
echo "  then: scripts/pretrain-pipeline/gpu_stage.sh $OUT_ROOT"
