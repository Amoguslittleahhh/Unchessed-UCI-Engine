#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CONFIG=${CONFIG:-$ROOT/config/unarchitectured_v1_training.json}
exec "$ROOT/scripts/training/verda_hydra_v5_multigpu_train.sh" "$@"
