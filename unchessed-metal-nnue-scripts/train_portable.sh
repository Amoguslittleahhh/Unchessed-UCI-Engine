#!/usr/bin/env bash
# Wrapper kept beside the data pipeline; the implementation remains the
# original root trainer and portable resource policy.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$ROOT/scripts/train_nnue_portable.sh" "$@"
