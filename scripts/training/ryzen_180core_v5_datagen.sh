#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

BASE_CONFIG=${CONFIG:-config/v5_180core_datagen.json}
CPU_PROFILE_CONFIG=${CPU_PROFILE_CONFIG:-config/verda_cpu_profiles.json}
PLAN=${PLAN:-/data/unchessed/guide-v5/plan.json}
MANIFEST=${MANIFEST:-/data/unchessed/guide-v5/MANIFEST.json}
RESOLVED_CONFIG="$(dirname "$PLAN")/resolved-cpu-datagen.json"

mkdir -p "$(dirname "$PLAN")" "$(dirname "$MANIFEST")"
python3 tools/verda_cpu_profile.py resolve \
  --profiles "$CPU_PROFILE_CONFIG" --base-config "$BASE_CONFIG" \
  --output "$RESOLVED_CONFIG" >/dev/null
CONFIG="$RESOLVED_CONFIG"
VCPU_COUNT=$(python3 - "$CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["cpu"]["verda_vcpus"])
PY
)

VERDA_DATA_PATH=${VERDA_DATA_PATH:-/data}
python3 tools/verda_v5_preflight.py --role cpu \
  --data-path "$VERDA_DATA_PATH" --expected-logical-cpus "$VCPU_COUNT" --strict \
  --json "$(dirname "$PLAN")/verda-cpu-preflight.json"
python3 tools/v5_180core_datagen.py topology --config "$CONFIG"
if [[ -f "$PLAN" ]]; then
  python3 - "$CONFIG" "$PLAN" <<'PY'
import hashlib, json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
plan = json.load(open(sys.argv[2], encoding="utf-8"))
digest = hashlib.sha256(
    json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if plan.get("config_sha256") != digest:
    raise SystemExit("existing plan was built for a different CPU profile/config; move or delete it")
PY
else
  python3 tools/v5_180core_datagen.py plan --config "$CONFIG" --output "$PLAN"
fi
python3 tools/v5_180core_datagen.py run --plan "$PLAN" --dry-run
python3 tools/v5_180core_datagen.py run --plan "$PLAN"
python3 tools/v5_180core_datagen.py status --plan "$PLAN"
python3 tools/v5_180core_datagen.py finalize --plan "$PLAN" --output "$MANIFEST"
echo "$VCPU_COUNT-vCPU teacher labeling complete: $MANIFEST"
