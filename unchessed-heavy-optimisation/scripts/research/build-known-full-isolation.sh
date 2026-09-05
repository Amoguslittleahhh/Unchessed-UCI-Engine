#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.cargo/env"
ROOT=/home/ubuntu/unchessed-research
MAIN=/home/ubuntu/unchessed-main-inspect
OUT=${1:-/tmp/unchessed-known-full-isolation}
rm -rf "$OUT"
cp -a "$MAIN" "$OUT"
python3 - "$OUT/unchessed-core/src/uci.rs" <<'PY'
from pathlib import Path
p = Path(__import__('sys').argv[1])
s = p.read_text()
old = '''    let multipv_shown = job.opt.multipv;
    let multipv_search = if adaptive_now {
        multipv_shown.max(5)
    } else {
        multipv_shown
    };'''
new = '''    let multipv_shown = job.opt.multipv;
    let known_full = adaptive_now
        && !job.opt.limit_strength
        && model.lock().unwrap().engine_suspect();
    let multipv_search = if adaptive_now && !known_full {
        multipv_shown.max(5)
    } else {
        multipv_shown
    };'''
if old not in s:
    raise SystemExit('main MultiPV block not found; source may have changed')
p.write_text(s.replace(old, new, 1))
PY
cargo build --release --manifest-path "$OUT/Cargo.toml" -p unchessed-adapter
BIN="$OUT/target/release/unchessed-adapter"
sha256sum "$BIN"
printf 'built isolated candidate: %s\n' "$BIN"
printf 'main source unchanged: '
git -C "$MAIN" diff --quiet && echo yes
