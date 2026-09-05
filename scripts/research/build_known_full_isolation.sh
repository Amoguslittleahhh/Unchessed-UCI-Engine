#!/bin/bash
# Build a candidate that is untouched `main` plus exactly one behavioral
# change: manus/research-facilities's `known_full` MultiPV-narrowing
# block from unchessed-heavy-optimisation/unchessed-core/src/uci.rs
# (commit 63101a8). Nothing else from that branch (Cargo profile,
# adapt.rs, aegis_v4_runtime.rs, etc.) is included, so an SPRT of this
# candidate against unmodified main isolates that one change's effect.
#
# Adapted from manus's own scripts/research/build-known-full-isolation.sh
# (manus/research-facilities commit 56e1860) to this reviewer's WSL
# checkout layout. Build into $HOME, not /tmp -- this WSL image cleans
# /tmp aggressively enough that a freshly built binary can disappear
# within seconds.
set -e
SRC=~/unchessed-sprt-build
OUT=${1:-~/unchessed-known-full-isolation}
rm -rf "$OUT"
cp -a "$SRC" "$OUT"
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
