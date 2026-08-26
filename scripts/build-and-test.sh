#!/usr/bin/env bash
# One-shot build, test, and smoke for the Unchessed Rust workspace.
#
# The sandbox this project develops in has no Rust toolchain and cannot
# reach the toolchain hosts (see "When there is no Rust toolchain" in
# docs/dev-environment.md). On any machine that has one, this script is the
# entire job:
#
#   bash scripts/build-and-test.sh            # debug build + test + smoke
#   bash scripts/build-and-test.sh --release  # additionally: release build
#                                             # (opt-level 3 + LTO, slow)
#
# Toolchain: rustc >= 1.70 (the code uses std::sync::OnceLock; any recent
# stable is fine). To use a toolchain provided at a non-default location,
# point TOOLCHAIN_DIR at the toolchain root (the directory containing bin/):
#
#   TOOLCHAIN_DIR=/opt/rust-1.85.0-x86_64-unknown-linux-gnu \
#     bash scripts/build-and-test.sh
#
# A standard rustup layout (~/.cargo/bin + ~/.rustup) works as-is once
# ~/.cargo/bin is on PATH.

# Disk budget (measured 2026-08-26 on a 21 GB volume with 19 GB free):
# the workspace is small (13.6k lines of core, ZERO external crate
# dependencies — every member is std-only plus path deps), so
#   target/debug  ~200-400 MB,  target (test profile, opt-level 2) ~300-500 MB,
#   target/release (opt 3 + LTO + 1 codegen unit) ~0.5-1 GB.
# Space has never been the blocker; the toolchain is.
set -euo pipefail
cd "$(dirname "$0")/.."

# Optional provided toolchain: TOOLCHAIN_DIR is the toolchain root
# (the directory containing bin/cargo). Plain extracted
# rust-1.7x+-x86_64-unknown-linux-gnu trees work directly; rustup shims work
# too if their ~/.rustup is visible from the default environment.
if [ -n "${TOOLCHAIN_DIR:-}" ]; then
  test -x "$TOOLCHAIN_DIR/bin/cargo" || {
    echo "TOOLCHAIN_DIR=$TOOLCHAIN_DIR has no executable bin/cargo" >&2
    exit 3
  }
  export PATH="$TOOLCHAIN_DIR/bin:$PATH"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found on PATH. Nothing else in this script can proceed;" >&2
  echo "see docs/dev-environment.md for the egress audit of why the sandbox" >&2
  echo "cannot fetch a toolchain itself." >&2
  exit 3
fi

# Guard against a partial checkout before spending minutes on a build.
test -f Cargo.lock
for m in unchessed-core unchessed-adapter unchessed-reviewer unchessed-datagen; do
  test -d "$m/src" || { echo "missing member $m/src" >&2; exit 2; }
done

echo "== cargo build --workspace (debug) =="
cargo build --workspace

echo "== cargo test --workspace (test profile: opt-level 2) =="
# Runs every unit test, including the perft correctness gate, the
# SearchParams/EvalParams defaults, and budget_speeds_up_as_clock_drains.
# The offline toolchain shipped via `arena-rust-toolchain` on PyPI has no
# rustdoc, and this codebase has no doctests, so the doc-test phase is
# skipped (nothing is lost) when rustdoc is absent.
if command -v rustdoc >/dev/null 2>&1; then
  cargo test --workspace
else
  echo "(rustdoc not in this toolchain; no doctests exist, running --lib --bins)"
  cargo test --workspace --lib --bins
fi

ADAPTER=target/debug/unchessed-adapter
test -x "$ADAPTER"

echo "== UCI smoke: startpos depth 5 =="
out=$(printf 'uci\nisready\nposition startpos\ngo depth 5\nquit\n' | "$ADAPTER")
echo "$out" | grep -q "^bestmove" || { echo "no bestmove from startpos search" >&2; exit 1; }
echo "$out" | grep "^bestmove" | head -1

echo "== UCI smoke: matetrack back-rank mate (must find Ra8#) =="
# First position of benchmarks/matetrack.epd; the unique mate is a1a8.
out=$(printf 'uci\nsetoption name Adaptive value false\nposition fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1\ngo depth 6\nquit\n' | "$ADAPTER")
echo "$out" | grep "^bestmove" | head -1
echo "$out" | grep -q "^bestmove a1a8" || { echo "expected bestmove a1a8 (Ra8#)" >&2; exit 1; }

if [ "${1:-}" = "--release" ]; then
  echo "== cargo build --workspace --release (opt 3 + LTO; slow) =="
  cargo build --workspace --release
fi

echo "== disk =="
df -h . | tail -1
echo "OK: build + tests + smoke passed"
