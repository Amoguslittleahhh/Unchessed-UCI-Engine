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

# Rust needs a native C linker (cc/gcc/clang) to produce a binary at all --
# without this check, a missing one only surfaces minutes later as a bare
# "linker `cc` not found" from deep inside `cargo build`'s first link step,
# after the compile-only work already ran. Skipped on Windows: MSVC's
# link.exe comes from a Visual Studio install cargo already knows how to
# find via its own toolchain detection, not a plain PATH lookup here.
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) ;;  # MSVC toolchain, not a PATH-visible cc/gcc/clang
  *)
    if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
      echo "no native C linker (cc/gcc/clang) found on PATH." >&2
      echo "cargo needs one to link any binary -- install build-essential" >&2
      echo "(Debian/Ubuntu) or the equivalent for this OS before continuing." >&2
      exit 3
    fi
    ;;
esac

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

echo "== cargo test --workspace perft::tests::deep -- --ignored (excluded from the default run above) =="
# perft::tests::deep is #[ignore]d (it's slow, ~13s) and this is the only
# place that supplies --ignored -- without it, the deepest correctness gate
# this workspace has never actually runs as part of a normal gate pass.
cargo test --workspace perft::tests::deep -- --ignored

ADAPTER=target/debug/unchessed-adapter
test -x "$ADAPTER"

# `go` starts the real search in a background thread inside the adapter and
# returns control to its UCI read loop immediately; sending `quit` right
# after in the same pipe races that search against a stop request a real
# GUI would never issue mid-search (it always waits for `bestmove` first).
# The engine now (correctly) honors `stop`/`quit` essentially immediately
# rather than after an up-to-~2048-node grace window, so what used to be a
# lucky, unintentional head start is no longer there to mask this: without
# a real pause here, `quit` can arrive before a single depth completes,
# discarding this smoke test's power to check anything. These are trivial,
# sub-10ms searches once actually running, so a short fixed pause is not a
# meaningful cost.
echo "== UCI smoke: startpos depth 5 =="
out=$( { printf 'uci\nisready\nposition startpos\ngo depth 5\n'; sleep 2; printf 'quit\n'; } | "$ADAPTER")
echo "$out" | grep -q "^bestmove" || { echo "no bestmove from startpos search" >&2; exit 1; }
echo "$out" | grep "^bestmove" | head -1

echo "== UCI smoke: matetrack back-rank mate (must find Ra8#) =="
# First position of benchmarks/matetrack.epd; the unique mate is a1a8.
out=$( { printf 'uci\nsetoption name Adaptive value false\nposition fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1\ngo depth 6\n'; sleep 2; printf 'quit\n'; } | "$ADAPTER")
echo "$out" | grep "^bestmove" | head -1
echo "$out" | grep -q "^bestmove a1a8" || { echo "expected bestmove a1a8 (Ra8#)" >&2; exit 1; }

if [ "${1:-}" = "--release" ]; then
  echo "== cargo build --workspace --release (opt 3 + LTO; slow) =="
  cargo build --workspace --release

  echo "== cargo test --workspace --release (the debug-profile pass above never" \
       "exercises the optimized code path actually shipped) =="
  if command -v rustdoc >/dev/null 2>&1; then
    cargo test --workspace --release
  else
    cargo test --workspace --release --lib --bins
  fi
fi

echo "== disk =="
df -h . | tail -1
echo "OK: build + tests + smoke passed"
