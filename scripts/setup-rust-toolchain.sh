#!/bin/bash
# Idempotent Rust toolchain bootstrap for a fresh/ephemeral sandbox.
#
# Why this exists: multiple arena rounds (15, 18) pushed Rust changes
# that don't compile, because the sandbox had no rustc and the round's
# own doc just noted "(need rustc)" as a caveat instead of installing
# one. That caveat is not a substitute for actually building the code.
# `rust-toolchain.toml` in the repo root pins the toolchain version once
# rustup exists; this script gets rustup itself onto a sandbox that has
# none, in one command, safe to re-run.
#
# Usage: source this (not just execute it) so cargo/rustc land on PATH
# in the current shell without needing a fresh login:
#   source scripts/setup-rust-toolchain.sh
set -u

if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
  echo "rustc/cargo already on PATH: $(rustc --version), $(cargo --version)"
  return 0 2>/dev/null || exit 0
fi

if [ -f "$HOME/.cargo/env" ]; then
  # rustup was installed before in this filesystem, just not sourced yet
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
  if command -v cargo >/dev/null 2>&1; then
    echo "found existing rustup install, sourced $HOME/.cargo/env: $(rustc --version)"
    return 0 2>/dev/null || exit 0
  fi
fi

echo "no rustc found -- installing via rustup (official installer, non-interactive)..."
if ! curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default; then
  echo "ERROR: rustup install failed. If this sandbox has no outbound network access," >&2
  echo "that is the real blocker -- flag it explicitly in the round's doc instead of" >&2
  echo "silently pushing unbuilt code." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
. "$HOME/.cargo/env"
echo "installed: $(rustc --version), $(cargo --version)"
echo "cargo test --workspace --release now has no excuse to skip."
