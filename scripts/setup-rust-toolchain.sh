#!/bin/bash
# Idempotent Rust toolchain bootstrap for a fresh/ephemeral sandbox.
#
# Tries, in order:
#   1. rustc/cargo already on PATH
#   2. existing rustup at $HOME/.cargo/env
#   3. Debian/Ubuntu apt (rustc + cargo)
#   4. official rustup installer (https://sh.rustup.rs)
#
# Each step prints what it did. If both apt and rustup fail, that is a
# real environment blocker — paste this script's stdout into the round
# doc instead of pushing unbuilt Rust.
#
# Usage (source so PATH updates in the current shell):
#   source scripts/setup-rust-toolchain.sh
set -u

report() { echo "[setup-rust-toolchain] $*"; }

if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
  report "already on PATH: $(rustc --version), $(cargo --version)"
  return 0 2>/dev/null || exit 0
fi

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    report "sourced $HOME/.cargo/env: $(rustc --version), $(cargo --version)"
    return 0 2>/dev/null || exit 0
  fi
  report "found $HOME/.cargo/env but rustc/cargo still missing"
fi

APT_OK=0
RUSTUP_OK=0

report "apt path: trying rustc + cargo..."
if command -v apt-get >/dev/null 2>&1; then
  if sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
     sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rustc cargo; then
    if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
      report "apt path: OK — $(rustc --version), $(cargo --version)"
      APT_OK=1
    else
      report "apt path: packages claimed install but rustc/cargo not on PATH"
    fi
  else
    report "apt path: FAILED (update/install error; often a filtered Debian CDN)"
  fi
else
  report "apt path: skipped (no apt-get)"
fi

if [ "$APT_OK" -eq 1 ]; then
  return 0 2>/dev/null || exit 0
fi

report "rustup path: curl https://sh.rustup.rs ..."
set +e
set -o pipefail
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default
RUSTUP_STATUS=$?
set +o pipefail
if [ "$RUSTUP_STATUS" -eq 0 ] && [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
  if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    report "rustup path: OK — $(rustc --version), $(cargo --version)"
    RUSTUP_OK=1
  else
    report "rustup path: installer returned 0 but rustc/cargo not on PATH"
  fi
else
  report "rustup path: FAILED status=$RUSTUP_STATUS (often TLS to sh.rustup.rs / static.rust-lang.org)"
fi

if [ "$RUSTUP_OK" -eq 1 ]; then
  return 0 2>/dev/null || exit 0
fi

report "BLOCKER: apt FAILED and rustup FAILED. No rustc in this environment."
report "Do not push unbuilt .rs changes as verified. Paste this log in the round doc."
return 1 2>/dev/null || exit 1
