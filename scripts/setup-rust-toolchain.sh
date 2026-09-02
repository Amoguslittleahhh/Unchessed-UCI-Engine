#!/bin/bash
# Idempotent Rust toolchain bootstrap for a fresh/ephemeral sandbox.
#
# Tries, in order:
#   1. rustc/cargo already on PATH
#   2. existing rustup at $HOME/.cargo/env
#   3. Debian/Ubuntu apt (rustc + cargo)
#   4. official rustup installer (https://sh.rustup.rs)
#   5. Homebrew on Linux, as a last resort (untested against arena's
#      actual sandbox as of this writing -- report back whether this one
#      works). Different infrastructure than the first two: its installer
#      lives at raw.githubusercontent.com (GitHub's own CDN, confirmed
#      reachable when github.com HTML/API is), and `brew install rust`
#      pulls a precompiled bottle from ghcr.io (GitHub Container
#      Registry) rather than static.rust-lang.org or the Debian archive.
#      Heavier than the other two (clones Homebrew's git repo, needs
#      build-essential/curl/git), so it's the fallback of last resort,
#      not the first thing to reach for.
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

BREW_OK=0
report "brew path: installing Homebrew (raw.githubusercontent.com), then 'brew install rust'..."
if command -v brew >/dev/null 2>&1; then
  report "brew path: brew already present, skipping install step"
elif command -v git >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
  NONINTERACTIVE=1 bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  BREW_INSTALL_STATUS=$?
  if [ "$BREW_INSTALL_STATUS" -ne 0 ]; then
    report "brew path: FAILED status=$BREW_INSTALL_STATUS installing Homebrew itself (raw.githubusercontent.com unreachable, or missing build deps)"
  fi
  # Homebrew on Linux installs to /home/linuxbrew/.linuxbrew by default
  if [ -x /home/linuxbrew/.linuxbrew/bin/brew ]; then
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
  fi
else
  report "brew path: skipped (no git or no curl)"
fi

if command -v brew >/dev/null 2>&1; then
  if brew install rust; then
    if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
      report "brew path: OK — $(rustc --version), $(cargo --version)"
      BREW_OK=1
    else
      report "brew path: 'brew install rust' claimed success but rustc/cargo not on PATH"
    fi
  else
    report "brew path: FAILED ('brew install rust' — often ghcr.io unreachable, falls back to building from source and times out)"
  fi
fi

if [ "$BREW_OK" -eq 1 ]; then
  return 0 2>/dev/null || exit 0
fi

report "BLOCKER: apt, rustup, and brew all FAILED. No rustc in this environment."
report "Do not push unbuilt .rs changes as verified. Paste this log in the round doc."
return 1 2>/dev/null || exit 1
