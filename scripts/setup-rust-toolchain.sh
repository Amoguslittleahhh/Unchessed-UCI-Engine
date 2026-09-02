#!/bin/bash
# Idempotent Rust toolchain bootstrap for a fresh/ephemeral sandbox.
#
# Why this exists: multiple arena rounds (15, 18) pushed Rust changes
# that don't compile, because the sandbox had no rustc and the round's
# own doc just noted "(need rustc)" as a caveat instead of installing
# one. Arena's sandbox (Debian 12 KVM/E2B-style VM) filters outbound
# HTTPS: GitHub's HTML/API is reachable, but rustup.rs and the Debian
# package CDN often fail TLS. There is no single install path known to
# work through that filter yet, so this tries a few and reports exactly
# what happened with each -- silence is not an acceptable outcome here.
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

echo "no rustc found -- trying install paths in order, reporting each result:"

# Path 1: apt (Debian's own repos may resolve through a different route
# than the rustup.rs/CDN hosts that are known to fail here -- worth
# trying independently rather than assuming the same filter blocks it).
if command -v apt-get >/dev/null 2>&1; then
  echo "-- trying apt-get install rustc cargo --"
  if apt-get update -qq 2>&1 | tail -5 && apt-get install -y -qq rustc cargo 2>&1 | tail -5; then
    if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
      echo "apt install worked: $(rustc --version), $(cargo --version)"
      echo "note: Debian's apt rustc/cargo can lag the 'stable' channel that"
      echo "rust-toolchain.toml pins -- if cargo complains about the channel,"
      echo "that mismatch is expected; report the actual version back."
      return 0 2>/dev/null || exit 0
    fi
  fi
  echo "apt path did not produce a working rustc/cargo, moving on."
else
  echo "no apt-get on this system, skipping that path."
fi

# Path 2: rustup official installer.
echo "-- trying rustup (https://sh.rustup.rs) --"
if curl --proto '=https' --tlsv1.2 -sSf --connect-timeout 10 https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default 2>&1 | tail -20; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env" 2>/dev/null || true
  if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    echo "rustup install worked: $(rustc --version), $(cargo --version)"
    return 0 2>/dev/null || exit 0
  fi
fi
echo "rustup path did not produce a working rustc/cargo."

echo "ERROR: no install path worked in this sandbox." >&2
echo "This is a real, reportable blocker -- put the exact curl/apt error output" >&2
echo "from above in the round's doc instead of silently pushing unbuilt code." >&2
echo "If someone with access to this sandbox's network policy can add an allowed" >&2
echo "host (an internal mirror, or a proxy for static.rust-lang.org/rustup.rs)," >&2
echo "that is the actual fix; this script cannot work around a network it cannot reach." >&2
return 1 2>/dev/null || exit 1
