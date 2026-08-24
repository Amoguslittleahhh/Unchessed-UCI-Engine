# Development environment

## Python tooling

Everything under `tools/` needs three packages. From a fresh clone:

```sh
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements-dev.txt
.venv/bin/python -m pytest tools/ -q
```

Measured from an empty venv: **4.2 s to install, 0.9 s to run the suite**
(31 passed, 6 skipped). `torch` is deliberately excluded — see the comments in
`tools/requirements-dev.txt`.

## Why dependencies are not vendored into the repo

This was proposed and measured, then rejected. Recording the numbers so the
question does not get reopened from scratch:

| Option | Cost | Benefit |
|---|---|---|
| `pip install` from index | 4.2 s, one command | always correct for the local platform |
| Commit wheels to git | ~23 MB in history **forever** | saves ~4 s |

23 MB of platform-specific binaries permanently in history, to save four
seconds, is a bad trade. Worse, it does not even work reliably: the `numpy`
wheel is built for a specific interpreter version, architecture, and libc, so
a committed wheel is wrong for anyone on a different setup — the repo would
be shipping a binary that only helps the machine it was captured on, while
everyone else pays the clone cost.

Git also stores every version ever committed. Refreshing vendored wheels
would add another ~23 MB each time rather than replacing them.

The general rule: **vendor source you wrote, not binaries you can re-fetch.**
A lockfile or requirements file reproduces an environment far more cheaply
than the artifacts themselves.

## When there is no Rust toolchain

Some environments have no `cargo`/`rustc` and cannot obtain one — the
toolchain hosts (`sh.rustup.rs`, `static.rust-lang.org`, `crates.io`) are
unreachable, and there is no toolchain on PyPI. `maturin` and
`setuptools-rust` exist there but only *invoke* an existing `rustc`; they do
not ship one. Distro packages are likewise unavailable when the apt sources
are empty.

Disk space is not the blocker, and freeing it does not help.

In that situation `cargo test --workspace --release` cannot run, and any Rust
change **must be disclosed as unverified**. The available fallbacks are:

1. **Bracket balance** — `python3 tools/rust_bracket_check.py --all`.
   Catches unbalanced `()`/`[]`/`{}`, the most likely way an edit corrupts a
   file. It is literal-aware (skips strings, char literals, raw strings, and
   nested block comments) but knows nothing about types or names. **A pass
   means "not obviously corrupt", not "compiles".**
2. **Symbol review** — for a new struct literal, check every field name and
   order against the struct declaration; for new calls, confirm each
   referenced symbol is in scope and imported.

Neither substitutes for a build. Compile before trusting the result.

## Ephemeral scratch space

`/tmp` does not survive between sessions here: virtualenvs, scratch C
benchmarks, and downloaded corpora all disappear. Anything worth keeping
belongs in the repo and committed — that is why the bracket checker now lives
in `tools/` instead of being rewritten from memory each time.

See also `docs/workspace-reset-recovery.md` for the related failure where the
git checkout itself reverts to an older commit.
