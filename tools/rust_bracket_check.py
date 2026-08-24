#!/usr/bin/env python3
"""Literal-aware bracket balance checker for Rust sources.

This exists because parts of this project get edited in environments where no
Rust toolchain is available (no cargo, no rustc, and the toolchain hosts are
unreachable). In that situation the usual safety net -- `cargo check` -- is
gone, and the easiest way to ship a broken tree is an edit that drops or
doubles a brace.

This is emphatically NOT a substitute for compiling. It catches exactly one
class of error: unbalanced `()`, `[]`, `{}`. It knows nothing about types,
borrows, names, or syntax. A file can pass this and still fail to build in a
hundred ways. Treat a pass as "the edit did not obviously corrupt the block
structure", nothing more.

What makes it non-trivial is that a naive counter is wrong on real Rust: a
brace inside a string, char, or comment must not count. So the scanner strips
those first, handling the cases that actually appear in this codebase:

  - line comments (`//`) and *nested* block comments (`/* /* */ */`), which
    Rust allows and C does not;
  - raw strings with arbitrary hash counts (`r"..."`, `r#"..."#`);
  - normal strings with backslash escapes;
  - char literals, including `'\\''`, while not mistaking a lifetime
    (`&'a str`, `'files: for ...`) for the start of one.

That last case is the subtle one: `'a` is a lifetime, `'a'` is a char. The
scanner only consumes a char literal when the quote actually closes in the
shape of a char literal, otherwise it treats the tick as ordinary text.

Usage:
    python3 tools/rust_bracket_check.py FILE [FILE ...]
    python3 tools/rust_bracket_check.py --all      # every .rs tracked in repo

Exit status is 0 when every file balances, 1 otherwise, so it can be used as
a cheap pre-commit gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CHAR_LITERAL = re.compile(r"'(\\.|[^\\'])'")
OPENERS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = {")": "(", "]": "[", "}": "{"}


def strip_literals_and_comments(src: str) -> str:
    """Blank out comments and literals, preserving newlines for line numbers.

    Everything removed is replaced by a space (and newlines are kept) so that
    positions reported later still line up with the original file.
    """
    out: list[str] = []
    i, n = 0, len(src)

    def keep_newlines(text: str) -> str:
        return "".join(c if c == "\n" else " " for c in text)

    while i < n:
        c = src[i]

        # line comment
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(keep_newlines(src[i:j]))
            i = j

        # block comment, which nests in Rust
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            out.append(keep_newlines(src[i:j]))
            i = j

        # raw string: r"..." / r#"..."# / br#"..."#
        elif (c == "r" or (c == "b" and src.startswith("br", i))) and (
            m := re.match(r'b?r(#*)"', src[i:])
        ):
            hashes = m.group(1)
            term = '"' + hashes
            j = src.find(term, i + m.end())
            j = n if j < 0 else j + len(term)
            out.append(keep_newlines(src[i:j]))
            i = j

        # normal string
        elif c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                elif src[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            out.append(keep_newlines(src[i:j]))
            i = j

        # char literal vs lifetime: only consume if it closes like a char
        elif c == "'":
            m = CHAR_LITERAL.match(src, i)
            if m:
                out.append(keep_newlines(m.group(0)))
                i = m.end()
            else:
                out.append(" ")  # a lifetime tick; harmless
                i += 1

        else:
            out.append(c)
            i += 1

    return "".join(out)


def check(path: Path) -> list[str]:
    """Return a list of human-readable problems; empty means balanced."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"{path}: cannot read: {exc}"]

    cleaned = strip_literals_and_comments(src)
    stack: list[tuple[str, int, int]] = []
    line = col = 1
    problems: list[str] = []

    for ch in cleaned:
        if ch == "\n":
            line += 1
            col = 1
            continue
        col += 1
        if ch in OPENERS:
            stack.append((ch, line, col))
        elif ch in CLOSERS:
            if not stack:
                problems.append(f"{path}:{line}:{col}: unexpected closing '{ch}'")
                return problems
            top, oline, ocol = stack[-1]
            if top != CLOSERS[ch]:
                problems.append(
                    f"{path}:{line}:{col}: '{ch}' closes '{top}' "
                    f"opened at {oline}:{ocol}"
                )
                return problems
            stack.pop()

    for ch, oline, ocol in reversed(stack):
        problems.append(f"{path}:{oline}:{ocol}: unclosed '{ch}'")
    return problems


def tracked_rust_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.rs"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"cannot list tracked files: {exc}", file=sys.stderr)
        return []
    return [root / line for line in out.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument(
        "--all", action="store_true", help="check every .rs file tracked by git"
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="only report failures")
    args = ap.parse_args()

    paths = tracked_rust_files() if args.all else args.files
    if not paths:
        ap.error("no files given; pass paths or --all")

    failed = 0
    for p in paths:
        problems = check(p)
        if problems:
            failed += 1
            for line in problems:
                print(line)
        elif not args.quiet:
            print(f"{p}: balanced")

    total = len(paths)
    if failed:
        print(f"\n{failed}/{total} file(s) unbalanced", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\nall {total} file(s) balanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
