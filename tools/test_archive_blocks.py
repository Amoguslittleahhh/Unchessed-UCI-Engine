#!/usr/bin/env python3
"""Tests for tools/archive_blocks.py and the committed data/archive/.

Covers: the verify subcommand against manifest.json, layout-table
integrity, manifest internal consistency (totals, eras, years, rated
counts), a sampled move-legality re-check on representative files,
--help, and the import scan. No network or staged sources required —
it runs against the committed archive.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
TOOL = TOOLS / "archive_blocks.py"
ARCHIVE = REPO_ROOT / "data" / "archive"
MANIFEST = ARCHIVE / "manifest.json"
LAYOUT = TOOLS / "archive_layout.json"
sys.path.insert(0, str(TOOLS))

import archive_blocks as ab  # noqa: E402
from training_blocks import _parse_one_game, split_pgn_games  # noqa: E402

ERA_BOUNDS = {name: (lo, hi) for name, lo, hi in ab.ERA_BUCKETS}


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)


def test_verify_subcommand():
    r = run_cli("verify")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "0 drift(s)" in r.stdout
    assert "294 blocks checked" in r.stdout


def test_layout_integrity():
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    assert len(layout) >= 284
    seen_dests: set[str] = set()
    for key, entry in layout.items():
        src, repo_path = key.split(":", 1)
        assert src in ab.SOURCES, f"unknown source key {src!r} in {key}"
        assert entry.get("file"), f"{key} has no file name"
        split_era = "split_by_era" in entry.get("note", "")
        if not split_era:
            valid = set(ERA_BOUNDS) | {"womens", "correspondence",
                                       "world-championships"}
            assert entry.get("dir") in valid, f"{key} has bad dir {entry.get('dir')!r}"
            dest = f"{entry['dir']}/{entry['file']}"
            assert dest not in seen_dests, f"duplicate dest {dest}"
            seen_dests.add(dest)
        else:
            assert not entry.get("dir"), f"split entry {key} must not pin a dir"
    # every source file the layout references exists in a fresh fetch stage
    # is checked by cmd_fetch/build; here just assert no obviously broken
    # repo paths (must start with a dir or a file name, no leading slash)
    for key in layout:
        _, repo_path = key.split(":", 1)
        assert repo_path and not repo_path.startswith("/"), key


def test_manifest_consistency():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    t = m["totals"]
    assert t["files"] == len(m["blocks"])
    assert t["files"] == len(list(ARCHIVE.rglob("*.pgn")))
    assert t["bytes"] == sum(b["bytes"] for b in m["blocks"])
    assert t["games"] == sum(b["games"] for b in m["blocks"])
    assert t["illegal_games_dropped"] == sum(
        b["illegal_games_dropped"] for b in m["blocks"])
    assert t["games"] == sum(b["games_total_before_clean"] for b in m["blocks"]) \
        - t["illegal_games_dropped"]
    assert t["cross_file_duplicate_games"] >= 0
    for b in m["blocks"]:
        p = REPO_ROOT / b["path"]
        assert p.exists(), b["path"]
        assert p.stat().st_size == b["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == b["sha256"]
        assert b["theme"] in ERA_BOUNDS or b["theme"] in (
            "womens", "correspondence", "world-championships")
        if b["years"]:
            assert b["years"][0] <= b["years"][1]
            assert b["years"][0] >= 1400, f"implausible year in {b['path']}"
        assert 0 <= b["rated_games"] <= b["games"]
        # era-pinned files: the majority year sits in the era dir, so at
        # least one game's year reaches the era's lower bound
        if b["theme"] in ERA_BOUNDS and b["years"]:
            lo, _hi = ERA_BOUNDS[b["theme"]]
            assert b["years"][1] >= lo, f"era mismatch in {b['path']}: {b['years']}"


def _sampled_legal(path: Path, n: int = 30) -> int:
    events: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.append(record)

    lg = logging.getLogger("chess.pgn")
    old_level, old_propagate = lg.level, lg.propagate
    cap = _Cap()
    lg.addHandler(cap)
    lg.setLevel(logging.WARNING)
    lg.propagate = False
    games = split_pgn_games(path.read_text(encoding="utf-8",
                                            errors="replace"))[:n]
    ok = 0
    try:
        for g in games:
            r = _parse_one_game(g)
            assert r is not None and r != "illegal" and r != "unparseable", \
                f"illegal game in committed {path.name}"
            ok += 1
    finally:
        lg.removeHandler(cap)
        lg.setLevel(old_level)
        lg.propagate = old_propagate
    assert not [e for e in events if e.levelno >= logging.WARNING]
    return ok


def test_sampled_legality_on_representative_files():
    picks = [
        "classics-1834-1899/de-la-bourdonnais-mcdonnell-paris-1834.pgn",
        "world-championships/london-1851-first-world-championship.pgn",
        "1946-1970/nostalgia-1941-1960.pgn",
        "1971-1999/megadatabase-2600-1970-1998.pgn",
        "womens/polgar-judith.pgn",
        "2000-plus/2009_wtcc.pgn",
    ]
    for rel in picks:
        p = ARCHIVE / rel
        assert p.exists(), rel
        _sampled_legal(p)
    # and the FEN+move round-trip on one row of the 1834 match
    g0 = split_pgn_games((ARCHIVE /
        "classics-1834-1899/de-la-bourdonnais-mcdonnell-paris-1834.pgn")
        .read_text(errors="replace"))[0]
    game = chess.pgn.read_game(io.StringIO(g0))
    board = chess.Board()
    for mv in game.mainline_moves():
        assert mv in board.legal_moves
        board.push(mv)


def test_help_and_import_scan():
    out = run_cli("--help")
    assert out.returncode == 0 and "archive" in out.stdout.lower()
    src = TOOL.read_text(encoding="utf-8")
    for name in re.findall(r"^import (\w+)$", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"
    for name in re.findall(r"^from (\w+) import", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
