#!/usr/bin/env python3
"""Tests for tools/build_elo_bands.py (real-human 100-3200 banding).

Covers: band_for boundary rules, a fixture count/build with verbatim
round-trip and cap behaviour, the --help/import scan, and the committed
data/training-elo/ set (manifest consistency, band-invariant on a
sampled per-game parse, legality spot-check).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import chess

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
TOOL = TOOLS / "build_elo_bands.py"
DATASET = REPO_ROOT / "data" / "training-elo"
sys.path.insert(0, str(TOOLS))

import build_elo_bands as beb  # noqa: E402
from training_blocks import _parse_one_game, split_pgn_games  # noqa: E402


def test_band_for_boundaries():
    assert beb.band_for(100, 100) == 100
    assert beb.band_for(100, 149) == 100            # mean 124.5
    assert beb.band_for(100, 150) == 100            # mean 125
    assert beb.band_for(140, 160) == 100            # mean 150
    assert beb.band_for(1000, 1099) == 1000
    assert beb.band_for(1000, 1100) == 1000         # mean 1050
    assert beb.band_for(3200, 3299) == 3200
    assert beb.band_for(3250, 3350) == 3300         # mean 3300 -> overflow
    assert beb.band_for(3400, 3499) == 3300         # clamped to overflow
    assert beb.band_for(99, 3500) is None           # below min
    assert beb.band_for(100, 3500) is None          # strict max
    assert beb.band_for(3500, 3500) is None


def _game(white_elo, black_elo, n=6) -> str:
    tags = (f'[Event "T"]\n[White "W{white_elo}"]\n[Black "B{black_elo}"]\n'
            f'[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n'
            '[Result "1-0"]\n\n')
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"][:n]
    text = tags
    for i, san in enumerate(moves):
        if i % 2 == 0:
            text += f"{i // 2 + 1}. "
        text += san + " "
    text += "1-0\n"
    return text


def test_fixture_count_and_build(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.pgn").write_text(
        _game(1500, 1600) + _game(2000, 2100) + _game(2000, 2100)
        + _game(50, 3000) + "[White \"X\"]\n[Result \"*\"]\n\n1. e4 e5 *\n")
    r = subprocess.run([sys.executable, str(TOOL), "count", "--source",
                        str(src)], capture_output=True, text=True,
                       cwd=REPO_ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout[r.stdout.index("{"):])
    assert report["per_band"]["1500"] == 1
    assert report["per_band"]["2000"] == 2
    assert report["banded_total"] == 3

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(TOOL), "build", "--source",
                        str(src), "--out", str(out), "--cap", "1"],
                       capture_output=True, text=True, cwd=REPO_ROOT,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["per_band_kept"]["2000"] == 1        # cap applied
    assert m["per_band_available"]["2000"] == 2   # counting continued
    assert m["kept_total"] == 2
    # verbatim round-trip: the kept game's bytes appear in the source
    kept = (out / "elo-1500.pgn").read_text()
    src_text = (src / "a.pgn").read_text()
    games = split_pgn_games(kept)
    assert len(games) == 1
    assert games[0] in src_text
    assert '[WhiteElo "1500"]' in kept


def test_committed_dataset_consistency():
    m = json.loads((DATASET / "manifest.json").read_text())
    blocks = m["blocks"]
    assert len(blocks) == 33
    assert m["totals"]["games"] == sum(b["games"] for b in blocks)
    assert 30000 <= m["totals"]["games"] <= 40000
    assert m["totals"]["illegal_games_dropped"] == sum(
        b["illegal_games_dropped"] for b in blocks)
    for b in blocks:
        p = REPO_ROOT / b["path"]
        assert p.exists()
        assert p.stat().st_size == b["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == b["sha256"]
        games = split_pgn_games(p.read_text(encoding="utf-8", errors="replace"))
        assert len(games) == b["games"]
        # band invariant on every game of the tail bands + a sample of
        # the largest bands (mean in [band, band+100))
        if b["band"] in (1000, 2000, 2800, 3300) or b["games"] <= 20:
            for g in games:
                w = int(re.search(r'^\[WhiteElo "(\d+)"', g, re.M).group(1))
                bl = int(re.search(r'^\[BlackElo "(\d+)"', g, re.M).group(1))
                mean = (w + bl) / 2
                lo, hi = b["band"], b["band"] + 100
                if b["band"] == 3300:
                    hi = 3500
                assert lo <= mean < hi, (b["band"], w, bl)


def test_sampled_legality_on_largest_band():
    p = DATASET / "elo-2000.pgn"
    games = split_pgn_games(p.read_text(encoding="utf-8", errors="replace"))
    ok = 0
    for g in games[:50]:
        r = _parse_one_game(g)
        assert r is not None and r != "illegal" and r != "unparseable", \
            "illegal game in committed dataset"
        ok += 1
    assert ok == 50


def test_help_and_import_scan():
    out = subprocess.run([sys.executable, str(TOOL), "--help"],
                         capture_output=True, text=True, cwd=REPO_ROOT,
                         timeout=60)
    assert out.returncode == 0 and "bands" in out.stdout.lower()
    src = TOOL.read_text(encoding="utf-8")
    for name in re.findall(r"^import (\w+)$", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"
    for name in re.findall(r"^from (\w+) import", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"


if __name__ == "__main__":
    import tempfile
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
