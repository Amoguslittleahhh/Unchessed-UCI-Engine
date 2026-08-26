#!/usr/bin/env python3
"""Tests for tools/training_blocks.py and the committed data/training blocks.

Two layers:
  * manifest consistency against the committed blocks (hashes, sizes,
    game counts, band invariants) — the "reliable" claim, made checkable;
  * tool behavior on synthetic PGNs — including the regression that
    python-chess 1.x silently drops illegal SAN (logging a warning
    instead of raising), which a naive "parse succeeded == legal"
    validator would count as legal.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from training_blocks import (  # noqa: E402
    BANDS,
    REPO_ROOT,
    _parse_one_game,
    band_for,
    game_tags,
    split_pgn_games,
)

DATA = REPO / "data" / "training"
MANIFEST = DATA / "manifest.json"

GAME_T = '''[Event "t"]
[Site "t"]
[Date "2020.01.01"]
[White "{w}"]
[Black "{b}"]
[WhiteElo "{we}"]
[BlackElo "{be}"]
[Result "{res}"]

{moves}
'''


def make_game(we: str, be: str, res: str, moves: str, white: str = "W", black: str = "B") -> str:
    return GAME_T.format(w=white, b=black, we=we, be=be, res=res, moves=moves)


# ---------------------------------------------------------------------------
# Tool behavior (synthetic)
# ---------------------------------------------------------------------------

class TestBandFor:
    def test_boundaries(self):
        assert band_for(1400, 1400) == "elo-1400-1700"  # mean 1400 -> upper band
        assert band_for(1399, 1399) == "elo-0000-1400"
        assert band_for(100, 100) == "elo-0000-1400"
        assert band_for(3500, 3500) == "elo-2600-3500"  # inclusive top
        assert band_for(3499, 3500) == "elo-2600-3500"

    def test_rejects_bad_ratings(self):
        assert band_for(0, 1500) is None
        assert band_for(99, 1500) is None
        assert band_for(21244, 1500) is None  # the mirror's garbage values
        assert band_for(1500, 0) is None


class TestSplitter:
    def test_split_and_verbatim(self, tmp_path):
        legal = make_game("1800", "1820", "1-0", "1. e4 e5 2. Nf3 Nc6 *")
        unrated = make_game("0", "0", "1-0", "1. d4 d5 *")
        unfinished = make_game("1900", "1850", "*", "1. e4 e5 *")
        src = tmp_path / "in.pgn"
        src.write_text(legal + "\n" + unrated + "\n" + unfinished)
        out = tmp_path / "out"
        r = subprocess.run(
            [sys.executable, str(TOOLS / "training_blocks.py"), "split", str(src), "--prefix", str(out),
             "--keep-quarantine"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["total_games"] == 3
        assert d["by_band"] == {"elo-1700-2000": 1}
        assert d["quarantined"] == 2
        band = out / "elo-1700-2000.pgn"
        assert band.read_text() == legal, "banded games must be copied verbatim"
        q = out / "_quarantine.pgn"
        qt = q.read_text()
        assert "d4 d5" in qt and "1. e4 e5 *" in qt and "1. e4 e5 2." not in qt

    def test_deterministic(self, tmp_path):
        src = tmp_path / "in.pgn"
        games = "\n".join(make_game("1500", str(1500 + i), "1-0", "1. e4 c5 *") for i in range(20))
        src.write_text(games)
        outs = []
        for i in range(2):
            out = tmp_path / f"out{i}"
            subprocess.run(
                [sys.executable, str(TOOLS / "training_blocks.py"), "split", str(src), "--prefix", str(out)],
                capture_output=True, timeout=120,
            )
            outs.append(hashlib.sha256((out / "elo-1400-1700.pgn").read_bytes()).hexdigest())
        assert outs[0] == outs[1]


class TestIllegalSanDetection:
    """python-chess 1.x silently drops illegal SAN and logs a warning; a
    validator that treats 'parse returned' as 'legal' overcounts. This is
    the regression for that bug."""

    def test_illegal_san_is_detected_not_silent(self):
        text = make_game("1800", "1820", "1-0", "1. e4 b1 2. Nf3 *")
        r = _parse_one_game(text)
        assert r == "illegal"

    def test_legal_game_passes(self):
        text = make_game("1800", "1820", "1-0", "1. e4 e5 2. Nf3 Nc6 3. Bb5 *")
        r = _parse_one_game(text)
        assert r is not None and r != "illegal" and r != "unparseable"

    def test_validate_counts_illegal(self, tmp_path):
        src = tmp_path / "in.pgn"
        legal = "\n".join(make_game("1800", "1820", "1-0", "1. e4 e5 2. Nf3 Nc6 *") for _ in range(10))
        bad = make_game("1800", "1820", "1-0", "1. e4 b1 2. Nf3 *")
        src.write_text(legal + "\n" + bad + "\n")
        r = subprocess.run(
            [sys.executable, str(TOOLS / "training_blocks.py"), "validate", str(src), "--sample", "11", "--seed", "1"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0
        d = json.loads(r.stdout)
        assert d["sample_illegal"] >= 1
        assert d["sample_legal"] + d["sample_illegal"] == d["sample_size"]

    def test_clean_drops_illegal(self, tmp_path):
        src = tmp_path / "in.pgn"
        legal = make_game("1800", "1820", "1-0", "1. e4 e5 2. Nf3 Nc6 *")
        bad = make_game("1800", "1820", "1-0", "1. e4 b1 2. Nf3 *")
        src.write_text(legal + "\n" + bad + "\n")
        r = subprocess.run(
            [sys.executable, str(TOOLS / "training_blocks.py"), "clean", str(src)],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0
        d = json.loads(r.stdout)
        assert d["games_total"] == 2 and d["games_kept"] == 1 and d["illegal"] == 1
        assert "1. e4 e5 2. Nf3 Nc6" in src.read_text()
        assert "b1" not in src.read_text()


# ---------------------------------------------------------------------------
# Committed blocks
# ---------------------------------------------------------------------------

class TestCommittedBlocks:
    def test_manifest_matches_files(self):
        manifest = json.loads(MANIFEST.read_text())
        assert manifest["source"]["commit"].startswith("ed88abd")
        for b in manifest["blocks"]:
            p = REPO / b["path"]
            assert p.is_file(), b["path"]
            data = p.read_bytes()
            assert len(data) == b["bytes"], b["path"]
            assert hashlib.sha256(data).hexdigest() == b["sha256"], b["path"]
            assert b["games"] >= 1
            assert b["full_move_validated"] is True

    def test_lichess_blocks_respect_band_invariant(self):
        """Every game in a lichess band file has its mean rating in-band
        (header-level; the move-legality half was done by `clean`)."""
        manifest = json.loads(MANIFEST.read_text())
        band_of = {name: (lo, hi) for lo, hi, name in BANDS}
        for b in manifest["blocks"]:
            if b["tier"] != "lichess":
                continue
            lo, hi = band_of[b["band_or_note"]]
            p = REPO / b["path"]
            n = 0
            for g in split_pgn_games(p.read_text(encoding="utf-8", errors="replace")):
                t = game_tags(g)
                w, bl = int(t["WhiteElo"]), int(t["BlackElo"])
                m = (w + bl) / 2
                assert lo <= m < hi or (hi == 3500 and m <= hi), (b["path"], w, bl)
                n += 1
            assert n == b["games"], b["path"]

    def test_verify_subcommand_passes(self):
        r = subprocess.run(
            [sys.executable, str(TOOLS / "training_blocks.py"), "verify"],
            capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "0 drift(s)" in r.stdout

    def test_total_game_count(self):
        manifest = json.loads(MANIFEST.read_text())
        assert sum(b["games"] for b in manifest["blocks"]) == 71987


class TestCli:
    def test_help_runs_standalone(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "training_blocks.py"), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert out.returncode == 0
        assert "band" in out.stdout.lower()

    def test_only_declared_dependencies(self):
        import ast

        src = (TOOLS / "training_blocks.py").read_text()
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        assert mods == {
            "__future__", "argparse", "hashlib", "json", "random",
            "re", "shutil", "subprocess", "sys", "pathlib", "io", "logging", "chess",
        }, mods
