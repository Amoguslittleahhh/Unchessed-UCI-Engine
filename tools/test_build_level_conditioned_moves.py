#!/usr/bin/env python3
"""Tests for tools/build_level_conditioned_moves.py (Maia-style labels).

Covers: the both-players window rule, bullet/unrated/no-window skips,
first/last-move trimming, active/opponent elo alternation, determinism,
the sample selection, --help, the import scan, and a real-block smoke
run (small block: all rows satisfy the window invariant).
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
TOOL = REPO_ROOT / "tools" / "build_level_conditioned_moves.py"
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_level_conditioned_moves as blm  # noqa: E402

SAN_60 = [
    "h4", "g5", "Rh2", "f6", "f4", "g4", "c3", "c6", "b3", "b6",
    "d4", "Ba6", "f5", "Qc8", "Bh6", "Kf7", "e3", "Qe8", "Qxg4", "d5",
    "Qf4", "Bb5", "Qg5", "Bxh6", "a3", "a5", "Rh1", "Bd3", "Kd2", "Qd7",
    "Kd1", "e6", "Rh2", "Qa7", "Ke1", "Bxf5", "c4", "Be4", "Ne2", "Nd7",
    "Qxg8+", "Kxg8", "Ra2", "Kf8", "b4", "e5", "Ra1", "Qa6", "Nbc3", "Kf7",
    "Nb1", "Bd3", "Kf2", "Ke8", "Kg1", "c5", "bxc5", "Ke7", "Nf4", "Bf8",
]


def write_game(path: Path, *, white: str, black: str, white_elo: str,
               black_elo: str, event: str = "Test", result: str = "1-0",
               san: list[str] | None = None, time_control: str = "") -> Path:
    # Independent replay: raises on any illegal move in the fixture.
    board = chess.Board()
    for s in (san if san is not None else SAN_60):
        board.push_san(s)
    tags = [
        ("Event", event), ("Site", "?"), ("Date", "2022.08.02"), ("Round", "1"),
        ("White", white), ("Black", black), ("Result", result),
        ("WhiteElo", white_elo), ("BlackElo", black_elo),
    ]
    if time_control:
        tags.append(("TimeControl", time_control))
    moves = san if san is not None else SAN_60
    text = "".join(f'[{k} "{v}"]\n' for k, v in tags) + "\n"
    for i, s in enumerate(moves, start=1):
        if i % 2 == 1:
            text += f"{(i + 1) // 2}. "
        text += f"{s} "
    text += result + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)


def collect(path, windows, tf=0, tl=0):
    rows, skips = [], {k: 0 for k in blm.SKIP_KEYS}
    for row, reason in blm.iter_rows([path], windows, tf, tl):
        if reason == "game":
            continue
        if reason:
            skips[reason] += 1
        else:
            rows.append(row)
    return rows, skips


def test_fixture_window_and_trimming(tmp_path):
    p = write_game(tmp_path / "f.pgn", white="W", black="B",
                   white_elo="1500", black_elo="1550")
    rows, skips = collect(p, (1500,), 10, 10)
    # 60 plies, trim 10 each side -> 40 rows (plies 11..50)
    assert all(v == 0 for v in skips.values())
    assert len(rows) == 40
    assert [r["move_ply"] for r in rows] == list(range(11, 51))
    # active/opponent alternation
    for r in rows:
        assert r["level_window"] == 1500
        if r["side"] == "white":
            assert (r["active_elo"], r["opponent_elo"]) == (1500, 1550)
        else:
            assert (r["active_elo"], r["opponent_elo"]) == (1550, 1500)
    # FEN + move legality: replay from the row
    for r in rows[:3]:
        b = chess.Board(r["fen"])
        mv = chess.Move.from_uci(r["move_uci"])
        assert mv in b.legal_moves, r


def test_fixture_both_players_rule(tmp_path):
    # mean 1600 (would fall in window 1600) but one player outside -> skipped
    p = write_game(tmp_path / "f.pgn", white="W", black="B",
                   white_elo="1500", black_elo="1700")
    rows, skips = collect(p, (1500, 1600, 1700), 0, 0)
    assert rows == []
    assert skips["no_window"] == 1


def test_fixture_skips(tmp_path):
    bullet = write_game(tmp_path / "b.pgn", white="W", black="B",
                        white_elo="1500", black_elo="1550", event="Bullet Mayhem")
    unrated = write_game(tmp_path / "u.pgn", white="W", black="B",
                         white_elo="", black_elo="")
    stalemate = write_game(tmp_path / "s.pgn", white="W", black="B",
                           white_elo="1500", black_elo="1550", result="*")
    rows, skips = collect(bullet, (1500,), 0, 0)
    assert rows == []
    assert skips["bullet"] == 1 and skips["unparseable_or_desync"] == 0
    rows, skips = collect(unrated, (1500,), 0, 0)
    assert rows == []
    assert skips["unrated"] == 1
    rows, skips = collect(stalemate, (1500,), 0, 0)
    assert rows == []
    assert skips["other_result"] == 1


def test_fixture_determinism_and_sample(tmp_path):
    p = write_game(tmp_path / "f.pgn", white="W", black="B",
                   white_elo="1500", black_elo="1550")
    r1, _ = collect(p, (1500,), 10, 10)
    r2, _ = collect(p, (1500,), 10, 10)
    h = lambda rs: hashlib.sha256(
        json.dumps(rs, separators=(",", ":")).encode()).hexdigest()
    assert h(r1) == h(r2)
    # even stride over 40 rows, keep 2 -> indices 0 and 20 -> plies 11 and 31
    idx = blm.even_stride_indices(40, 2)
    assert idx == {0, 20}
    assert {r["move_ply"] for r in [r1[i] for i in sorted(idx)]} == {11, 31}


def test_cli_help_and_import_scan():
    out = run_cli("--help")
    assert out.returncode == 0 and "level-conditioned" in out.stdout.lower()
    src = TOOL.read_text(encoding="utf-8")
    for name in re.findall(r"^import (\w+)$", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"
    for name in re.findall(r"^from (\w+) import", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"


def test_cli_real_block_smoke(tmp_path):
    block = REPO_ROOT / "data" / "training" / "lichess-2022-10-05" / "elo-0000-1400.pgn"
    out_file = tmp_path / "out.jsonl"
    prof_file = tmp_path / "profile.json"
    out = run_cli("--blocks", str(block), "--out", str(out_file),
                  "--profile-out", str(prof_file), "--trim-first", "10",
                  "--trim-last", "10")
    assert out.returncode == 0, out.stderr
    profile = json.loads(prof_file.read_text(encoding="utf-8"))
    assert profile["rows"] == sum(profile["per_window"].values())
    assert profile["rows"] > 0
    assert profile["games_seen"] == 695
    # window invariant: every row's both elos inside its window; and the
    # FENs replay (the stream-parse desync bug would break exactly this)
    lines = out_file.read_text(encoding="utf-8").splitlines()
    n = 0
    for line in lines:
        r = json.loads(line)
        L = r["level_window"]
        assert L <= r["active_elo"] < L + 100
        assert L <= r["opponent_elo"] < L + 100
        n += 1
    assert n == profile["rows"]
    step = max(1, n // 50)
    for line in lines[::step][:50]:
        r = json.loads(line)
        b = chess.Board(r["fen"])
        mv = chess.Move.from_uci(r["move_uci"])
        assert mv in b.legal_moves, r
    # determinism of the sha
    out2 = tmp_path / "out2.jsonl"
    prof2 = tmp_path / "profile2.json"
    out = run_cli("--blocks", str(block), "--out", str(out2), "--profile-out", str(prof2))
    assert out.returncode == 0
    p2 = json.loads(prof2.read_text(encoding="utf-8"))
    assert p2["sha256_of_rows"] == profile["sha256_of_rows"]
    assert p2["rows"] == profile["rows"]


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
