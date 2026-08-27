#!/usr/bin/env python3
"""Tests for tools/maia3_cloud_selfplay/generate.py (cloud scale-out).

Runnable in the sandbox (no cloud needed):
  * elo-plan determinism and the game-0 plan anchor against the
    committed data/selfplay/ reference set (seed 42),
  * per-game substream independence/determinism,
  * model anchor outputs (startpos top-1 at 1500/1500 = 0.6165,
    at 200/3000 = 0.1666, measured calibration anchors),
  * a real 4-worker end-to-end mini-run (8 games) through
    generate -> manifest -> validate -> calibration,
  * --help / import scan.

Model-dependent tests skip cleanly when the ONNX mirror is not staged
(venv + `tools/selfplay_elo_mixer.py fetch-model`).
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
PKG = TOOLS / "maia3_cloud_selfplay"
GEN = PKG / "generate.py"
REFERENCE_PGN = REPO_ROOT / "data" / "selfplay" / "maia3-100-3200.pgn"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(TOOLS))

import generate  # noqa: E402

MODEL = Path("/tmp/smi/simple_maia3_inference/maia3_simplified.onnx")
needs_model = pytest.mark.skipif(not MODEL.exists(),
                                 reason="ONNX mirror not staged")


def _read_all_pgn(text: str):
    import chess.pgn
    stream = io.StringIO(text)
    games = []
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        games.append(g)
    return games


def test_plan_deterministic():
    a = generate.plan_elo(100, 3200, 500, 42)
    b = generate.plan_elo(100, 3200, 500, 42)
    assert (a == b).all()
    assert a.shape == (500, 2)
    assert a.min() >= 100 and a.max() <= 3200
    c = generate.plan_elo(100, 3200, 500, 43)
    assert not (a == c).all()


def test_plan_anchors_to_committed_reference():
    """Game 0 of the seed-42 plan must equal the committed reference
    set's first game (both streams are fresh Random(42) before their
    first two randints). Later games deliberately differ: the reference
    pipeline interleaved move sampling into the same stream, while this
    generator uses independent per-game substreams (documented)."""
    assert REFERENCE_PGN.exists()
    games = _read_all_pgn(REFERENCE_PGN.read_text())
    assert len(games) == 200
    plan = generate.plan_elo(100, 3200, 200, 42)
    w, b = int(games[0].headers["WhiteElo"]), int(games[0].headers["BlackElo"])
    assert (w, b) == (int(plan[0, 0]), int(plan[0, 1])), (w, b, plan[0].tolist())


def test_game_rng_independent_and_deterministic():
    r1 = generate.game_rng(1, 5)
    r2 = generate.game_rng(1, 5)
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]
    r3 = generate.game_rng(1, 6)
    assert [r3.random() for _ in range(5)] != [r2.random() for _ in range(5)]
    r4 = generate.game_rng(2, 5)
    assert [r4.random() for _ in range(5)] != [r2.random() for _ in range(5)]


@needs_model
def test_model_anchors():
    from selfplay_elo_mixer import Maia3
    m = Maia3(MODEL)
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    _p, _l, top1_1500 = m.probs(startpos, 1500, 1500)
    assert abs(top1_1500 - 0.6165) < 0.002, top1_1500
    _p, _l, top1_low = m.probs(startpos, 200, 3000)
    assert abs(top1_low - 0.1666) < 0.002, top1_low
    # conditioning is per-side and monotone-ish at the extremes
    _p, _l, top1_hi = m.probs(startpos, 3000, 200)
    assert top1_hi > top1_low


@needs_model
def test_end_to_end_mini_run(tmp_path):
    """4 workers x 2 games, seed 777: generate -> validate -> calibration."""
    out = tmp_path / "mini"
    r = subprocess.run(
        [sys.executable, str(GEN), "generate",
         "--model", str(MODEL), "--out", str(out),
         "--games", "8", "--seed", "777", "--workers", "4"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    m = json.loads((out / "manifest.json").read_text())
    assert m["games"] == 8
    assert m["seed"] == 777
    assert len(m["shards"]) == 4
    assert sum(s["games"] for s in m["shards"]) == 8
    assert m["label_rows"] > 0
    for s in m["shards"]:
        pgn = REPO_ROOT / s["pgn"] if not s["pgn"].startswith("/") \
            else Path(s["pgn"])
        assert pgn.exists()
    # labels schema matches the committed reference schema
    with (out / "labels" / "shard-00000.jsonl").open() as fh:
        row = json.loads(fh.readline())
    assert set(row) == {"game", "elo_white", "elo_black", "fen",
                        "move_uci", "move_ply", "side", "elo_self",
                        "elo_oppo", "top1_prob", "ldw"}
    # plan consistency: shard 0's game elos match the plan stream
    plan = generate.plan_elo(100, 3200, 8, 777)
    gid = row["game"]
    assert (row["elo_white"], row["elo_black"]) == \
        (int(plan[gid, 0]), int(plan[gid, 1]))
    # calibration written by the built-in validation pass
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["error_count"] == 0
    # 694 moves < 1000 -> the check reports None (insufficient data);
    # the low/high means must still show the conditioning direction
    assert cal["calibration"]["check_passed"] in (True, None)
    lo, hi = (cal["calibration"]["low_end_mean"],
              cal["calibration"]["high_end_mean"])
    if lo is not None and hi is not None:
        assert lo < hi
    # re-run validate standalone
    r = subprocess.run(
        [sys.executable, str(GEN), "validate", "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    assert "VALIDATION PASSED" in r.stdout


def test_help_and_import_scan():
    out = subprocess.run([sys.executable, str(GEN), "--help"],
                         capture_output=True, text=True, cwd=REPO_ROOT,
                         timeout=60)
    assert out.returncode == 0 and "maia-3" in out.stdout.lower()
    src = GEN.read_text(encoding="utf-8")
    import re
    for name in re.findall(r"^import (\w+)$", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"
    for name in re.findall(r"^from (\w+) import", src, re.M):
        assert __import__(name) is not None, f"unimportable: {name}"


if __name__ == "__main__":
    import tempfile
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            # honor @needs_model in standalone mode
            skipped = False
            for mark in getattr(fn, "pytestmark", []):
                mk = getattr(mark, "mark", mark)
                if getattr(mk, "condition", None):
                    print(f"SKIP {name} ({getattr(mk, 'reason', '')})")
                    skipped = True
                    break
            if skipped:
                continue
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
