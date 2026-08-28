#!/usr/bin/env python3
"""Tests for tools/maia3_cloud_selfplay/generate.py (cloud scale-out).

Runnable in the sandbox (no cloud needed):
  * elo-plan determinism and the game-0 plan anchor against the
    committed data/selfplay/ reference set (seed 42),
  * per-game substream independence/determinism,
  * mixed-pool plan determinism, per-engine elo ranges, pool/elo
    bound filtering,
  * single-engine plan stream == legacy plan_elo stream (backward
    compatibility of the maia3-only configuration),
  * strength-ladder monotonicity (lc0 movetime, rubichess NPS cap),
  * model anchor outputs (startpos top-1 at 1500/1500 = 0.6165,
    at 200/3000 = 0.1666, measured calibration anchors),
  * a real 4-worker end-to-end mini-run (8 games, maia3-only) through
    generate -> manifest -> validate -> calibration,
  * an end-to-end mixed-engine mini-run (maia3 + RubiChess, when the
    scratch-built RubiChess binary is available),
  * --help / import scan.

Model-dependent tests skip cleanly when the ONNX mirror is not staged
(venv + `tools/selfplay_elo_mixer.py fetch-model`). Mixed-engine tests
skip when the scratch engine binary is not built (fetch-engines).
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
RC_BINARY = Path("/tmp/engines/src/rc/src/RubiChess")
needs_model = pytest.mark.skipif(not MODEL.exists(),
                                 reason="ONNX mirror not staged")
needs_rc = pytest.mark.skipif(not RC_BINARY.exists(),
                              reason="RubiChess scratch build not present")

NEW_ROW_KEYS = {"game", "elo_white", "elo_black", "fen", "move_uci",
                "move_ply", "side", "elo_self", "elo_oppo", "engine",
                "elo_quality", "top1_prob", "ldw"}


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


def test_mixed_plan_deterministic_and_ranges():
    pool = [generate.PROFILE_BY_ID[i]
            for i in ("maia3", "stockfish", "lc0", "rubichess")]
    a = generate.plan_games(pool, 2000, 99, 100, 3200)
    b = generate.plan_games(pool, 2000, 99, 100, 3200)
    assert (a == b).all()
    c = generate.plan_games(pool, 2000, 100, 100, 3200)
    assert not (a == c).all()
    assert a.shape == (2000, 4)
    ids = [p.id for p in pool]
    seen = set()
    for row in a:
        for col, elo_col in ((0, 1), (2, 3)):
            e = int(row[col])
            seen.add(ids[e])
            p = pool[e]
            assert int(row[elo_col]) >= p.elo_min
            assert int(row[elo_col]) <= p.elo_max
    assert seen == set(ids)  # all four engines appear


def test_mixed_plan_respects_elo_bounds():
    pool = [generate.PROFILE_BY_ID[i]
            for i in ("maia3", "stockfish", "lc0")]
    plan = generate.plan_games(pool, 1000, 7, 200, 1500)
    for row in plan:
        for col, elo_col in ((0, 1), (2, 3)):
            p = pool[int(row[col])]
            lo, hi = generate.effective_range(p, 200, 1500)
            assert lo <= int(row[elo_col]) <= hi
    # stockfish's native floor 1320 must be respected even though the
    # user asked from 200
    sf = [r for r in plan if pool[int(r[0])].id == "stockfish"]
    assert all(int(r[1]) >= 1320 for r in sf)
    sf2 = [r for r in plan if pool[int(r[2])].id == "stockfish"]
    assert all(int(r[3]) <= 1500 for r in sf2)


def test_mixed_plan_rejects_empty_overlap():
    pool = [generate.PROFILE_BY_ID["stockfish"]]
    with pytest.raises(SystemExit):
        generate.plan_games(pool, 10, 1, 100, 1300)


def test_single_engine_plan_matches_legacy_stream():
    """The maia3-only configuration must keep the original plan_elo
    stream byte-for-byte (documented backward compatibility)."""
    pool = [generate.PROFILE_BY_ID["maia3"]]
    plan = generate.plan_games(pool, 500, 42, 100, 3200)
    legacy = generate.plan_elo(100, 3200, 500, 42)
    assert (plan[:, 0] == 0).all() and (plan[:, 2] == 0).all()
    assert (plan[:, 1] == legacy[:, 0]).all()
    assert (plan[:, 3] == legacy[:, 1]).all()


def test_ladders_monotone():
    elos = list(range(600, 3201, 50))
    lc0 = [generate.ladder_value(generate.LC0_MOVETIME_LADDER, e)
           for e in elos]
    assert lc0 == sorted(lc0)                      # more time = stronger
    assert generate.ladder_value(
        generate.LC0_MOVETIME_LADDER, 500) == lc0[0]  # clamped low
    assert generate.ladder_value(
        generate.LC0_MOVETIME_LADDER, 3200) == 500
    rc = [generate.ladder_value(generate.RC_NPS_LADDER, e) for e in elos]
    # higher elo -> higher (or uncapped) NPS cap = stronger
    for lo, hi in zip(rc, rc[1:]):
        assert hi >= lo
    assert rc[-1] == 2147483647  # effectively uncapped at the top
    assert generate.ladder_value(generate.RC_NPS_LADDER, 500) == rc[0]


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
    """4 workers x 2 games, seed 777, maia3-only: generate ->
    validate -> calibration. Same plan stream as the original 2M
    generator for this seed."""
    out = tmp_path / "mini"
    r = subprocess.run(
        [sys.executable, str(GEN), "generate",
         "--model", str(MODEL), "--engines", "maia3",
         "--out", str(out),
         "--games", "8", "--seed", "777", "--workers", "4"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    m = json.loads((out / "manifest.json").read_text())
    assert m["games"] == 8
    assert m["seed"] == 777
    assert m["mode"] == "maia3-only"
    assert len(m["shards"]) == 4
    assert sum(s["games"] for s in m["shards"]) == 8
    assert m["label_rows"] > 0
    for s in m["shards"]:
        pgn = REPO_ROOT / s["pgn"] if not s["pgn"].startswith("/") \
            else Path(s["pgn"])
        assert pgn.exists()
    # labels schema: original keys + engine/elo_quality
    with (out / "labels" / "shard-00000.jsonl").open() as fh:
        row = json.loads(fh.readline())
    assert set(row) == NEW_ROW_KEYS
    assert row["engine"] == "maia3" and row["elo_quality"] == "calibrated"
    # plan consistency: shard 0's game elos match the legacy plan stream
    plan = generate.plan_elo(100, 3200, 8, 777)
    gid = row["game"]
    assert (row["elo_white"], row["elo_black"]) == \
        (int(plan[gid, 0]), int(plan[gid, 1]))
    # calibration written by the built-in validation pass
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["error_count"] == 0
    assert set(cal["engine_stats"]) == {"maia3"}
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


@needs_model
@needs_rc
def test_mixed_engine_mini_run(tmp_path):
    """maia3 + RubiChess, 8 games, 2 workers: the resident UCI engine
    pool, per-side engine/elo headers, null top1_prob on engine rows,
    engine_stats in the calibration report."""
    out = tmp_path / "mixed"
    eng_dir = tmp_path / "engines" / "rubichess"
    eng_dir.mkdir(parents=True)
    (eng_dir / "RubiChess").symlink_to(RC_BINARY)
    rc_net = Path("/tmp/engines/src/NN/nn-da9c99e92a-20260819.nnue")
    if rc_net.exists():
        # engines.json so the generator passes --nnue (NNUE build);
        # without it the generator falls back to Use_NNUE=false
        json_path = tmp_path / "engines" / "engines.json"
        json_path.write_text(json.dumps({
            "rubichess": {"binary": str(RC_BINARY),
                          "net": str(rc_net)},
        }))
    r = subprocess.run(
        [sys.executable, str(GEN), "generate",
         "--model", str(MODEL),
         "--engines", "maia3,rubichess",
         "--engines-dir", str(tmp_path / "engines"),
         "--out", str(out),
         "--games", "8", "--seed", "778", "--workers", "2"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    m = json.loads((out / "manifest.json").read_text())
    assert m["mode"] == "mixed-engine"
    assert m["games"] == 8
    ids = {e["id"] for e in m["engines"]}
    assert ids == {"maia3", "rubichess"}
    # every game has engine + quality headers, consistent with labels
    text = "".join((out / "pgn" / p).read_text()
                   for p in os.listdir(out / "pgn"))
    games = _read_all_pgn(text)
    assert len(games) == 8
    plan = generate.plan_games(
        [generate.PROFILE_BY_ID["maia3"],
         generate.PROFILE_BY_ID["rubichess"]], 8, 778, 100, 3200)
    for g in games:
        gid = int(g.headers["Round"])
        row = plan[gid]
        pw = g.headers["WhiteEngine"]
        pb = g.headers["BlackEngine"]
        assert pw in ("maia3", "rubichess")
        assert int(g.headers["WhiteElo"]) == int(row[1])
        assert int(g.headers["BlackElo"]) == int(row[3])
        assert g.headers["WhiteEloQuality"] in ("calibrated", "approximate")
        assert g.headers["WhiteEloQuality"] == \
            ("calibrated" if pw == "maia3" else "approximate")
        assert len(list(g.mainline_moves())) > 0
    # label rows: maia rows carry a top1, engine rows carry nulls
    rows = []
    for p in os.listdir(out / "labels"):
        with (out / "labels" / p).open() as fh:
            for line in fh:
                rows.append(json.loads(line))
    assert rows
    maia_rows = [x for x in rows if x["engine"] == "maia3"]
    rc_rows = [x for x in rows if x["engine"] == "rubichess"]
    if maia_rows:
        assert all(0 < x["top1_prob"] <= 1 for x in maia_rows)
        assert all(x["ldw"] is not None for x in maia_rows)
    if rc_rows:
        assert all(x["top1_prob"] is None for x in rc_rows)
        assert all(x["ldw"] is None for x in rc_rows)
        assert all(x["elo_quality"] == "approximate" for x in rc_rows)
    # calibration report aggregates per-engine stats
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["error_count"] == 0
    assert set(cal["engine_stats"]) == {"maia3", "rubichess"}
    assert cal["engine_stats"]["maia3"]["moves"] == len(maia_rows)
    assert cal["engine_stats"]["rubichess"]["moves"] == len(rc_rows)


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
            # honor @needs_model / @needs_rc in standalone mode
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
