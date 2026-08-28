#!/usr/bin/env python3
"""Tests for the move-prediction pretrain bridge + probe.

Hermetic (no model, no cloud):
  * bridge invariants: every target action is in its legal set (the
    double-mirror guard), the STM-normalization property (a
    black-to-move position encodes exactly like the mirrored
    white-to-move position, including the target action), dual-elo
    extraction from header ratings,
  * game-disjoint train/val split,
  * a numerical gradient check of the probe's backprop on a toy batch
    (analytical vs central-difference, float32 tolerance),
  * an end-to-end toy train -> report cycle (tiny model, few rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pretrain_move_dataset as bridge  # noqa: E402
import pretrain_move_predictor as pmp  # noqa: E402

START_W = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_B = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
CASTLE_W = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
PROMO_B = "7k/5P2/8/8/8/8/8/K7 b - - 0 2"
PROMO_W = "8/P7/8/8/8/8/8/K6k w - - 0 1"


def _row(fen, move_uci, ew=1500, eb=1700):
    return {"fen": fen, "move_uci": move_uci, "elo_self": ew,
            "elo_oppo": eb, "engine": "maia3", "quality": "calibrated",
            "game_id": 0, "ply": 1}


def test_target_always_in_legal_set():
    rows = [_row(START_W, "e2e4"), _row(START_B, "e7e5"),
            _row(CASTLE_W, "e1g1"), _row(PROMO_B, "h8h7"),
            _row(PROMO_W, "a7a8q")]
    arrays = bridge.build(rows)
    assert arrays["bad"] == 0
    for i in range(arrays["rows"]):
        legal = [int(a) for a in arrays["legal"][i, :int(
            arrays["legal_count"][i])]]
        assert int(arrays["action"][i]) in legal, i


def test_stm_normalization_mirrors_white_view():
    """Black-to-move startpos must encode exactly like the mirrored
    white-to-move startpos — same planes, and the mirrored move gets
    the same action code as its white-view counterpart."""
    a = bridge.build([_row(START_B, "e7e5")])
    b = bridge.build([_row(START_W, "e2e4")])
    assert (a["bitboards"][0] == b["bitboards"][0]).all()
    assert a["castling"][0] == b["castling"][0]
    assert int(a["action"][0]) == int(b["action"][0])
    # and the black-view legal set mirrors the white-view one
    la = set(int(x) for x in a["legal"][0, :int(a["legal_count"][0])])
    lb = set(int(x) for x in b["legal"][0, :int(b["legal_count"][0])])
    assert la == lb


def test_castling_and_promotion_encode():
    c = bridge.build([_row(CASTLE_W, "e1g1")])
    assert c["castling"][0] == 0b1111  # both sides have KQ
    legal = [int(x) for x in c["legal"][0, :int(c["legal_count"][0])]]
    assert int(c["action"][0]) in legal
    p = bridge.build([_row(PROMO_B, "h8h7")])
    # black-to-move promo-adjacent position: the mover is black;
    # after the mirror, the pawn (original white) sits on the
    # mirrored 8th rank and the promo action belongs to the opponent
    # here — just assert internal consistency + a black reply exists
    assert p["bad"] == 0


def test_header_elo_extraction(tmp_path):
    import chess
    import chess.pgn

    pgn_text = ('[Event "t"]\n[Site "-"]\n[Date "2026.08.28"]\n'
                '[White "A"]\n[Black "B"]\n[Result "1-0"]\n'
                '[WhiteElo "1234"]\n[BlackElo "876"]\n\n'
                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0\n")
    p = tmp_path / "g.pgn"
    p.write_text(pgn_text)
    rows = bridge.rows_from_pgn(p)
    assert len(rows) == 6
    assert rows[0]["elo_self"] == 1234 and rows[0]["elo_oppo"] == 876
    assert rows[1]["elo_self"] == 876 and rows[1]["elo_oppo"] == 1234
    assert all(r["engine"] == "human" and r["quality"] == "human"
               for r in rows)


def test_split_is_game_disjoint():
    rows = []
    for g in range(10):
        for ply in range(3):
            r = _row(START_W if ply % 2 == 0 else START_B,
                     "e2e4" if ply % 2 == 0 else "e7e5", ew=1000 + g,
                     eb=1500)
            r["game_id"] = g
            r["ply"] = ply + 1
            rows.append(r)
    arrays = bridge.build(rows)
    mask, val_ids = bridge.split_games(arrays, 3, arrays["game_id"])
    val = set(val_ids)
    assert len(val) == 3
    train_games = {int(g) for g in arrays["game_id"][mask]}
    val_games = {int(g) for g in arrays["game_id"][~mask]}
    assert train_games.isdisjoint(val_games)
    assert train_games | val_games == set(range(10))


def test_numerical_gradient_toy():
    """Analytical backprop vs central differences on a 4-row toy
    batch (float32 tolerance: ~1e-2 relative)."""
    rows = [_row(START_W, "e2e4"), _row(START_B, "e7e5"),
            _row(CASTLE_W, "e1g1"), _row(PROMO_B, "h8h7")]
    d = bridge.build(rows)
    n = d["rows"]
    x = pmp.features(d)[:n]
    legal, lc, y = d["legal"][:n], d["legal_count"][:n], d["action"][:n]
    model = pmp.MLP([x.shape[1], 8, 8, 20480], seed=1)
    logits, acts = model.forward(x)
    lsm = pmp.log_softmax_masked(logits, legal, lc)
    gl = np.exp(lsm)
    for i in range(n):
        gl[i, int(y[i])] -= 1.0
    gl /= n
    gW = [None] * 3
    gb = [None] * 3
    dd = gl
    for i in range(2, -1, -1):
        if i < 2:
            dd = dd * (acts[i + 1] > 0)
        gW[i] = dd.T @ acts[i]
        gb[i] = dd.sum(axis=0)
        if i > 0:
            dd = dd @ model.W[i]

    def loss_at():
        l, _ = model.forward(x)
        s = pmp.log_softmax_masked(l, legal, lc)
        return -float(np.mean([s[i, int(y[i])] for i in range(n)]))

    eps = 1e-5
    rng = np.random.default_rng(3)
    active = sorted({int(a) for i in range(n)
                     for a in legal[i, :int(lc[i])]})
    max_rel = 0.0
    for trial in range(6):
        layer = int(rng.integers(0, 3))
        if trial % 2 == 0:
            if layer < 2:
                arr = model.W[layer]
                idx = (int(rng.integers(arr.shape[0])),
                       int(rng.integers(arr.shape[1])))
            else:
                a = active[int(rng.integers(len(active)))]
                arr = model.W[layer]
                idx = (a, int(rng.integers(arr.shape[1])))
            g = gW[layer]
        else:
            if layer < 2:
                arr = model.b[layer]
                idx = int(rng.integers(arr.shape[0]))
            else:
                a = active[int(rng.integers(len(active)))]
                arr = model.b[layer]
                idx = a
            g = gb[layer]
        orig = float(arr[idx])
        arr[idx] = orig + eps
        lp = loss_at()
        arr[idx] = orig - eps
        lm = loss_at()
        arr[idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = float(g[idx])
        max_rel = max(max_rel, abs(ana - num) / max(1e-8, abs(num)))
    # float32 chain + central differences at eps=1e-5: ~1e-2 relative
    assert max_rel < 2e-2, max_rel


def test_toy_train_produces_report(tmp_path):
    rows = []
    for g in range(12):
        for i, (fen, mv) in enumerate([(START_W, "e2e4"),
                                       (START_B, "e7e5"),
                                       (CASTLE_W, "e1g1")]):
            r = _row(fen, mv, ew=1000 + 100 * (g % 5), eb=1500)
            r["game_id"] = g
            r["ply"] = i + 1
            rows.append(r)
    arrays = bridge.build(rows)
    n = arrays["rows"]
    import numpy as _np

    out = tmp_path / "data"
    out.mkdir()
    mask, _ = bridge.split_games(arrays, 3, arrays["game_id"])
    for name, sel in (("shard-train.npz", mask),
                      ("shard-val.npz", ~mask)):
        keep = {k: v[sel] for k, v in arrays.items()
                if k not in ("rows", "bad")}
        _np.savez_compressed(str(out / name), **keep)
    (out / "manifest.json").write_text("{}")

    x = pmp.features(arrays)[:n]
    legal, lc, y = arrays["legal"][:n], arrays["legal_count"][:n], \
        arrays["action"][:n]
    model = pmp.MLP([x.shape[1], 16, 16, 20480], seed=1)
    m = [np.zeros_like(w) for w in model.W]
    v = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]
    for t in range(1, 4):
        pmp.batch_step(model, x, legal, lc, y, 1e-3, m, v, mb, vb, t)
    report = pmp.diagnose(model, {**arrays}, sweep_positions=6, seed=1)
    assert report["conditioning_sweep"]["positions"] == 6
    assert "positions_flipped_any" in report["conditioning_sweep"]
    assert report["val_rows"] == n
    assert report["baseline_top1_accuracy"] > 0


if __name__ == "__main__":
    import tempfile
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames[:
                        fn.__code__.co_argcount]:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
