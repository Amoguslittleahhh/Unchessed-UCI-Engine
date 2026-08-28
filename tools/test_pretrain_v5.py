#!/usr/bin/env python3
"""Tests for the v5 pretrain pipeline (CPU builder + GPU trainer).

Hermetic (no cloud, tiny temp PGNs):
  * v5 header + record pack/unpack round-trip, magic rejection,
  * promotion/castling 16-bit action encoding (exact values),
  * builder: dual-elo swap by side, quality from engine headers,
    default-engine fallback, target-in-legal guard, game-disjoint
    split, ep-file domain,
  * trusted-only quality filter (all-rows semantics),
  * GPU trainer (skipped when torch is absent): selfcheck, and a
    real-data smoke — build tiny v5 shards, load them with the v5
    shard reader, run 2 optimizer steps of the dual-elo oracle and a
    conditioning sweep.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO_ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import pretrain_v5_data as v5  # noqa: E402

import chess  # noqa: E402

GAME_A = """[Event "t"]
[Site "-"]
[Date "2026.08.28"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "1234"]
[BlackElo "876"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1-0
"""
GAME_B = """[Event "t"]
[Site "-"]
[Date "2026.08.28"]
[White "A"]
[Black "B"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1700"]
[WhiteEngine "maia3"]
[BlackEngine "lc0"]
[WhiteEloQuality "calibrated"]
[BlackEloQuality "approximate"]

1. d4 d5 2. c4 e6 0-1
"""
GAME_C = """[Event "t"]
[Site "-"]
[Date "2026.08.28"]
[White "A"]
[Black "B"]
[Result "1/2-1/2"]
[WhiteElo "1200"]
[BlackElo "900"]

1. e4 e6 1/2-1/2
"""


def _build(tmp_path, games, extra=(), name="g.pgn"):
    if isinstance(games, str):
        games = [games]
    p = tmp_path / name
    # blank line between games: python-chess needs the separator
    p.write_text("\n\n".join(g.strip("\n") for g in games))
    out = tmp_path / "v5"
    cmd = [sys.executable, str(TOOLS / "pretrain_v5_data.py"), "build",
           "--pgn", str(p), "--out", str(out), "--val-games", "1",
           *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out


def _load_side(out: Path, side: str) -> list[dict]:
    import numpy as np

    records = []
    for shard in sorted((out / side).glob("*.v5")):
        with shard.open("rb") as handle:
            count = v5.parse_header(handle.read(v5.HEADER_BYTES))
        mm = np.memmap(str(shard), dtype=np.uint8, mode="r",
                       offset=v5.HEADER_BYTES,
                       shape=(count, v5.RECORD_BYTES))
        for i in range(count):
            records.append(v5.unpack_record(bytes(mm[i])))
    return records


# ----------------------------------------------------------------------
# wire format
# ----------------------------------------------------------------------

def test_header_roundtrip_and_magic_rejection():
    header = v5.make_header(7)
    assert v5.parse_header(header) == 7
    assert header[:8] == v5.MAGIC
    wrong = b"UNCHD4R0" + header[8:]
    with pytest.raises(ValueError):
        v5.parse_header(wrong)


def _sample_row():
    bb = tuple(range(12))
    hist = (1, 2, 3, 4, 5, 6, 7, 8)
    return dict(bb=bb, move=0x123, promotion=1, wdl=2, rating=1500,
                castling=0xF, ep_file=3, halfmove=5, time_class=0,
                flags=4, history_len=8, history=hist,
                game_hash=0xAABBCCDD00112233,
                player_hash=0x1122334455667788,
                teacher_score=0, best_move=0, best_score=0,
                move_score=0, ply=9, remaining_ms=0, increment_ms=0,
                base_reserved=0, legal_count=2, target_action=0x123,
                teacher_best_action=v5.ACTION_SENTINEL, policy_kind=0,
                legal_flags=0,
                legal_actions=(0x123, 0x456) +
                              (v5.ACTION_SENTINEL,) *
                              (v5.MAX_LEGAL_ACTIONS - 2),
                legal_regrets=(0,) * v5.MAX_LEGAL_ACTIONS,
                elo_oppo=1700, pretrain_quality=1)


def test_record_pack_unpack_roundtrip():
    row = _sample_row()
    payload = v5.pack_record(**row)
    assert len(payload) == v5.RECORD_BYTES
    out = v5.unpack_record(payload)
    assert out["rating"] == 1500 and out["elo_oppo"] == 1700
    assert out["pretrain_quality"] == 1
    assert out["target_action"] == 0x123
    assert out["teacher_best_action"] == v5.ACTION_SENTINEL
    assert out["history"] == row["history"]
    assert out["legal_count"] == 2
    assert out["legal_actions"][1] == 0x456


def test_v5_action_encoding_exact_values():
    # f2-f8=Q: f2 (square 13) mirrors to f7 (53), f8 (61) to f1 (5);
    # queen promo = 3, kind bit set
    q = v5.v5_action(chess.Move.from_uci("f2f8q"))
    assert q == (53 | (5 << 6) | (3 << 12) | (1 << 14))
    # e1-g1 castle (no promo): e1 (4) mirrors to e8 (60), g1 (7) to g8
    g1 = v5.v5_action(chess.Move.from_uci("e1g1"))
    assert g1 == 60 | (62 << 6)
    assert g1 & (1 << 14) == 0
    # d7-d8 knight promotion
    n = v5.v5_action(chess.Move.from_uci("d7d8n"))
    d7s = chess.square_mirror(chess.D7)
    d8s = chess.square_mirror(chess.D8)
    assert n == d7s | (d8s << 6) | (0 << 12) | (1 << 14)


# ----------------------------------------------------------------------
# builder
# ----------------------------------------------------------------------

def test_builder_dual_elo_and_split(tmp_path):
    out = _build(tmp_path, [GAME_A, GAME_B])
    train = _load_side(out, "train")
    val = _load_side(out, "val")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["games"]["train"] == 1
    assert manifest["games"]["val"] == 1
    # game-disjoint
    train_games = {r["game_hash"] for r in train}
    val_games = {r["game_hash"] for r in val}
    assert train_games.isdisjoint(val_games)
    # the 8-ply game (GAME_A) vs the 4-ply game (GAME_B)
    assert len(train) + len(val) == 12
    # dual-elo swap by side (GAME_A elos 1234/876)
    a_rows = [r for r in train + val if r["rating"] in (1234, 876)]
    assert a_rows
    for r in a_rows:
        assert {r["rating"], r["elo_oppo"]} == {1234, 876}
    # first ply white to move: rating = white elo
    white_first = [r for r in a_rows if r["ply"] == 1]
    assert white_first and white_first[0]["rating"] == 1234
    # quality from engine headers (GAME_B: maia3 + lc0)
    quals = {r["pretrain_quality"] for r in train + val}
    assert v5.QUALITY_HUMAN in quals       # GAME_A (no engine headers)
    assert v5.QUALITY_CALIBRATED in quals  # maia3 side
    assert v5.QUALITY_APPROXIMATE in quals  # lc0 side
    # invariants on every row
    for r in train + val:
        legal = list(r["legal_actions"][:r["legal_count"]])
        assert r["target_action"] in legal
        assert r["ep_file"] == 0xFF or 0 <= r["ep_file"] <= 7
        assert 0 <= r["castling"] <= 15
        assert 0 <= r["wdl"] <= 2
    # the castling move (GAME_A ply 8) set FLAG_CASTLE
    castle = [r for r in a_rows if r["target_action"] ==
              v5.v5_action(chess.Move.from_uci("e1g1"))]
    assert castle and castle[0]["flags"] & 1
    # history: the move at ply N carries its previous plies (capped at
    # 8) — ply 8 therefore has exactly 7
    deep = [r for r in a_rows if r["ply"] == 8]
    assert deep and deep[0]["history_len"] == 7


def test_builder_default_engine(tmp_path):
    out = _build(tmp_path, [GAME_A, GAME_C],
                 extra=("--default-engine", "maia3"))
    train = _load_side(out, "train")
    val = _load_side(out, "val")
    records = train + val
    assert records
    assert all(r["pretrain_quality"] == v5.QUALITY_CALIBRATED
               for r in records)
    assert all(r["policy_kind"] == v5.POLICY_HUMAN for r in records)
    # GAME_C is a draw -> wdl proxy = 1 on both sides
    c_rows = [r for r in records if r["rating"] in (1200, 900)]
    assert c_rows and all(r["wdl"] == 1 for r in c_rows)


def test_trusted_filter_excludes_approximate_games(tmp_path):
    out = _build(tmp_path, [GAME_A, GAME_B, GAME_C],
                 extra=("--quality-filter", "calibrated,native,human"))
    records = _load_side(out, "train") + _load_side(out, "val")
    manifest = json.loads((out / "manifest.json").read_text())
    # GAME_B (has approximate rows) is excluded; A + C survive
    assert manifest["games"]["train"] + manifest["games"]["val"] == 2
    assert all(r["pretrain_quality"] in
               (v5.QUALITY_CALIBRATED, v5.QUALITY_NATIVE,
                v5.QUALITY_HUMAN) for r in records)
    assert not any(r["rating"] in (1500, 1700) for r in records)


# ----------------------------------------------------------------------
# GPU trainer (torch)
# ----------------------------------------------------------------------

def test_selfcheck_cpu():
    import argparse

    torch = pytest.importorskip("torch", reason="torch not installed")
    del torch
    from pretrain_v1_a100 import selfcheck

    args = argparse.Namespace(config=str(REPO_ROOT /
                                         "config/pretrain_v1_training.json"))
    assert selfcheck(args) == 0


def test_loader_training_smoke(tmp_path):
    torch = pytest.importorskip("torch", reason="torch not installed")
    import numpy as np

    from a100_common import configure_torch, load_config
    from pretrain_v1_a100 import (
        UnarchitecturedV1OracleDualElo,
        UnarchitecturedV5RecordShards,
        conditioning_sweep,
        pretrain_loss,
        prepare_v5_batch,
    )
    from train_unarchitectured_v1_a100 import make_optimizer

    out = _build(tmp_path, [GAME_A, GAME_C])
    train = UnarchitecturedV5RecordShards(
        [str(p) for p in sorted((out / "train").glob("*.v5"))])
    val = UnarchitecturedV5RecordShards(
        [str(p) for p in sorted((out / "val").glob("*.v5"))])
    assert train.total == 8 and val.total == 2

    config, _hw = load_config(str(REPO_ROOT /
                                  "config/pretrain_v1_training.json"),
                              "pretrain")
    small = {**config, "d_model": 64, "board_layers": 2,
             "board_heads": 4, "board_ffn": 128,
             "gab_token_projection": 4, "gab_hidden": 16,
             "gab_templates": 8, "decoder_layers": 1,
             "decoder_heads": 4, "decoder_ffn": 128, "history_width": 16,
             "policy_adapter_rank": 8, "concept_count": 16,
             "concept_width": 8, "activation_checkpointing": False}
    configure_torch(20260828, True)
    device = torch.device("cpu")
    model = UnarchitecturedV1OracleDualElo(small).to(device)
    optimizer = make_optimizer(model, small["learning_rate"],
                               small["weight_decay"], device)
    records = train.gather(np.arange(train.total))
    loss = None
    for _ in range(2):
        batch = prepare_v5_batch(records, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, _ = pretrain_loss(output, batch, small, [1, 1, 0.5, 1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    assert torch.isfinite(loss)
    # dual-elo: same position, different elos -> different logits
    b = prepare_v5_batch(train.gather(np.array([0, 0])), device)
    b["rating"] = torch.tensor([600, 3200], dtype=torch.int64)
    b["elo_oppo"] = torch.tensor([1500, 1500], dtype=torch.int64)
    with torch.no_grad():
        lgts = model(b)["logits"].float()
    assert not torch.allclose(lgts[0], lgts[1])
    # conditioning sweep runs over real records
    sweep = conditioning_sweep(model, records, device,
                               sweep=(600, 1200, 300))
    assert sweep["positions"] == train.total
    assert 0 <= sweep["positions_flipped_any"] <= train.total


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
