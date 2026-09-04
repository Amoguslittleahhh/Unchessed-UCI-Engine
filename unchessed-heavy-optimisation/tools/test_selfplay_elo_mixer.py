#!/usr/bin/env python3
"""Tests for tools/selfplay_elo_mixer.py (Maia-3 random-elo self-play).

Covers: the 4352-move index scheme (forward + inverse, entry-pinned
against the reference ALL_MOVES_MAIA3 table when the ONNX mirror is
available in /tmp, and by construction rules otherwise), the
mirror-move transform (inverse of chess.Board.mirror), the PGN writer
round-trip, the committed data/selfplay/ dataset (every game re-parses,
every move replays legal, headers match labels), --help, and the import
scan. The ONNX model itself is NOT committed (GPL-3.0, fetched via
fetch-model); the dataset test is the end-to-end check that needs no
model.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
TOOL = TOOLS / "selfplay_elo_mixer.py"
sys.path.insert(0, str(TOOLS))

import selfplay_elo_mixer as sp  # noqa: E402

DATASET_PGN = REPO_ROOT / "data" / "selfplay" / "maia3-100-3200.pgn"
DATASET_LABELS = REPO_ROOT / "data" / "selfplay" / "maia3-100-3200-labels.jsonl"


def _read_all(stream_text: str):
    """python-chess 1.11: read_game reads ONE game per call (not a generator)."""
    stream = io.StringIO(stream_text)
    games = []
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        games.append(g)
    return games


def _reference_table():
    """ALL_MOVES_MAIA3 from the mirror, if staged under /tmp/smi."""
    p = Path("/tmp/smi/simple_maia3_inference/constants.py")
    if not p.exists():
        return None
    namespace: dict = {}
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), namespace)
    return namespace["ALL_MOVES_MAIA3"]


def test_index_scheme():
    ref = _reference_table()
    if ref is not None:
        assert len(ref) == 4352
        for uci, idx in ref.items():
            assert sp.move_to_index(uci) == idx, uci
        for i in range(4352):
            assert ref.get(sp.index_to_move(i)) == i, i
    else:
        # construction rules, pinned by hand-verified entries
        assert sp.move_to_index("a1b1") == 1          # frm-major flat 64x64
        assert sp.move_to_index("b1a1") == 64
        assert sp.move_to_index("h8a1") == 4032
        assert sp.move_to_index("a1a1") == 0          # diagonal entries exist
        assert sp.move_to_index("a7a8q") == 4096
        assert sp.move_to_index("a7a8r") == 4097
        assert sp.move_to_index("a7a8b") == 4098
        assert sp.move_to_index("a7a8n") == 4099
        assert sp.move_to_index("h7h8n") == 4351
        assert sp.index_to_move(4351) == "h7h8n"
        assert sp.index_to_move(4096) == "a7a8q"
    # every legal move of a middlegame maps to a distinct, invertible index
    board = chess.Board("r1bq1rk1/pp2ppbp/2np1np1/3p1Q2/2PPP3/2N5/BP4P1/R1B1K2R w KQ - 0 18")
    idxs = [sp.move_to_index(m.uci()) for m in board.legal_moves]
    assert len(set(idxs)) == len(idxs)
    for i in idxs:
        assert 0 <= i < 4352


def test_mirror_move():
    # inverse of chess.Board.mirror(): rank r -> 9-r, file unchanged
    assert sp.mirror_move("e2e4") == "e7e5"
    assert sp.mirror_move("a2a8q") == "a7a1q"
    assert sp.mirror_move(sp.mirror_move("g1f3")) == "g1f3"
    # the full transform round-trips the legal-move set of a black position
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    mb = board.mirror()
    mapped = {sp.mirror_move(m.uci()) for m in mb.legal_moves}
    assert mapped == {m.uci() for m in board.legal_moves}


def test_pgn_writer_roundtrip(tmp_path):
    games = [
        {"headers": {"Event": "T", "White": "W", "Black": "B",
                     "WhiteElo": "1500", "BlackElo": "2000", "Result": "1-0"},
         "san": ["e4", "e5", "Nf3", "Nc6"], "result": "1-0"},
        {"headers": {"Event": "T", "White": "W", "Black": "B",
                     "WhiteElo": "300", "BlackElo": "900", "Result": "1/2-1/2"},
         "san": ["d4", "d5"], "result": "1/2-1/2"},
    ]
    out = tmp_path / "x.pgn"
    sp.write_pgn(out, games)
    parsed = _read_all(out.read_text())
    assert len(parsed) == 2
    assert parsed[0].headers["WhiteElo"] == "1500"
    assert [m.uci() for m in parsed[0].mainline_moves()] == [
        "e2e4", "e7e5", "g1f3", "b8c6"]
    assert parsed[1].headers["BlackElo"] == "900"


def test_committed_dataset_is_legal_and_consistent():
    assert DATASET_PGN.exists(), "run the self-play generator first"
    assert DATASET_LABELS.exists()
    text = DATASET_PGN.read_text(encoding="utf-8")
    games = _read_all(text)
    assert len(games) == 200
    labels = [json.loads(l) for l in DATASET_LABELS.read_text().splitlines()]
    assert len(labels) > 0
    by_game: dict[int, list[dict]] = {}
    for r in labels:
        by_game.setdefault(r["game"], []).append(r)
    assert set(by_game) == set(range(1, len(games) + 1))
    total_plies = 0
    for g in games:
        w, b = int(g.headers["WhiteElo"]), int(g.headers["BlackElo"])
        assert 100 <= w <= 3200 and 100 <= b <= 3200
        rows = by_game[int(g.headers["Round"])]
        moves = list(g.mainline_moves())
        assert len(rows) == len(moves), (g.headers["Round"], len(rows), len(moves))
        total_plies += len(moves)
        # replay: every move legal, label FENs/moves/sides/elos consistent
        board = chess.Board()
        for row, mv in zip(rows, moves):
            assert row["fen"] == board.fen()
            assert row["move_uci"] == mv.uci()
            white_turn = board.turn == chess.WHITE
            assert row["side"] == ("white" if white_turn else "black")
            assert row["elo_self"] == (w if white_turn else b)
            assert row["elo_oppo"] == (b if white_turn else w)
            assert 0 < row["top1_prob"] <= 1
            board.push(mv)
    assert total_plies == len(labels)
    # both elos must actually vary across games (random uci elo mix)
    assert len({(int(g.headers["WhiteElo"]), int(g.headers["BlackElo"]))
                for g in games}) > 150


def test_help_and_import_scan():
    out = subprocess.run([sys.executable, str(TOOL), "--help"],
                         capture_output=True, text=True, cwd=REPO_ROOT,
                         timeout=60)
    assert out.returncode == 0 and "self-play" in out.stdout.lower()
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
