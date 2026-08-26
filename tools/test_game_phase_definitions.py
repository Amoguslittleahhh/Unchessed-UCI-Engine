#!/usr/bin/env python3
"""Tests for tools/measure_game_phase_definitions.py.

The Lichess definition is pinned against:
  * the exact per-block score table from scalachess Divider.scala
    (spot values hand-computed from the paper's Appendix B / the source);
  * two real corpus positions in which the mixedness term alone or the
    backrank-sparseness term alone changes the phase (values measured with
    the tool on 2026-08-26);
  * the corpus builder's stored phase tags (the mirror must agree with all
    600 of them, which also pins that the mirror was copied faithfully).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import chess

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from measure_game_phase_definitions import (  # noqa: E402
    DEFINITIONS,
    backrank_sparse,
    corpus_builder_phase,
    lichess_material_only_phase,
    lichess_mixedness,
    lichess_phase,
    majors_and_minors,
    read_corpus_positions,
    read_epd_positions,
    _block_score,
)

CORPUS = REPO / "artifacts" / "unarchitectured-v1-calibration-corpus.jsonl"
MATRACK = REPO / "benchmarks" / "matetrack.epd"
ARTIFACT = REPO / "benchmarks" / "unarchitectured-v1" / "game-phase-definitions.json"

# Corpus positions where exactly one Lichess term decides the phase
# (measured with the tool, 2026-08-26; see the artifact).
MIXEDNESS_DECIDES_FEN = "rn1qk2r/p3ppbp/2p1bnp1/1p4N1/2pPP3/2N5/PP2BPPP/R1BQ1RK1 b kq - 1 9"
MIXEDNESS_DECIDES_VALUE = 153
BACKRANK_DECIDES_FEN = "r1b1r1k1/1pp1qppp/2n2n2/p3p3/2P5/PPB1PN2/2Q1BPPP/R3K2R w KQ - 1 13"
BACKRANK_DECIDES_MIXEDNESS = 146


def _board_with_white_pieces(squares):
    b = chess.Board.empty()
    b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    for sq in squares:
        b.set_piece_at(sq, chess.Piece(chess.ROOK, chess.WHITE))
    return b


class TestBlockScoreTable:
    """Hand-computed values from the Divider.scala score function."""

    @pytest.mark.parametrize(
        ("y", "white", "black", "expected"),
        [
            (1, 0, 1, 2),    # 1 + y
            (7, 0, 1, 8),    # 1 + y
            (1, 0, 2, 7),    # 2 + (6-y), y < 6
            (6, 0, 2, 0),    # y < 6 fails
            (1, 0, 3, 9),    # 3 + (7-y), y < 7
            (6, 0, 4, 4),    # 3 + (7-y), y < 7
            (7, 0, 4, 0),    # y < 7 fails
            (1, 1, 0, 8),    # 1 + (8-y)
            (2, 1, 0, 7),
            (1, 1, 1, 8),    # 5 + |4-y|
            (7, 1, 1, 8),    # 5 + |4-y|
            (4, 1, 1, 5),    # 5 + 0
            (1, 1, 2, 10),   # 4 + (7-y)
            (1, 1, 3, 11),   # 5 + (7-y)
            (1, 2, 0, 0),    # y > 2 fails
            (3, 2, 0, 3),    # 2 + (y-2)
            (1, 2, 1, 4),    # 4 + (y-1)
            (7, 2, 1, 10),   # 4 + (y-1)
            (4, 2, 2, 7),    # flat 7
            (1, 3, 0, 0),    # y > 1 fails
            (2, 3, 0, 4),    # 3 + (y-1)
            (1, 3, 1, 5),    # 5 + (y-1)
            (1, 4, 0, 0),    # "group of 4 on the homerow = 0"
            (3, 4, 0, 5),    # 3 + (y-1)
            (1, 5, 1, 0),    # white > 4: 0
            (1, 0, 5, 0),    # black > 4: 0
        ],
    )
    def test_table(self, y, white, black, expected):
        assert _block_score(y, white, black) == expected

    def test_initial_position_mixedness_is_zero(self):
        assert lichess_mixedness(chess.Board()) == 0

    def test_empty_board_mixedness_is_zero(self):
        assert lichess_mixedness(chess.Board.empty()) == 0

    def test_scattered_beats_home_row_cluster(self):
        # A same-color home row scores 0 only when fully packed:
        # score(1, 2, 0) = 0, and a full row a1-h1 puts exactly two pieces
        # in every 2x2 block it touches.
        home = chess.Board.empty()
        for x in range(8):
            home.set_piece_at(chess.square(x, 0), chess.Piece(chess.PAWN, chess.WHITE))
        assert lichess_mixedness(home) == 0
        # An almost-full row leaks one lone pawn into the last block:
        # score(1, 1, 0) = 1 + (8-1) = 8.
        home2 = chess.Board.empty()
        for x in range(7):
            home2.set_piece_at(chess.square(x, 0), chess.Piece(chess.PAWN, chess.WHITE))
        assert lichess_mixedness(home2) == 8
        # Two white pawns a1-b1 next to two black pawns e1-f1. Every
        # affected block, hand-computed from the table:
        #   (a-b): white 2            -> score(1, 2, 0) = 0
        #   (b-c): white 1 (b1 lone)  -> score(1, 1, 0) = 1 + 7 = 8
        #   (d-e): black 1 (e1 lone)  -> score(1, 0, 1) = 1 + 1 = 2
        #   (e-f): black 2            -> score(1, 0, 2) = 2 + 5 = 7
        #   (f-g): black 1 (f1 lone)  -> score(1, 0, 1) = 2
        # total 19 > 0: enemy pieces intermixed on the home row score.
        mixed = chess.Board.empty()
        for sq in (chess.A1, chess.B1):
            mixed.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
        for sq in (chess.E1, chess.F1):
            mixed.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
        assert lichess_mixedness(mixed) == 19


class TestLichessPhase:
    def test_initial_position_is_opening(self):
        b = chess.Board()
        assert majors_and_minors(b) == 14
        assert not backrank_sparse(b)
        assert lichess_phase(b) == "opening"

    def test_endgame_threshold_is_six_majors_and_minors(self):
        # 6 rooks total (5 white + 1 black): endgame.
        b = _board_with_white_pieces([chess.A1, chess.H1, chess.E2, chess.E5, chess.C5])
        b.set_piece_at(chess.A8, chess.Piece(chess.ROOK, chess.BLACK))
        assert majors_and_minors(b) == 6
        assert lichess_phase(b) == "endgame"
        # 7: no longer endgame; black's backrank has 3 pieces -> sparse,
        # so the position is middlegame (threshold AND rule order both checked).
        b.set_piece_at(chess.C8, chess.Piece(chess.ROOK, chess.BLACK))
        assert majors_and_minors(b) == 7
        assert lichess_phase(b) == "middlegame"

    def test_mixedness_alone_decides_middlegame(self):
        b = chess.Board(MIXEDNESS_DECIDES_FEN)
        assert majors_and_minors(b) > 10
        assert not backrank_sparse(b)
        assert lichess_mixedness(b) == MIXEDNESS_DECIDES_VALUE
        assert lichess_material_only_phase(b) == "opening"
        assert lichess_phase(b) == "middlegame"

    def test_backrank_sparseness_alone_decides_middlegame(self):
        b = chess.Board(BACKRANK_DECIDES_FEN)
        assert majors_and_minors(b) > 10
        assert backrank_sparse(b)
        assert lichess_mixedness(b) == BACKRANK_DECIDES_MIXEDNESS  # <= 150
        assert lichess_material_only_phase(b) == "opening"
        assert lichess_phase(b) == "middlegame"

    def test_material_only_ignores_backrank_and_mixedness(self):
        b = chess.Board(MIXEDNESS_DECIDES_FEN)
        assert lichess_material_only_phase(b) == "opening"
        b2 = chess.Board(BACKRANK_DECIDES_FEN)
        assert lichess_material_only_phase(b2) == "opening"


class TestCorpusBuilderMirror:
    def test_mirror_matches_every_stored_tag(self):
        recs = read_corpus_positions(CORPUS)
        assert len(recs) == 600
        for fen, tag in recs:
            assert corpus_builder_phase(chess.Board(fen)) == tag, fen

    def test_endgame_boundary_total_pieces(self):
        # 12 pieces total with 5 non-pawns: the non_pawn <= 4 trigger does
        # NOT fire, so the total <= 12 branch is what makes it endgame.
        def board(extra_pawn):
            b = chess.Board.empty()
            b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
            for sq in (chess.A1, chess.B1, chess.C1, chess.D1):
                b.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.WHITE))
            for sq in (chess.F1, chess.G1):
                b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
            b.set_piece_at(chess.A8, chess.Piece(chess.KNIGHT, chess.BLACK))
            for sq in (chess.B8, chess.C8, chess.D8):
                b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
            if extra_pawn:
                b.set_piece_at(chess.F8, chess.Piece(chess.PAWN, chess.BLACK))
            b.fullmove_number = 13  # above the opening threshold
            return b

        b12, b13 = board(False), board(True)
        assert len(b12.piece_map()) == 12
        non_pawn = sum(1 for p in b12.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
        assert non_pawn == 5  # endgame trigger must come from the total
        assert corpus_builder_phase(b12) == "endgame"
        assert len(b13.piece_map()) == 13
        assert corpus_builder_phase(b13) == "middlegame"

    def test_opening_boundary_fullmove(self):
        # 14 pieces, 5 non-pawns (no endgame trigger), fullmove 12 ->
        # opening; fullmove 13 -> middlegame.
        def board(fullmove):
            b = chess.Board.empty()
            b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
            for sq in (chess.F1, chess.G1, chess.F8, chess.G8):
                b.set_piece_at(
                    sq,
                    chess.Piece(chess.QUEEN, chess.WHITE if sq < 32 else chess.BLACK),
                )
            b.set_piece_at(chess.H2, chess.Piece(chess.KNIGHT, chess.WHITE))
            for sq in (chess.A1, chess.B1, chess.C1):
                b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
            for sq in (chess.A8, chess.B8, chess.C8, chess.D8):
                b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
            b.fullmove_number = fullmove
            return b

        b12, b13 = board(12), board(13)
        assert len(b12.piece_map()) == 14
        assert corpus_builder_phase(b12) == "opening"
        assert corpus_builder_phase(b13) == "middlegame"


class TestMatetrack:
    def test_every_matetrack_position_is_endgame_everywhere(self):
        recs = read_epd_positions(MATRACK)
        assert len(recs) == 7
        for fen, label in recs:
            b = chess.Board(fen)
            for name, fn in DEFINITIONS.items():
                assert fn(b) == "endgame", (label, name, fen)


class TestArtifact:
    def test_committed_artifact_is_consistent(self):
        data = json.loads(ARTIFACT.read_text())
        total = data["by_definition"]["lichess"]["total"]
        assert total == 607  # 600 corpus + 7 matetrack
        for name in ("lichess", "lichess_material_only", "corpus_builder"):
            d = data["by_definition"][name]
            assert sum(d["counts"].values()) == total
        # matetrack rows all endgame
        for rec in data["per_position_epd"]:
            assert rec["lichess"] == rec["lichess_material_only"] == rec["corpus_builder"] == "endgame"
        assert "retrain" in data["conclusion"]


class TestCli:
    def test_help_runs_standalone(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "measure_game_phase_definitions.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0
        assert "Lichess" in out.stdout

    def test_no_input_is_an_error(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "measure_game_phase_definitions.py")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 2

    def test_only_declared_dependency_is_chess(self):
        """Fresh-clone guarantee: the tool imports nothing beyond stdlib and
        `chess`, which tools/requirements-dev.txt installs."""
        import ast

        src = (TOOLS / "measure_game_phase_definitions.py").read_text()
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        assert mods == {
            "__future__",
            "argparse",
            "json",
            "sys",
            "collections",
            "pathlib",
            "chess",
        }, mods
