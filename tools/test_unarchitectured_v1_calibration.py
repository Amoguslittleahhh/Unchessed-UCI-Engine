#!/usr/bin/env python3
"""Regression tests for the round-6 provenance-disjoint calibration tooling.

These lock in the parts that would silently invalidate a calibration report if
they broke: the position encoder must keep matching the Rust runtime's
`position_to_input`, the corpus builder must keep excluding in-repo fixtures,
and the statistics helpers must stay correct.

The tests that need `python-chess` skip cleanly when it is absent, so this file
still runs in a bare checkout.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import chess  # noqa: F401

    HAVE_CHESS = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_CHESS = False


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


class EncoderTests(unittest.TestCase):
    """The encoder must agree with `position_to_input` in aegis_v4_runtime.rs."""

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_self_check_passes(self):
        result = subprocess.run(
            [sys.executable, str(TOOLS / "unarchitectured_v1_position_encoding.py"), "--self-check"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_start_position_matches_frozen_fixture(self):
        module = load_module("_encoding", TOOLS / "unarchitectured_v1_position_encoding.py")
        board = chess.Board()
        encoded = module.encode_position(board)
        self.assertEqual(len(encoded["legal_actions"]), 20)
        self.assertEqual(encoded["castling"], 15)
        self.assertEqual(encoded["ep_file"], module.NO_EP_FILE)
        # g1f3 is the frozen parity fixture's best action.
        self.assertEqual(
            module.encode_action(board, chess.Move.from_uci("g1f3")), 1350
        )

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_black_to_move_is_vertically_flipped(self):
        module = load_module("_encoding", TOOLS / "unarchitectured_v1_position_encoding.py")
        white = module.encode_position(chess.Board())
        black_board = chess.Board()
        black_board.push_uci("e2e4")
        black = module.encode_position(black_board)
        # Under the mover-perspective flip, the side to move always sees its
        # own pawns on encoded rank 1 with mover value 1. Black has not moved a
        # pawn after 1. e4, so all eight are still there.
        self.assertEqual(white["pieces"][8:16], [1] * 8)
        self.assertEqual(black["pieces"][8:16], [1] * 8)
        # White's e4 pawn (square 28) flips to index 36 as an opponent pawn (7).
        self.assertEqual(black["pieces"][36], 7)
        self.assertEqual(black["pieces"].count(7), 8)
        self.assertEqual(
            module.encode_action(black_board, chess.Move.from_uci("e7e5")),
            module.encode_action(chess.Board(), chess.Move.from_uci("e2e4")),
        )

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_promotion_and_castling_encoding(self):
        module = load_module("_encoding", TOOLS / "unarchitectured_v1_position_encoding.py")
        board = chess.Board("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
        action = module.encode_action(board, chess.Move.from_uci("b7b8q"))
        self.assertEqual(action >> 12, 4, "queen promotion code")
        knight = module.encode_action(board, chess.Move.from_uci("b7b8n"))
        self.assertEqual(knight >> 12, 1, "knight promotion code")

        # Mover-relative castling bits: black to move sees its own rights first.
        black = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        encoded = module.encode_position(black)
        self.assertEqual(encoded["castling"], 15)
        black_king_only = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b k - 0 1")
        self.assertEqual(module.encode_position(black_king_only)["castling"], 1)


class CorpusBuilderTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_fixture_fens_are_excluded(self):
        module = load_module(
            "_corpus", TOOLS / "build_unarchitectured_v1_calibration_corpus.py"
        )
        # Every in-repo smoke fixture and the start position must be excluded so
        # calibration cannot silently score itself on its own fixtures.
        self.assertIn(chess.STARTING_FEN, module.EXCLUDED_FENS)
        self.assertIn(
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
            module.EXCLUDED_FENS,
        )
        self.assertGreaterEqual(len(module.EXCLUDED_FENS), 9)

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_phase_classification(self):
        module = load_module(
            "_corpus", TOOLS / "build_unarchitectured_v1_calibration_corpus.py"
        )
        self.assertEqual(
            module.classify_phase(chess.Board("8/5pk1/6p1/3p4/3P1P2/5KP1/8/8 w - - 0 40")),
            "endgame",
        )
        opening = chess.Board()
        opening.push_uci("e2e4")
        self.assertEqual(module.classify_phase(opening), "opening")

    def test_help_runs_standalone(self):
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "build_unarchitectured_v1_calibration_corpus.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "No module named 'chess'" in result.stderr:
            self.skipTest("python-chess not installed")
        self.assertEqual(result.returncode, 0, result.stderr)


class StatisticsTests(unittest.TestCase):
    """The reported uncertainty must actually be correct."""

    def load_calibrator(self):
        path = TOOLS / "calibrate_unarchitectured_v1_policy.py"
        try:
            return load_module("_calibrate", path)
        except ImportError as error:  # torch/chess absent
            self.skipTest(f"calibration dependencies unavailable: {error}")

    def test_wilson_interval_brackets_estimate(self):
        module = self.load_calibrator()
        low, high = module.wilson_interval(153, 600)
        self.assertLess(low, 153 / 600)
        self.assertGreater(high, 153 / 600)
        # A significant result must exclude the ~1/30 random-ordering floor.
        self.assertGreater(low, 0.05)
        self.assertEqual(module.wilson_interval(0, 0), (0.0, 0.0))

    def test_mcnemar_detects_and_ignores(self):
        module = self.load_calibrator()
        # Lopsided disagreement is significant.
        a = [1] * 79 + [0] * 20 + [1] * 100
        b = [0] * 79 + [1] * 20 + [1] * 100
        only_a, only_b, pvalue = module.mcnemar_pvalue(a, b)
        self.assertEqual((only_a, only_b), (79, 20))
        self.assertLess(pvalue, 1e-6)

        # Identical predictions are never significant.
        same = [1, 0, 1, 1, 0]
        self.assertEqual(module.mcnemar_pvalue(same, same), (0, 0, 1.0))

        # Balanced disagreement is not significant.
        _, _, balanced = module.mcnemar_pvalue([1, 0, 1, 0], [0, 1, 0, 1])
        self.assertGreater(balanced, 0.5)

    def test_default_policy_kind_matches_encoder_constant(self):
        """The literal used for --help must not drift from POLICY_GUIDE."""
        module = self.load_calibrator()
        encoding = load_module("_encoding", TOOLS / "unarchitectured_v1_position_encoding.py")
        self.assertEqual(module._DEFAULT_POLICY_KIND, encoding.POLICY_GUIDE)

    def test_wdl_bucket_deadband(self):
        module = self.load_calibrator()
        self.assertEqual(module.wdl_bucket(200), 0)
        self.assertEqual(module.wdl_bucket(0), 1)
        self.assertEqual(module.wdl_bucket(-200), 2)
        self.assertEqual(module.wdl_bucket(50), 1, "deadband is inclusive")

    @unittest.skipUnless(HAVE_CHESS, "python-chess not installed")
    def test_heuristic_baseline_prefers_winning_captures(self):
        module = self.load_calibrator()
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        capture = module.heuristic_move_score(board, chess.Move.from_uci("e4d5"))
        quiet = module.heuristic_move_score(board, chess.Move.from_uci("a2a3"))
        self.assertGreater(capture, quiet)


if __name__ == "__main__":
    unittest.main()
