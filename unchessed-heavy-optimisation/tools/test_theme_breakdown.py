"""Tests for the tactical-theme tagging and its committed breakdown.

The tagger's whole value is that it is *independent* of the model it judges --
it reads the teacher's scores and the board only. These tests pin that
property down, check the classifier on positions whose themes are obvious by
construction, and assert the committed artifact still supports the claims in
`docs/unarchitectured-metal-theme-breakdown.md`.

No torch required.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "benchmarks" / "unarchitectured-metal" / "theme-breakdown-2026-08-25.json"
DOC = ROOT / "docs" / "unarchitectured-metal-theme-breakdown.md"

import sys  # noqa: E402

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tag_calibration_themes import classify  # noqa: E402


class ClassifierTests(unittest.TestCase):
    """Hand-built positions where the correct theme is unambiguous."""

    def test_capture_and_hanging_piece(self):
        # Black queen on d5 completely undefended, white pawn on e4 takes it.
        board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
        themes = classify(board, "e4d5", {"e4d5": 900, "e1e2": 0}, False)
        self.assertIn("capture", themes)
        self.assertIn("hanging_piece", themes)
        self.assertNotIn("quiet", themes)

    def test_defended_capture_is_not_hanging(self):
        # Queen on d5 defended by the pawn on c6, so it is not hanging.
        board = chess.Board("4k3/8/2p5/3q4/4P3/8/8/4K3 w - - 0 1")
        themes = classify(board, "e4d5", {"e4d5": 100, "e1e2": 0}, False)
        self.assertIn("capture", themes)
        self.assertNotIn("hanging_piece", themes)

    def test_quiet_move(self):
        board = chess.Board()
        themes = classify(board, "e2e4", {"e2e4": 30, "d2d4": 28}, False)
        self.assertIn("quiet", themes)
        self.assertNotIn("capture", themes)
        self.assertNotIn("check", themes)

    def test_check_is_tagged(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        themes = classify(board, "a1a8", {"a1a8": 100000, "g1g2": 0}, True)
        self.assertIn("check", themes)
        self.assertIn("mate_available", themes)

    def test_only_good_move_needs_a_real_gap(self):
        board = chess.Board()
        wide = classify(board, "e2e4", {"e2e4": 300, "d2d4": 10}, False)
        self.assertIn("only_good_move", wide)
        narrow = classify(board, "e2e4", {"e2e4": 30, "d2d4": 28}, False)
        self.assertNotIn("only_good_move", narrow)

    def test_many_good_moves_needs_three_close(self):
        board = chess.Board()
        themes = classify(
            board, "e2e4", {"e2e4": 30, "d2d4": 25, "g1f3": 20, "c2c4": 5}, False
        )
        self.assertIn("many_good_moves", themes)

    def test_endgame_technique_counts_non_kings(self):
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        themes = classify(board, "e2e4", {"e2e4": 50, "e1d2": 10}, False)
        self.assertIn("endgame_technique", themes)

    def test_illegal_best_move_yields_no_themes(self):
        """A mislabelled entry must produce nothing, not a wrong tag."""
        board = chess.Board()
        self.assertEqual(classify(board, "e7e5", {"e7e5": 0}, False), set())
        self.assertEqual(classify(board, "not-a-move", {}, False), set())

    def test_tagging_ignores_the_model(self):
        """classify() takes only board + teacher scores -- no model input."""
        import inspect

        params = list(inspect.signature(classify).parameters)
        self.assertEqual(params, ["board", "best_uci", "scores", "saw_mate"])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(ARTIFACT.read_text())

    def test_covers_the_corpus(self):
        self.assertEqual(self.report["positions_tagged"], 600)
        self.assertEqual(self.report["positions_scored"], 600)

    def test_every_tagged_entry_is_consistent(self):
        counts = self.report["theme_counts"]
        recount: dict[str, int] = {}
        for entry in self.report["tagged"]:
            for theme in entry["themes"]:
                recount[theme] = recount.get(theme, 0) + 1
        self.assertEqual(recount, counts)

    def test_capture_beats_quiet_by_a_wide_margin(self):
        """The headline finding, asserted against numbers not prose."""
        per = self.report["per_theme"]
        self.assertGreater(
            per["capture"]["top1_accuracy"], per["quiet"]["top1_accuracy"] * 3
        )

    def test_quiet_dominates_the_corpus(self):
        self.assertGreater(self.report["theme_counts"]["quiet"], 400)

    def test_per_theme_cells_are_well_formed(self):
        for theme, cell in self.report["per_theme"].items():
            with self.subTest(theme=theme):
                self.assertGreater(cell["positions"], 0)
                self.assertGreaterEqual(cell["top1_accuracy"], 0.0)
                self.assertLessEqual(cell["top1_accuracy"], 1.0)
                self.assertGreaterEqual(cell["mean_regret_cp"], 0.0)


class DocTests(unittest.TestCase):
    def test_doc_numbers_match_the_artifact(self):
        text = DOC.read_text()
        per = json.loads(ARTIFACT.read_text())["per_theme"]
        for value in (
            f"{per['capture']['top1_accuracy']:.4f}",
            f"{per['quiet']['top1_accuracy']:.4f}",
            f"{per['mate_available']['mean_regret_cp']:.1f}",
        ):
            self.assertIn(value, text, f"{value} missing from the doc")

    def test_doc_keeps_the_default_off_conclusion(self):
        self.assertIn("justifies enabling", DOC.read_text())


if __name__ == "__main__":
    unittest.main()
