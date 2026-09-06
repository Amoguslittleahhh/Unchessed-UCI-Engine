"""Tests for the hint-disagreement fixtures.

Two things are checked here, and neither needs `torch`:

1. The committed artifact is internally coherent -- the chess claims in it are
   re-derived with `python-chess` rather than trusted. An earlier draft of the
   generator listed a "mate" that a simple recapture refuted; that mistake was
   caught by re-deriving, so the check is kept permanently.

2. The Rust safety tests in `unchessed-core/src/search.rs` use exactly the
   logits recorded in the artifact. Those numbers are transcribed by hand into
   Rust, and a silent typo there would weaken a safety test while leaving it
   green -- the test would still pass, just no longer testing the real model's
   ranking. Comparing the two sources makes that impossible.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = (
    ROOT / "benchmarks" / "unarchitectured-metal" / "hint-disagreements-2026-08-24.json"
)
SEARCH_RS = ROOT / "unchessed-core" / "src" / "search.rs"

# Rust test function -> artifact entry it was generated from.
RUST_FIXTURES = {
    "real_checkpoint_ranking_cannot_suppress_back_rank_mate": "back_rank_mate_white",
    "real_checkpoint_ranking_keeps_middlegame_search_sound": "greek_gift_setup",
}

PAIR = re.compile(r'\("([a-h][1-8][a-h][1-8][qrbn]?)",\s*(-?\d+\.\d+)')


def load_artifact():
    return {r["name"]: r for r in json.loads(ARTIFACT.read_text())}


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.entries = load_artifact()

    def test_artifact_exists_and_is_non_empty(self):
        self.assertTrue(self.entries)

    def test_positions_and_moves_are_legal(self):
        for name, r in self.entries.items():
            with self.subTest(name=name):
                board = chess.Board(r["fen"])
                move = chess.Move.from_uci(r["correct_move"])
                self.assertIn(move, board.legal_moves)
                self.assertEqual(len(list(board.legal_moves)), r["legal_count"])

    def test_mate_claims_are_true_and_unique(self):
        """`correct_is_mate` must be re-derivable, and the mate must be sole."""
        for name, r in self.entries.items():
            with self.subTest(name=name):
                board = chess.Board(r["fen"])
                probe = board.copy()
                probe.push(chess.Move.from_uci(r["correct_move"]))
                self.assertEqual(probe.is_checkmate(), r["correct_is_mate"])

                mates = []
                for mv in list(board.legal_moves):
                    board.push(mv)
                    if board.is_checkmate():
                        mates.append(mv.uci())
                    board.pop()
                if r["correct_is_mate"]:
                    self.assertEqual(mates, [r["correct_move"]])
                else:
                    self.assertEqual(mates, [])

    def test_recorded_ranking_is_self_consistent(self):
        """rank, top move and gap must follow from the logits themselves."""
        for name, r in self.entries.items():
            with self.subTest(name=name):
                moves, logits = r["moves"], r["logits"]
                self.assertEqual(len(moves), r["legal_count"])
                self.assertEqual(len(logits), r["legal_count"])

                order = sorted(range(len(logits)), key=lambda i: -logits[i])
                ci = moves.index(r["correct_move"])
                self.assertEqual(order.index(ci), r["correct_rank"])
                self.assertEqual(moves[order[0]], r["model_top_move"])
                self.assertEqual(r["disagrees"], r["correct_rank"] != 0)
                self.assertAlmostEqual(
                    r["model_top_logit"] - r["correct_logit"],
                    r["logit_gap"],
                    places=5,
                )

    def test_at_least_one_real_disagreement_exists(self):
        """The whole point of the artifact is the disagreements."""
        self.assertTrue([r for r in self.entries.values() if r["disagrees"]])


class RustFixtureSyncTests(unittest.TestCase):
    """The hand-transcribed Rust logits must equal the recorded ones."""

    def setUp(self):
        self.entries = load_artifact()
        self.src = SEARCH_RS.read_text()

    def _scored_block(self, fn_name: str) -> dict[str, float]:
        start = self.src.index(fn_name)
        arr = self.src.index("let scored = [", start)
        end = self.src.index("];", arr)
        return {m: float(v) for m, v in PAIR.findall(self.src[arr:end])}

    def test_rust_tests_are_present(self):
        for fn in RUST_FIXTURES:
            self.assertIn(fn, self.src, f"{fn} missing from search.rs")

    def test_rust_logits_match_the_artifact(self):
        for fn, entry in RUST_FIXTURES.items():
            with self.subTest(fixture=fn):
                rust = self._scored_block(fn)
                r = self.entries[entry]
                expected = dict(zip(r["moves"], r["logits"]))
                self.assertEqual(
                    set(rust), set(expected), "move set differs from the artifact"
                )
                for move, value in expected.items():
                    self.assertAlmostEqual(rust[move], value, places=6, msg=move)

    def test_rust_fixtures_are_genuine_disagreements(self):
        """A fixture where the model already agrees would test nothing."""
        for fn, entry in RUST_FIXTURES.items():
            with self.subTest(fixture=fn):
                self.assertTrue(self.entries[entry]["disagrees"])


if __name__ == "__main__":
    unittest.main()
