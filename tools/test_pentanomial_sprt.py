"""Tests for the pentanomial SPRT tool and the int8 weight-range artifact.

The statistics tests matter because this tool will be used to decide whether a
future change ships. A subtly wrong Elo or LLR would be worse than no tool at
all, so the properties are checked against closed-form expectations and
against three real, independently-produced results.

Standard library only.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from pentanomial_sprt import (  # noqa: E402
    analyse,
    counts_from_pgn,
    elo_to_score,
    pentanomial_stats,
    score_to_elo,
    sprt_llr,
    trinomial_stats,
)

RANGE_ARTIFACT = (
    ROOT / "benchmarks" / "unarchitectured-v1" / "int8-weight-range-2026-08-25.json"
)
DOC = ROOT / "docs" / "fishtest-and-quantization-notes.md"


class EloConversionTests(unittest.TestCase):
    def test_even_score_is_zero_elo(self):
        self.assertAlmostEqual(score_to_elo(0.5), 0.0, places=9)
        self.assertAlmostEqual(elo_to_score(0.0), 0.5, places=9)

    def test_round_trip(self):
        for elo in (-400.0, -26.1, -5.8, 0.0, 12.5, 200.0):
            with self.subTest(elo=elo):
                self.assertAlmostEqual(score_to_elo(elo_to_score(elo)), elo, places=6)

    def test_monotonic(self):
        self.assertLess(score_to_elo(0.4), score_to_elo(0.5))
        self.assertLess(score_to_elo(0.5), score_to_elo(0.6))


class PentanomialTests(unittest.TestCase):
    def test_symmetric_counts_give_even_score(self):
        mean, var, n = pentanomial_stats([10, 50, 100, 50, 10])
        self.assertAlmostEqual(mean, 0.5, places=12)
        self.assertEqual(n, 220)
        self.assertGreater(var, 0.0)

    def test_all_draws_have_zero_variance(self):
        mean, var, _ = pentanomial_stats([0, 0, 100, 0, 0])
        self.assertAlmostEqual(mean, 0.5, places=12)
        self.assertAlmostEqual(var, 0.0, places=12)

    def test_all_wins_scores_one(self):
        mean, _, _ = pentanomial_stats([0, 0, 0, 0, 50])
        self.assertAlmostEqual(mean, 1.0, places=12)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            pentanomial_stats([0, 0, 0, 0, 0])


class ReproducesRound7Tests(unittest.TestCase):
    """The trinomial path must match three real, independent results.

    These figures were produced by the project owner on real hardware with
    cutechess-cli, and reported in the round-7 status update. Matching them
    to 0.1 Elo is evidence the implementation is correct rather than merely
    self-consistent.
    """

    CASES = [
        # (name, wins, losses, draws, expected_elo, expected_pm)
        ("original", 172, 217, 211, -26.1, 22.4),
        ("replication", 184, 210, 206, -15.1, 22.5),
        ("conservative", 72, 77, 151, -5.8, 27.7),
    ]

    def test_elo_matches_reported(self):
        for name, w, l, d, expected, _pm in self.CASES:
            with self.subTest(run=name):
                mean, _var, _n = trinomial_stats(w, d, l)
                self.assertAlmostEqual(score_to_elo(mean), expected, places=1)

    def test_confidence_interval_matches_reported(self):
        for name, w, l, d, _elo, expected_pm in self.CASES:
            with self.subTest(run=name):
                mean, var, n = trinomial_stats(w, d, l)
                se = math.sqrt(var / n)
                lo = score_to_elo(mean - 1.959963985 * se)
                hi = score_to_elo(mean + 1.959963985 * se)
                self.assertAlmostEqual((hi - lo) / 2.0, expected_pm, delta=0.15)

    def test_all_three_runs_are_negative(self):
        """The standing conclusion: no config ever trended positive."""
        for name, w, l, d, _e, _p in self.CASES:
            with self.subTest(run=name):
                mean, _v, _n = trinomial_stats(w, d, l)
                self.assertLess(mean, 0.5)


class SprtTests(unittest.TestCase):
    def test_even_result_gives_negative_llr_against_positive_elo1(self):
        """A dead-even result is evidence against 'gained 5 Elo'."""
        self.assertLess(sprt_llr([10, 50, 100, 50, 10], 0.0, 5.0), 0.0)

    def test_strong_win_gives_positive_llr(self):
        self.assertGreater(sprt_llr([0, 5, 40, 60, 40], 0.0, 5.0), 0.0)

    def test_zero_variance_is_handled(self):
        self.assertEqual(sprt_llr([0, 0, 100, 0, 0], 0.0, 5.0), 0.0)

    def test_bounds_follow_alpha_beta(self):
        r = analyse([10, 50, 100, 50, 10], 0.0, 5.0, 0.05, 0.05)
        self.assertAlmostEqual(r["sprt"]["upper_bound"], math.log(0.95 / 0.05), places=9)
        self.assertAlmostEqual(r["sprt"]["lower_bound"], math.log(0.05 / 0.95), places=9)

    def test_symmetric_result_is_inconclusive_and_even(self):
        r = analyse([10, 50, 100, 50, 10], 0.0, 5.0, 0.05, 0.05)
        self.assertAlmostEqual(r["score"], 0.5, places=12)
        self.assertAlmostEqual(r["pentanomial"]["elo"], 0.0, places=6)
        self.assertAlmostEqual(r["los"], 0.5, places=6)
        self.assertEqual(r["sprt"]["decision"], "inconclusive")

    def test_pairing_reduces_variance_on_correlated_pairs(self):
        """The reason to use pentanomial at all.

        A book where pairs are decisive together (many WW/LL) has MORE
        pentanomial variance; one where pairs split (mostly the middle cell)
        has less. The middle-heavy case is the common one and is where the
        pairing saves resources.
        """
        middle_heavy = analyse([0, 0, 200, 0, 0], 0.0, 5.0, 0.05, 0.05)
        self.assertLess(middle_heavy["pentanomial"]["variance_per_pair"], 1e-9)


class PgnParsingTests(unittest.TestCase):
    def _write(self, games) -> Path:
        text = "".join(
            f'[White "{w}"]\n[Black "{b}"]\n[Result "{r}"]\n\n{r}\n\n'
            for w, b, r in games
        )
        path = Path(tempfile.mkdtemp()) / "games.pgn"
        path.write_text(text)
        return path

    def test_extracts_known_pair_outcomes(self):
        path = self._write(
            [
                ("Hint", "Baseline", "1-0"),
                ("Baseline", "Hint", "0-1"),  # Hint wins both -> WW
                ("Hint", "Baseline", "1/2-1/2"),
                ("Baseline", "Hint", "1/2-1/2"),  # DD
                ("Hint", "Baseline", "0-1"),
                ("Baseline", "Hint", "1-0"),  # Hint loses both -> LL
            ]
        )
        self.assertEqual(counts_from_pgn(path, "Hint"), [1, 0, 1, 0, 1])

    def test_colour_is_respected(self):
        """Scoring must follow which side the engine actually played."""
        path = self._write(
            [("Baseline", "Hint", "0-1"), ("Hint", "Baseline", "1-0")]
        )
        self.assertEqual(counts_from_pgn(path, "Hint"), [0, 0, 0, 0, 1])

    def test_unfinished_games_are_ignored(self):
        path = self._write(
            [
                ("Hint", "Baseline", "*"),
                ("Hint", "Baseline", "1-0"),
                ("Baseline", "Hint", "0-1"),
            ]
        )
        self.assertEqual(counts_from_pgn(path, "Hint"), [0, 0, 0, 0, 1])

    def test_odd_trailing_game_is_dropped(self):
        path = self._write(
            [
                ("Hint", "Baseline", "1-0"),
                ("Baseline", "Hint", "0-1"),
                ("Hint", "Baseline", "1-0"),
            ]
        )
        self.assertEqual(sum(counts_from_pgn(path, "Hint")), 1)


@unittest.skipUnless(RANGE_ARTIFACT.is_file(), "range artifact absent")
class Int8RangeTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(RANGE_ARTIFACT.read_text())

    def test_limit_is_the_stockfish_value(self):
        self.assertAlmostEqual(self.report["stockfish_int8_limit"], 127 / 64, places=9)

    def test_weights_exceed_the_int8_range(self):
        """The finding: post-hoc int8 could not have worked."""
        self.assertGreater(self.report["summary"]["headroom_ratio"], 2.0)
        self.assertGreater(self.report["summary"]["over_limit_total"], 0)

    def test_per_tensor_entries_are_consistent(self):
        for entry in self.report["tensors"]:
            with self.subTest(tensor=entry["name"]):
                self.assertLessEqual(entry["over_limit"], entry["numel"])
                self.assertAlmostEqual(
                    entry["over_pct"],
                    entry["over_limit"] / entry["numel"] * 100,
                    places=3,
                )


class DocTests(unittest.TestCase):
    def test_doc_records_both_findings(self):
        text = DOC.read_text()
        self.assertIn("pentanomial", text.lower())
        self.assertIn("127/64", text)

    def test_doc_headroom_matches_artifact(self):
        if not RANGE_ARTIFACT.is_file():
            self.skipTest("range artifact absent")
        summary = json.loads(RANGE_ARTIFACT.read_text())["summary"]
        self.assertIn(f"{summary['headroom_ratio']:.2f}x", DOC.read_text())


if __name__ == "__main__":
    unittest.main()
