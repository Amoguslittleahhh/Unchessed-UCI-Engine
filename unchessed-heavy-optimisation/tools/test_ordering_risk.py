"""Tests for the ordering-risk analysis.

These guard the two things that would silently invalidate the conclusion in
`docs/unarchitectured-metal-why-the-hint-costs-elo.md`:

1. The **baseline** must stay movegen order. The whole argument rests on
   comparing the hint against what it actually displaces. `go_with_root_hints`
   applies the hint only at `depth == start_depth`, where every root score is
   still `-MATE` so the fallback sort is a no-op. If that structure ever
   changes, this analysis is measuring the wrong thing and the doc becomes
   wrong -- so the structure itself is asserted here against the real source.

2. The **artifact must stay self-consistent**, and its headline claim (the
   neural policy is a better orderer than the real baseline on every metric)
   must actually hold in the committed numbers rather than only in the prose.

No torch needed: these read the committed artifact and the Rust source.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "benchmarks" / "unarchitectured-metal" / "ordering-risk-2026-08-24.json"
SEARCH_RS = ROOT / "unchessed-core" / "src" / "search.rs"
DOC = ROOT / "docs" / "unarchitectured-metal-why-the-hint-costs-elo.md"

# Metrics where a LOWER value is better.
LOWER_IS_BETTER = (
    "mean_first_move_regret_cp",
    "median_first_move_regret_cp",
    "p90_first_move_regret_cp",
    "blunder_rate_200cp",
    "disaster_rate_500cp",
    "mean_best_move_rank",
    "median_best_move_rank",
    "best_move_in_bottom_half_rate",
    "confident_blunder_rate",
)
# Metrics where a HIGHER value is better.
HIGHER_IS_BETTER = (
    "top1_accuracy",
    "best_move_in_top3_rate",
)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(ARTIFACT.read_text())

    def test_covers_the_whole_corpus(self):
        self.assertEqual(self.report["positions_scored"], 600)
        self.assertEqual(self.report["positions_skipped"], 0)

    def test_all_three_orderings_present(self):
        for key in ("neural", "heuristic", "movegen"):
            self.assertIn(key, self.report)
            self.assertEqual(self.report[key]["positions"], 600)

    def test_neural_beats_the_real_baseline_on_every_metric(self):
        """The headline claim, asserted against the numbers not the prose."""
        neural, movegen = self.report["neural"], self.report["movegen"]
        for metric in LOWER_IS_BETTER:
            with self.subTest(metric=metric):
                self.assertLess(neural[metric], movegen[metric])
        for metric in HIGHER_IS_BETTER:
            with self.subTest(metric=metric):
                self.assertGreater(neural[metric], movegen[metric])

    def test_neural_also_beats_mvv_lva(self):
        neural, heur = self.report["neural"], self.report["heuristic"]
        self.assertLess(neural["mean_first_move_regret_cp"], heur["mean_first_move_regret_cp"])
        self.assertLess(neural["blunder_rate_200cp"], heur["blunder_rate_200cp"])
        self.assertGreater(neural["top1_accuracy"], heur["top1_accuracy"])

    def test_confident_cases_are_not_worse_than_average(self):
        """Falsifies the 'confidently wrong' hypothesis explicitly."""
        neural = self.report["neural"]
        self.assertLessEqual(
            neural["confident_mean_regret_cp"],
            neural["mean_first_move_regret_cp"] * 1.05,
        )

    def test_worst_cases_are_recorded(self):
        self.assertTrue(self.report["worst_cases"])
        for case in self.report["worst_cases"]:
            self.assertGreaterEqual(case["regret_cp"], 0)
            self.assertIn("fen", case)


class SearchStructureTests(unittest.TestCase):
    """The baseline assumption must remain true of the real source."""

    def setUp(self):
        self.src = SEARCH_RS.read_text()

    def test_hint_applies_only_to_the_first_pass(self):
        self.assertIn(
            "if depth == start_depth && !root_hints.is_empty()",
            self.src,
            "hint is no longer scoped to the first pass; the ordering-risk "
            "analysis compares against the wrong baseline",
        )

    def test_root_scores_start_at_negative_mate(self):
        """Why the fallback sort is a no-op on the first pass."""
        block = self.src[self.src.index("struct RootMove") :][:1200]
        self.assertRegex(block, r"score:\s*-MATE")

    def test_fallback_sort_is_by_score(self):
        self.assertIn("roots.sort_by_key(|root| -root.score)", self.src)

    def test_preprocessing_is_charged_to_the_deadline(self):
        """The cost half of the explanation."""
        self.assertIn("preprocessing_elapsed", self.src)
        self.assertIn(
            "checked_sub(preprocessing_elapsed)",
            self.src,
            "hint cost is no longer charged against the move budget",
        )


class DocTests(unittest.TestCase):
    def test_doc_exists_and_states_the_falsification(self):
        text = DOC.read_text()
        self.assertIn("falsified", text.lower())

    def test_doc_numbers_match_the_artifact(self):
        """Headline figures in the table must come from the committed run."""
        text = DOC.read_text()
        report = json.loads(ARTIFACT.read_text())
        for value in (
            f"{report['neural']['top1_accuracy']:.4f}",
            f"{report['neural']['mean_first_move_regret_cp']:.1f}",
            f"{report['movegen']['mean_best_move_rank']:.2f}",
        ):
            self.assertIn(value, text, f"{value} missing from the doc table")

    def test_doc_keeps_the_default_off_conclusion(self):
        text = DOC.read_text().lower()
        self.assertIn("remains unjustified", text)


class ReferencedPathTests(unittest.TestCase):
    def test_every_repo_path_mentioned_in_the_doc_resolves(self):
        text = DOC.read_text()
        for match in re.findall(r"`((?:tools|docs|benchmarks|unchessed-core)/[^`]+)`", text):
            with self.subTest(path=match):
                self.assertTrue((ROOT / match).exists(), match)


if __name__ == "__main__":
    unittest.main()
