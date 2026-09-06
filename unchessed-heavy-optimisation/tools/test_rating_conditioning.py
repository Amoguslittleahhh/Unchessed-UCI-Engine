"""Tests for the rating-conditioning finding.

The claim is a negative one -- an input does nothing -- which is exactly the
kind of result that can be produced by a broken measurement rather than a
broken model. So these tests check that the *experiment* was capable of
detecting an effect, not just that it reported none:

  - the sweep actually spanned a wide rating range;
  - the tool varies the input it claims to vary;
  - the recorded deltas are non-zero (the input reaches the output at all, it
    is simply too weak to matter) rather than identically zero, which would
    instead suggest the rating was never plumbed through.

No torch needed.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = (
    ROOT / "benchmarks" / "unarchitectured-metal" / "rating-conditioning-2026-08-25.json"
)
TOOL = ROOT / "tools" / "analyse_rating_conditioning.py"
DOC = ROOT / "docs" / "rating-conditioning-finding.md"
UCI_RS = ROOT / "unchessed-core" / "src" / "uci.rs"


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(ARTIFACT.read_text())
        self.ratings = {int(k): v for k, v in self.report["ratings"].items()}

    def test_sweep_spans_a_wide_range(self):
        """A narrow sweep could miss a real effect."""
        self.assertLessEqual(min(self.ratings), 600)
        self.assertGreaterEqual(max(self.ratings), 3200)
        self.assertGreaterEqual(len(self.ratings), 5)

    def test_enough_positions(self):
        self.assertGreaterEqual(self.report["positions"], 100)

    def test_rating_never_changes_the_move(self):
        """The finding."""
        for rating, cell in self.ratings.items():
            with self.subTest(rating=rating):
                self.assertEqual(cell["top1_changed"], 0)

    def test_teacher_agreement_is_identical_across_ratings(self):
        """Independent confirmation: accuracy does not move either."""
        values = {
            round(c["teacher_agreement"], 6)
            for c in self.ratings.values()
            if c["teacher_agreement"] is not None
        }
        self.assertEqual(len(values), 1, f"agreement varied: {values}")

    def test_the_input_does_reach_the_output(self):
        """Deltas must be non-zero, or the measurement is the suspect.

        If every delta were exactly 0 the likely explanation would be that
        the rating never reached the network -- a bug in the harness. Small
        but non-zero deltas show the plumbing works and the signal is simply
        too weak to change any decision.
        """
        top = max(self.ratings)
        self.assertGreater(self.ratings[top]["max_abs_delta"], 0.0)

    def test_deltas_are_negligible_against_real_logit_gaps(self):
        """Two orders of magnitude below the gaps between candidate moves."""
        top = max(self.ratings)
        self.assertLess(self.ratings[top]["max_abs_delta"], 0.01)

    def test_deltas_grow_monotonically_with_rating(self):
        """Consistent with one scalar term scaled by the rating."""
        ordered = sorted(self.ratings)
        deltas = [self.ratings[r]["max_abs_delta"] for r in ordered]
        self.assertEqual(deltas, sorted(deltas))

    def test_policy_kind_is_recorded_and_weak(self):
        kind = self.report["policy_kind"]
        self.assertGreater(kind["scored"], 0)
        self.assertLess(kind["changed_pct"], 25.0)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.src = TOOL.read_text()

    def test_tool_varies_the_rating_input(self):
        self.assertIn('"rating": torch.tensor([rating]', self.src)

    def test_tool_varies_policy_kind(self):
        self.assertIn('"policy_kind": torch.tensor([policy_kind]', self.src)

    def test_default_sweep_covers_the_maia_ladder(self):
        match = re.search(r"DEFAULT_RATINGS = \[(.*?)\]", self.src)
        self.assertIsNotNone(match)
        values = [int(v) for v in re.findall(r"\d+", match.group(1))]
        self.assertLessEqual(min(values), 600)
        self.assertGreaterEqual(max(values), 2600)

    def test_help_needs_no_dependencies(self):
        """--help must work from a bare clone."""
        self.assertIn('if any(a in ("-h", "--help") for a in sys.argv[1:])', self.src)


class ScopeTests(unittest.TestCase):
    """The finding must not be overstated: UCI_Elo itself still works."""

    def test_uci_elo_option_still_exists(self):
        self.assertIn("UCI_Elo", UCI_RS.read_text())

    def test_doc_distinguishes_the_adaptive_path(self):
        text = DOC.read_text()
        self.assertIn("does not mean `UCI_Elo` is broken", text)
        self.assertIn("adaptive", text.lower())

    def test_doc_numbers_match_the_artifact(self):
        report = json.loads(ARTIFACT.read_text())
        agreement = next(
            c["teacher_agreement"]
            for c in report["ratings"].values()
            if c["teacher_agreement"] is not None
        )
        self.assertIn(f"{agreement:.4f}", DOC.read_text())


if __name__ == "__main__":
    unittest.main()
