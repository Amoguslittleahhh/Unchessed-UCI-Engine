"""Tests for the policy-prior calibration finding.

This is a *positive* result, which carries its own risk: a calibration metric
can look good for uninteresting reasons. Two in particular are guarded here.

1. **ECE is dominated by the majority bin.** 17,361 of 17,998 moves sit below
   0.1 predicted probability, so a model that assigned near-zero to everything
   would score a fine ECE while being useless. The tests therefore also check
   the confidence *separation* between correct and incorrect top-1 picks,
   which that degenerate model would fail.

2. **Separation could be a legal-move-count artifact.** Positions with few
   legal moves give more softmax mass per move and are easier to get right.
   The doc records the uniform-normalised comparison that rules this out, and
   a test pins that it stays recorded.

The softmax and reliability maths are also checked directly against
closed-form expectations, since a bug there would silently move every number
in the artifact.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyse_policy_prior_calibration import (  # noqa: E402
    N_BINS,
    nll,
    reliability,
    softmax,
)

ARTIFACT = (
    ROOT
    / "benchmarks"
    / "unarchitectured-metal"
    / "policy-prior-calibration-2026-08-25.json"
)
DOC = ROOT / "docs" / "policy-prior-calibration.md"


class SoftmaxTests(unittest.TestCase):
    def test_sums_to_one(self):
        for logits in ([0.0], [1.0, 2.0, 3.0], [-5.0, 5.0]):
            with self.subTest(logits=logits):
                self.assertAlmostEqual(sum(softmax(logits)), 1.0, places=9)

    def test_uniform_logits_give_uniform_probabilities(self):
        probs = softmax([2.0] * 5)
        for p in probs:
            self.assertAlmostEqual(p, 0.2, places=9)

    def test_order_is_preserved(self):
        probs = softmax([0.0, 1.0, 2.0])
        self.assertLess(probs[0], probs[1])
        self.assertLess(probs[1], probs[2])

    def test_low_temperature_sharpens(self):
        """T < 1 must concentrate mass, which is the direction we recommend."""
        base = softmax([0.0, 1.0], 1.0)
        sharp = softmax([0.0, 1.0], 0.5)
        self.assertGreater(max(sharp), max(base))

    def test_high_temperature_flattens(self):
        base = softmax([0.0, 1.0], 1.0)
        soft = softmax([0.0, 1.0], 4.0)
        self.assertLess(max(soft), max(base))

    def test_is_shift_invariant(self):
        a = softmax([1.0, 2.0, 3.0])
        b = softmax([101.0, 102.0, 103.0])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)


class ReliabilityTests(unittest.TestCase):
    def test_perfect_predictor_has_zero_error(self):
        """A logit set that always makes the best move ~certain."""
        samples = [([10.0, -10.0, -10.0], 0)] * 20
        _rows, ece = reliability(samples)
        self.assertLess(ece, 0.05)

    def test_inverted_predictor_has_large_error(self):
        """Confidently wrong must score badly."""
        samples = [([10.0, -10.0, -10.0], 1)] * 20
        _rows, ece = reliability(samples)
        self.assertGreater(ece, 0.3)

    def test_bins_partition_the_samples(self):
        samples = [([1.0, 0.5, 0.0], 0), ([0.0, 1.0, 2.0], 2)]
        rows, _ece = reliability(samples)
        self.assertEqual(sum(r["count"] for r in rows), 6)
        for r in rows:
            self.assertLessEqual(r["bin_low"], r["mean_predicted"])
            self.assertLessEqual(r["mean_predicted"], r["bin_high"] + 1e-9)
        self.assertLessEqual(len(rows), N_BINS)

    def test_nll_rewards_the_right_answer(self):
        confident_right = nll([([5.0, 0.0], 0)], 1.0)
        confident_wrong = nll([([5.0, 0.0], 1)], 1.0)
        self.assertLess(confident_right, confident_wrong)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(ARTIFACT.read_text())

    def test_covers_the_corpus(self):
        self.assertEqual(self.report["positions"], 600)

    def test_calibration_is_good(self):
        self.assertLess(self.report["ece_raw"], 0.02)

    def test_tempering_improves_calibration(self):
        self.assertLessEqual(
            self.report["ece_tempered"], self.report["ece_raw"] + 1e-9
        )
        self.assertLessEqual(
            self.report["nll_at_optimum"], self.report["nll_at_one"] + 1e-9
        )

    def test_model_is_underconfident_not_overconfident(self):
        """The unusual direction, and the reason T < 1."""
        self.assertLess(self.report["optimal_temperature"], 1.0)

    def test_confidence_separates_correct_from_wrong(self):
        """Guards against a degenerate near-zero-everywhere predictor.

        Such a model would post a fine ECE (the low bin dominates) while
        carrying no usable signal. It would fail this.
        """
        self.assertGreater(
            self.report["top1_confidence_correct"],
            self.report["top1_confidence_wrong"] * 1.3,
        )

    def test_both_outcome_classes_are_populated(self):
        self.assertGreater(self.report["n_correct"], 50)
        self.assertGreater(self.report["n_wrong"], 50)
        self.assertEqual(
            self.report["n_correct"] + self.report["n_wrong"],
            self.report["positions"],
        )

    def test_reliability_curve_is_recorded(self):
        self.assertTrue(self.report["reliability"])
        for row in self.report["reliability"]:
            with self.subTest(bin=row["bin_low"]):
                self.assertAlmostEqual(
                    row["gap"], row["actual"] - row["mean_predicted"], places=9
                )


class DocTests(unittest.TestCase):
    def test_numbers_match_the_artifact(self):
        text = DOC.read_text()
        r = json.loads(ARTIFACT.read_text())
        self.assertIn(f"{r['ece_raw']:.4f}", text)
        self.assertIn(f"{r['optimal_temperature']:.2f}", text)

    def test_doc_records_the_confound_check(self):
        """The legal-move-count normalisation must stay documented."""
        text = DOC.read_text()
        self.assertIn("uniform", text.lower())
        self.assertIn("6.21", text)
        self.assertIn("4.26", text)

    def test_doc_does_not_claim_the_hint_is_revived(self):
        text = DOC.read_text()
        self.assertIn("does not resurrect the root hint", text)

    def test_doc_rejects_mcts(self):
        self.assertIn("not applicable", DOC.read_text().lower())


if __name__ == "__main__":
    unittest.main()
