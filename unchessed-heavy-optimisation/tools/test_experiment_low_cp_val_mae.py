#!/usr/bin/env python3
"""Sanity checks for the low-cp val-MAE / persona experiment harness."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_low_cp_val_mae import bayes_mae_gaussian, sigmoid, wdl_p25


class BayesFloorTests(unittest.TestCase):
    def test_gaussian_mae_formula(self):
        self.assertAlmostEqual(bayes_mae_gaussian(70), 70 * math.sqrt(2 / math.pi), places=9)
        self.assertGreater(bayes_mae_gaussian(70), 50)
        self.assertLess(bayes_mae_gaussian(12), 10)

    def test_sub20_requires_low_sigma(self):
        self.assertGreater(bayes_mae_gaussian(70), 20)
        self.assertLess(bayes_mae_gaussian(20), 20)


class LossTests(unittest.TestCase):
    def test_sigmoid_bounds(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(10), 0.99)
        self.assertLess(sigmoid(-10), 0.01)

    def test_wdl_zero_at_perfect(self):
        t = 0.5
        raw = 0.0
        self.assertLess(wdl_p25(raw, t), 1e-4)


class ArtifactTests(unittest.TestCase):
    def test_committed_artifact_schema(self):
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "artifacts" / "low-cp-val-mae-persona-experiments.json"
        if not p.exists():
            self.skipTest("run tools/experiment_low_cp_val_mae.py first")
        data = json.loads(p.read_text())
        self.assertEqual(data["go_nogo"]["sub_20_cp_with_current_hce_5000_node_labels"], "NO-GO")
        self.assertTrue(data["go_nogo"]["persona_remains_active"])
        self.assertGreater(data["recipes"][0]["affine_true_fit_mae_cp"], 20)
        self.assertLess(data["recipes"][3]["affine_true_fit_mae_cp"], 20)


if __name__ == "__main__":
    unittest.main()
