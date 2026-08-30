"""Unit tests for nnue_train_control.py — no torch required."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnue_train_control import (
    EarlyStop,
    lr_at_epoch,
    lr_drop_epochs,
    recipe_from_env,
)


class LrScheduleTests(unittest.TestCase):
    def test_production_15_epoch_drops(self):
        # scripts/nnue-pipeline/full_pipeline.sh passes 15. The historical
        # trainer used int(15*0.6)=9 and int(15*0.8)=12 as 0-based indices.
        self.assertEqual(lr_drop_epochs(15), (9, 12))
        for ep in range(9):
            self.assertAlmostEqual(lr_at_epoch(ep, 15), 1e-3)
        for ep in range(9, 12):
            self.assertAlmostEqual(lr_at_epoch(ep, 15), 3e-4)
        for ep in range(12, 15):
            self.assertAlmostEqual(lr_at_epoch(ep, 15), 9e-5)

    def test_diagnostic_8_epoch_drops(self):
        # The reviewer 9M/27M runs used 8 epochs: drops at int(4.8)=4 and
        # int(6.4)=6. Confirms the copied-constant schedule scales with the
        # *requested* cap, not with data size — the F-01 shape.
        self.assertEqual(lr_drop_epochs(8), (4, 6))

    def test_rejects_zero_epochs(self):
        with self.assertRaises(ValueError):
            lr_drop_epochs(0)


class EarlyStopTests(unittest.TestCase):
    def test_first_observation_is_best(self):
        s = EarlyStop(patience=3, min_delta=0.1)
        is_best, stop = s.update(57.4)
        self.assertTrue(is_best)
        self.assertFalse(stop)
        self.assertEqual(s.best_epoch, 1)
        self.assertEqual(s.best, 57.4)

    def test_improvement_resets_patience(self):
        s = EarlyStop(patience=3, min_delta=0.1)
        s.update(60.0)
        s.update(60.0)  # miss 1
        s.update(60.0)  # miss 2
        self.assertEqual(s.bad, 2)
        is_best, stop = s.update(59.5)  # 0.5cp better
        self.assertTrue(is_best)
        self.assertFalse(stop)
        self.assertEqual(s.bad, 0)
        self.assertEqual(s.best_epoch, 4)

    def test_min_delta_ignores_tiny_wiggles(self):
        s = EarlyStop(patience=3, min_delta=0.1)
        self.assertEqual(s.update(50.0), (True, False))
        # 0.05cp better — below min_delta, counts as a miss
        self.assertEqual(s.update(49.95), (False, False))
        self.assertEqual(s.bad, 1)
        self.assertEqual(s.best, 50.0)
        # 0.2cp better — counts
        self.assertEqual(s.update(49.80), (True, False))
        self.assertEqual(s.bad, 0)
        self.assertEqual(s.best_epoch, 3)

    def test_stops_after_patience_non_improvements(self):
        # The 27M diagnostic shape: val-MAE bottoms then climbs. With
        # patience=3 the best checkpoint is kept and the run stops three
        # epochs after the minimum, instead of exporting the last (worse) net.
        s = EarlyStop(patience=3, min_delta=0.1)
        seq = [55.3, 51.1, 51.8, 52.9, 54.3]
        results = [s.update(m) for m in seq]
        self.assertEqual(
            results,
            [
                (True, False),   # 55.3 best
                (True, False),   # 51.1 best
                (False, False),  # miss 1
                (False, False),  # miss 2
                (False, True),   # miss 3 -> stop
            ],
        )
        self.assertEqual(s.best_epoch, 2)
        self.assertEqual(s.best, 51.1)

    def test_patience_zero_never_stops_but_still_tracks_best(self):
        s = EarlyStop(patience=0, min_delta=0.1)
        s.update(10.0)
        s.update(12.0)
        is_best, stop = s.update(13.0)
        self.assertFalse(is_best)
        self.assertFalse(stop)
        self.assertEqual(s.best_epoch, 1)
        self.assertEqual(s.best, 10.0)

    def test_nan_is_rejected(self):
        s = EarlyStop(patience=3)
        with self.assertRaises(ValueError):
            s.update(float("nan"))

    def test_climbing_from_first_epoch_still_exports_first(self):
        # 959k diagnostic: 57.4 -> 83.6. Best is epoch 1.
        s = EarlyStop(patience=3, min_delta=0.1)
        s.update(57.4)
        s.update(65.0)
        s.update(74.0)
        is_best, stop = s.update(83.6)
        self.assertFalse(is_best)
        self.assertTrue(stop)
        self.assertEqual(s.best_epoch, 1)


class RecipeEnvTests(unittest.TestCase):
    def test_defaults(self):
        os.environ.pop("EARLY_STOP_PATIENCE", None)
        os.environ.pop("EARLY_STOP_MIN_DELTA", None)
        patience, min_delta = recipe_from_env()
        self.assertEqual(patience, 3)
        self.assertEqual(min_delta, 0.1)

    def test_override_and_reject_negative(self):
        os.environ["EARLY_STOP_PATIENCE"] = "0"
        os.environ["EARLY_STOP_MIN_DELTA"] = "0.05"
        try:
            patience, min_delta = recipe_from_env()
            self.assertEqual(patience, 0)
            self.assertEqual(min_delta, 0.05)
            os.environ["EARLY_STOP_PATIENCE"] = "-1"
            with self.assertRaises(ValueError):
                recipe_from_env()
        finally:
            os.environ.pop("EARLY_STOP_PATIENCE", None)
            os.environ.pop("EARLY_STOP_MIN_DELTA", None)


if __name__ == "__main__":
    unittest.main()
