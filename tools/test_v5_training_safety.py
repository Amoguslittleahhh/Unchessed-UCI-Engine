#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG = json.loads((ROOT / "config/a100_hydra_v5_training.json").read_text())
SOURCE = (ROOT / "tools/train_hydra_oracle_v5_a100.py").read_text()
LAUNCHER = (ROOT / "scripts/training/a100_hydra_v5_train.sh").read_text()


class V5TrainingSafetyTests(unittest.TestCase):
    def test_epochs_are_dataset_sized_not_arbitrary_step_counts(self):
        for section in (CONFIG["oracle"], CONFIG["student_distillation"]):
            self.assertEqual(section["epoch_mode"], "without_replacement")
            self.assertNotIn("steps_per_epoch", section)
            self.assertGreaterEqual(section["minimum_optimizer_steps_per_epoch"], 100)
            self.assertGreaterEqual(section["minimum_validation_records"], 50_000)
        self.assertIn("class EpochRecordPrefetcher", SOURCE)
        self.assertIn("rng.permutation(shards.total)", SOURCE)
        self.assertIn('"sampling": "global_without_replacement"', SOURCE)

    def test_overfitting_stops_early(self):
        for section in (CONFIG["oracle"], CONFIG["student_distillation"]):
            self.assertLessEqual(section["early_stopping_patience"], 3)
            self.assertGreater(section["early_stopping_min_delta"], 0)
        self.assertIn("epochs_without_improvement", SOURCE)
        self.assertIn("if early_stop:", SOURCE)

    def test_cuda_graphs_are_disabled_and_memory_growth_is_guarded(self):
        self.assertTrue(CONFIG["hardware"]["disable_cudagraphs"])
        self.assertEqual(CONFIG["hardware"]["compile_mode"], "default")
        self.assertIn("TORCHINDUCTOR_CUDAGRAPHS", LAUNCHER)
        self.assertIn("reserved_memory_growth_exceeded", SOURCE)
        self.assertIn("CUDA reserved-memory growth guard triggered", SOURCE)

    def test_split_and_metric_fail_closed_contracts_exist(self):
        self.assertIn("require_distinct_shards", SOURCE)
        self.assertIn("require_finite_metrics", SOURCE)
        self.assertIn("minimum_records", SOURCE)

    def test_student_calibration_precedes_final_holdout(self):
        self.assertIn('"calibrate-student"', SOURCE)
        self.assertIn("fit_regret_calibration", SOURCE)
        calibrate = LAUNCHER.index("calibrate-student")
        final = LAUNCHER.index("evaluate-student")
        self.assertLess(calibrate, final)
        self.assertIn("student-v5.calibrated.pt", LAUNCHER)


if __name__ == "__main__":
    unittest.main()
