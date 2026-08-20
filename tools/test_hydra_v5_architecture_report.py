#!/usr/bin/env python3

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "hydra_v5_architecture_report.py"
SPEC = importlib.util.spec_from_file_location("hydra_v5_architecture_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ROOT = TOOLS.parent
CONFIG = json.loads((ROOT / "config/unchessed_hydra_v5.json").read_text())
STUDENT = json.loads((ROOT / "config/unchessed_hydra_v4.json").read_text())
TRAINING = json.loads((ROOT / "config/a100_hydra_v5_training.json").read_text())
PROFILES = json.loads((ROOT / "config/verda_gpu_profiles.json").read_text())


class HydraV5ArchitectureReportTests(unittest.TestCase):
    def test_oracle_budget_is_large_training_only_teacher(self):
        MODULE.validate_config(CONFIG)
        budget = MODULE.oracle_budget(CONFIG["offline_oracle"])
        self.assertEqual(budget["parameters"], 58_412_431)
        self.assertGreater(budget["forward_flops_approx"], 11_000_000_000)
        self.assertFalse(CONFIG["offline_oracle"]["runtime_deployed"])

    def test_runtime_student_stays_compact(self):
        report = MODULE.build_report(CONFIG, STUDENT)
        self.assertEqual(report["runtime_student"]["parameters"], 4_222_905)
        self.assertGreater(report["parameter_compression_factor"], 13.8)
        self.assertTrue(CONFIG["runtime_student"]["teacher_free_inference"])

    def test_180_core_topology_and_a100_reserve_are_explicit(self):
        report = MODULE.build_report(CONFIG, STUDENT)
        self.assertEqual(report["cpu_datagen"]["node_vcpus"], 180)
        self.assertEqual(report["cpu_datagen"]["workers"], 176)
        self.assertEqual(report["cpu_datagen"]["reserved_service_vcpus"], 4)
        self.assertEqual(report["cpu_datagen"]["aggregate_hash_mib"], 11_264)
        self.assertAlmostEqual(report["a100"]["target_vram_fraction"], 0.92)
        self.assertAlmostEqual(report["a100"]["minimum_free_vram_fraction"], 0.08)
        self.assertFalse(report["a100"]["fp8_allowed"])

    def test_training_losses_are_normalized(self):
        self.assertAlmostEqual(sum(TRAINING["oracle"]["loss_weights"].values()), 1.0)
        self.assertAlmostEqual(
            sum(TRAINING["student_distillation"]["loss_weights"].values()), 1.0
        )

    def test_verda_profile_matrix_scales_the_training_only_oracle(self):
        matrix = MODULE.verda_profile_matrix(PROFILES, TRAINING)
        counts = [entry["oracle_parameters"] for entry in matrix]
        self.assertEqual(
            counts,
            [878_114_575, 501_835_855, 230_537_295, 58_412_431, 29_144_367],
        )
        self.assertEqual(matrix[-1]["precision"], "float16")

    def test_rejects_runtime_oracle_and_unsafe_pruning(self):
        broken = copy.deepcopy(CONFIG)
        broken["offline_oracle"]["runtime_deployed"] = True
        with self.assertRaisesRegex(ValueError, "training-only"):
            MODULE.validate_config(broken)
        broken = copy.deepcopy(CONFIG)
        broken["safety"]["noncandidate_pruning_allowed"] = True
        with self.assertRaisesRegex(ValueError, "fallback"):
            MODULE.validate_config(broken)

    def test_rejects_fake_a100_fp8_claim(self):
        broken = copy.deepcopy(CONFIG)
        broken["a100_training"]["fp8_allowed"] = True
        with self.assertRaisesRegex(ValueError, "FP8"):
            MODULE.validate_config(broken)


if __name__ == "__main__":
    unittest.main()
