#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
ROOT = TOOLS.parent
MODULE_PATH = TOOLS / "unarchitectured_v1_architecture_report.py"
SPEC = importlib.util.spec_from_file_location(
    "unarchitectured_v1_architecture_report", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UnarchitecturedV1CanonicalTests(unittest.TestCase):
    def test_registry_and_runtime_magic(self):
        registry = json.loads((ROOT / "config/architecture_registry.json").read_text())
        canonical = registry["canonical"]
        self.assertEqual(canonical["id"], "unarchitectured-v1")
        self.assertEqual(canonical["name"], "Unarchitectured v1")
        self.assertEqual(canonical["runtime_magic"], "UNARCHV1")
        self.assertEqual(len(registry["experimental_predecessors"]), 6)

    def test_canonical_config_and_budget(self):
        config = json.loads((ROOT / "config/unarchitectured_v1.json").read_text())
        student = json.loads(
            (ROOT / "config/unarchitectured_v1_student.json").read_text()
        )
        profiles = json.loads((ROOT / "config/verda_gpu_profiles.json").read_text())
        training = json.loads(
            (ROOT / "config/unarchitectured_v1_training.json").read_text()
        )
        report = MODULE.build_report(config, student, profiles, training)
        self.assertEqual(report["schema"], 1)
        self.assertEqual(report["name"], "Unarchitectured v1")
        self.assertEqual(report["runtime_file_magic"], "UNARCHV1")
        self.assertEqual(report["runtime_student"]["parameters"], 4_222_905)
        self.assertTrue(report["autonomous_safety"]["external_watchdog"])
        self.assertTrue(report["feature_extraction"]["rust_gpu_schema_audit"])
        self.assertEqual(
            report["training_efficiency"]["epoch_mode"],
            "rotated_contiguous_global_batches",
        )

    def test_canonical_training_has_no_experimental_student_dependency(self):
        training = json.loads(
            (ROOT / "config/unarchitectured_v1_training.json").read_text()
        )
        self.assertEqual(
            training["student_distillation"]["student_config"],
            "config/unarchitectured_v1_student.json",
        )
        student = json.loads(
            (ROOT / "config/unarchitectured_v1_student.json").read_text()
        )
        self.assertNotIn("steps_per_epoch", student["chessformer"])

    def test_checkpoint_names_use_canonical_generation(self):
        source = (TOOLS / "train_hydra_oracle_v5_a100.py").read_text()
        self.assertIn("UNARCHV1_ORACLE_TRAINING_V1_DDP", source)
        self.assertIn("UNARCHV1_STUDENT_DISTILLATION_V1_DDP", source)
        self.assertNotIn("UNCHAPX1_", source)
        self.assertNotIn("UNCHAPX5_", source)

    def test_canonical_wrappers_exist(self):
        self.assertTrue(
            (ROOT / "scripts/training/verda_unarchitectured_v1_train.sh").is_file()
        )
        self.assertTrue(
            (ROOT / "tools/unarchitectured_v1_runtime_readiness.py").is_file()
        )


if __name__ == "__main__":
    unittest.main()
