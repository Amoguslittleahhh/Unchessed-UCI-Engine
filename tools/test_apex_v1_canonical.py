#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
ROOT = TOOLS.parent
MODULE_PATH = TOOLS / "apex_v1_architecture_report.py"
SPEC = importlib.util.spec_from_file_location("apex_v1_architecture_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ApexV1CanonicalTests(unittest.TestCase):
    def test_registry_has_exactly_one_canonical_architecture(self):
        registry = json.loads((ROOT / "config/architecture_registry.json").read_text())
        self.assertEqual(registry["canonical"]["id"], "apex-v1")
        self.assertEqual(registry["canonical"]["runtime_magic"], "UNCHAPX1")
        self.assertEqual(len(registry["experimental_predecessors"]), 5)
        self.assertTrue(registry["policy"]["predecessors_are_not_production_versions"])

    def test_canonical_config_and_budget(self):
        config = json.loads((ROOT / "config/unchessed_apex_v1.json").read_text())
        student = json.loads((ROOT / "config/unchessed_hydra_v4.json").read_text())
        profiles = json.loads((ROOT / "config/verda_gpu_profiles.json").read_text())
        training = json.loads((ROOT / "config/apex_v1_training.json").read_text())
        report = MODULE.build_report(config, student, profiles, training)
        self.assertEqual(report["schema"], 1)
        self.assertEqual(report["name"], "Unchessed Apex v1")
        self.assertEqual(report["runtime_file_magic"], "UNCHAPX1")
        self.assertEqual(report["runtime_student"]["parameters"], 4_222_905)

    def test_all_hydra_versions_are_explicitly_experimental(self):
        for version in range(1, 6):
            config = json.loads(
                (ROOT / f"config/unchessed_hydra_v{version}.json").read_text()
            )
            self.assertIn("experimental predecessor", config["status"])
            self.assertEqual(
                config["canonical_successor"], "config/unchessed_apex_v1.json"
            )

    def test_training_checkpoint_names_use_canonical_generation(self):
        source = (TOOLS / "train_hydra_oracle_v5_a100.py").read_text()
        self.assertIn("UNCHAPX1_ORACLE_TRAINING_V1_DDP", source)
        self.assertIn("UNCHAPX1_STUDENT_DISTILLATION_V1_DDP", source)
        self.assertNotIn("UNCHAPX5_", source)


if __name__ == "__main__":
    unittest.main()
