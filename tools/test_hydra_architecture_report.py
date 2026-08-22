#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("hydra_architecture_report.py")
SPEC = importlib.util.spec_from_file_location("hydra_architecture_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ROOT = MODULE_PATH.parent.parent
CONFIG = json.loads((ROOT / "config/unchessed_hydra_v1.json").read_text())


class HydraArchitectureReportTests(unittest.TestCase):
    def test_xt_dimensions_and_memory_are_consistent(self):
        budget = MODULE.xt_budget(CONFIG["xt_nnue"])
        self.assertEqual(budget["threat_dimensions"], 32_400)
        self.assertEqual(budget["materialized_threat_parameters"], 1_036_800)
        self.assertEqual(budget["runtime_position_bytes"], 22_528 * 256 * 2)
        self.assertEqual(budget["incremental_state_bytes_per_ply"], 1_152)

    def test_chessformer_budget_matches_frozen_config(self):
        budget = MODULE.chessformer_budget(CONFIG["chessformer"])
        self.assertEqual(budget["parameters"], 4_188_744)
        self.assertEqual(budget["root_forward_flops_approx"], 547_816_960)
        self.assertEqual(budget["attention_matrix_elements_per_layer"], 32_768)

    def test_joint_loss_is_normalized(self):
        weights = CONFIG["joint_training"]["loss_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
