#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("hydra_v2_architecture_report.py")
SPEC = importlib.util.spec_from_file_location("hydra_v2_architecture_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ROOT = MODULE_PATH.parent.parent
CONFIG = json.loads((ROOT / "config/unchessed_hydra_v2.json").read_text())


class HydraV2ArchitectureReportTests(unittest.TestCase):
    def test_xt_multiresolution_budget(self):
        budget = MODULE.xt_budget(CONFIG["xt_nnue"])
        self.assertEqual(budget["runtime_direct_threat_bytes"], 1_036_800)
        self.assertEqual(budget["runtime_xray_bytes"], 221_184)
        self.assertEqual(budget["runtime_pawn_topology_bytes"], 65_536)
        self.assertEqual(budget["incremental_state_bytes_per_ply"], 1_280)
        self.assertEqual(budget["full_padded_input"] % 32, 0)

    def test_elastic_exit_budgets_are_monotonic(self):
        budget = MODULE.chessformer_budget(CONFIG["chessformer"])
        exits = list(budget["exit_budgets"].values())
        flops = [entry["flops_approx"] for entry in exits]
        self.assertEqual(flops, sorted(flops))
        self.assertEqual(budget["parameters"], 4_606_296)
        self.assertGreater(budget["policy_dot_reduction_factor"], 18)

    def test_joint_objective_is_normalized(self):
        weights = CONFIG["joint_training"]["loss_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
