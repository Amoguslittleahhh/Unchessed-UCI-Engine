#!/usr/bin/env python3

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("hydra_v3_architecture_report.py")
SPEC = importlib.util.spec_from_file_location("hydra_v3_architecture_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ROOT = MODULE_PATH.parent.parent
CONFIG = json.loads((ROOT / "config/unchessed_hydra_v3.json").read_text())
TRAINING = json.loads((ROOT / "config/a100_hydra_v3_training.json").read_text())


class HydraV3ArchitectureReportTests(unittest.TestCase):
    def test_three_stage_xt_budget(self):
        budget = MODULE.xt_budget(CONFIG["xt_nnue"])
        self.assertEqual(budget["runtime_direct_threat_bytes"], 1_036_800)
        self.assertEqual(budget["runtime_xray_bytes"], 221_184)
        self.assertEqual(budget["runtime_pawn_topology_bytes"], 65_536)
        self.assertEqual(budget["incremental_state_bytes_per_ply"], 1_280)
        self.assertEqual(budget["calibration_bytes"], 512)
        self.assertLess(budget["fast_head_input"], budget["direct_padded_input"])
        self.assertLess(budget["direct_padded_input"], budget["full_padded_input"])
        self.assertEqual(budget["direct_padded_input"] % 32, 0)
        self.assertEqual(budget["full_padded_input"] % 32, 0)

    def test_private_history_adapter_is_small_and_explicit(self):
        budget = MODULE.chessformer_budget(CONFIG["chessformer"])
        self.assertEqual(budget["private_adapter_parameters"], 49_152)
        self.assertEqual(budget["history_adapter_parameters"], 13_120)
        self.assertLess(budget["private_adapter_parameters"], budget["parameters"] // 50)
        exits = list(budget["exit_budgets"].values())
        self.assertEqual(
            [entry["flops_approx"] for entry in exits],
            sorted(entry["flops_approx"] for entry in exits),
        )
        self.assertGreater(budget["policy_dot_reduction_factor"], 18)

    def test_v3_data_abi_is_frozen(self):
        budget = MODULE.data_budget(CONFIG["data"])
        self.assertEqual(budget["header_bytes"], 64)
        self.assertEqual(budget["record_bytes"], 160)
        self.assertEqual(budget["records_per_gib"], 6_710_886)

    def test_joint_objective_is_normalized(self):
        MODULE.validate_config(CONFIG)
        self.assertAlmostEqual(
            sum(CONFIG["joint_training"]["loss_weights"].values()), 1.0, places=12
        )

    def test_a100_trainer_matches_frozen_feature_schema(self):
        architecture = CONFIG["xt_nnue"]
        training = TRAINING["xt_nnue"]
        self.assertEqual(
            training["direct_threat_dimensions"],
            architecture["direct_threat_dimensions"],
        )
        self.assertEqual(training["xray_dimensions"], architecture["xray_hyperedge_dimensions"])
        self.assertEqual(
            training["pawn_topology_dimensions"], architecture["pawn_topology_dimensions"]
        )
        self.assertEqual(training["fast_ensemble_members"], 2)
        self.assertTrue(training["bootstrap_fast_ensemble"])
        self.assertAlmostEqual(sum(training["loss_weights"].values()), 1.0, places=12)

    def test_rejects_unpaired_elastic_exits(self):
        broken = copy.deepcopy(CONFIG)
        broken["chessformer"]["matryoshka_widths"] = [128, 256]
        with self.assertRaisesRegex(ValueError, "paired"):
            MODULE.validate_config(broken)

    def test_rejects_incomplete_semantic_record(self):
        broken = copy.deepcopy(CONFIG["data"])
        broken["contains_player_hash"] = False
        with self.assertRaisesRegex(ValueError, "mandatory"):
            MODULE.data_budget(broken)


if __name__ == "__main__":
    unittest.main()
