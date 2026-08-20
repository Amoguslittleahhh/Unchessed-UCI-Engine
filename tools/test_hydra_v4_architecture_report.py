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
MODULE_PATH = TOOLS / "hydra_v4_architecture_report.py"
SPEC = importlib.util.spec_from_file_location("hydra_v4_architecture_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ROOT = TOOLS.parent
CONFIG = json.loads((ROOT / "config/unchessed_hydra_v4.json").read_text())
TRAINING = json.loads((ROOT / "config/a100_hydra_v4_training.json").read_text())


class HydraV4ArchitectureReportTests(unittest.TestCase):
    def test_config_and_joint_objectives_are_normalized(self):
        MODULE.validate_config(CONFIG)
        self.assertAlmostEqual(sum(CONFIG["joint_training"]["loss_weights"].values()), 1.0)
        self.assertAlmostEqual(sum(TRAINING["chessformer"]["loss_weights"].values()), 1.0)

    def test_legal_regret_head_has_bounded_cost(self):
        budget = MODULE.chessformer_budget(CONFIG["chessformer"])
        self.assertEqual(budget["legal_regret_parameters"], 16_610)
        self.assertEqual(budget["parameters"], 4_222_905)
        self.assertLess(budget["legal_regret_parameters"], budget["parameters"] // 100)
        flops = [entry["flops_approx"] for entry in budget["exit_budgets"].values()]
        self.assertEqual(flops, sorted(flops))

    def test_v4_record_carries_every_legal_action_and_regret(self):
        data = MODULE.data_budget(CONFIG["data"])
        self.assertEqual(data["record_bytes"], 1088)
        self.assertEqual(data["legal_action_bytes"], 218 * 2)
        self.assertEqual(data["legal_regret_bytes"], 218 * 2)
        self.assertEqual(data["records_per_gib"], 986_895)

    def test_search_candidate_set_cannot_prune(self):
        report = MODULE.build_report(CONFIG)
        search = report["search_interface"]
        self.assertFalse(search["noncandidate_pruning_allowed"])
        self.assertTrue(search["full_legal_fallback_required"])
        broken = copy.deepcopy(CONFIG)
        broken["search_interface"]["noncandidate_pruning_allowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot remove"):
            MODULE.validate_config(broken)

    def test_a100_trainer_matches_architecture(self):
        model = CONFIG["chessformer"]
        trainer = TRAINING["chessformer"]
        self.assertEqual(trainer["exit_layers"], model["exit_layers"])
        self.assertEqual(trainer["matryoshka_widths"], model["matryoshka_widths"])
        self.assertEqual(trainer["regret_width"], model["legal_regret_width"])

    def test_rejects_collapsed_promotion_vocabulary(self):
        broken = copy.deepcopy(CONFIG)
        broken["chessformer"]["policy_action_vocabulary"] = 4096
        with self.assertRaisesRegex(ValueError, "promotion"):
            MODULE.validate_config(broken)


if __name__ == "__main__":
    unittest.main()
