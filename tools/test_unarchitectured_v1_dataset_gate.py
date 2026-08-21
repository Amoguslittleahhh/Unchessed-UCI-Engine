#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).parent
MODULE_PATH = TOOLS / "unarchitectured_v1_dataset_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "unarchitectured_v1_dataset_gate", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summary(records, human, guide, positions, games, players, duplicate_fraction=0.0):
    return {
        "records": records,
        "human": human,
        "guide": guide,
        "regret_labelled": guide,
        "positions": set(positions),
        "games": set(games),
        "players": set(players),
        "duplicate_positions": int(records * duplicate_fraction),
        "duplicate_fraction": duplicate_fraction,
    }


class UnarchitecturedV1DatasetGateTests(unittest.TestCase):
    def setUp(self):
        self.safety = {
            "data": {
                "minimum_human_fraction": 0.2,
                "minimum_guide_fraction": 0.2,
                "maximum_duplicate_position_fraction": 0.02,
            }
        }
        self.training = {
            "oracle": {
                "effective_batch_records": 10,
                "minimum_optimizer_steps_per_epoch": 10,
                "minimum_validation_records": 20,
            }
        }
        self.provenance = {
            "splits": {
                "train": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
                "tune": {"start_date": "2025-01-01", "end_date": "2025-06-30"},
                "final": {"start_date": "2025-07-01", "end_date": "2025-12-31"},
            }
        }

    def test_disjoint_balanced_dated_splits_pass(self):
        values = [
            summary(100, 50, 50, range(100), range(100), range(100)),
            summary(20, 10, 10, range(100, 120), range(100, 120), range(100, 120)),
            summary(20, 10, 10, range(120, 140), range(120, 140), range(120, 140)),
        ]
        with mock.patch.object(MODULE, "summarize", side_effect=values):
            report = MODULE.audit([], [], [], self.safety, self.training, self.provenance)
        self.assertTrue(report["passed"])

    def test_overlap_and_bad_date_fail_closed(self):
        values = [
            summary(100, 90, 10, [1, 2], [1], [1]),
            summary(20, 10, 10, [2, 3], [2], [2]),
            summary(20, 10, 10, [4], [3], [1]),
        ]
        bad_dates = {"splits": dict(self.provenance["splits"])}
        bad_dates["splits"]["final"] = {
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
        }
        with mock.patch.object(MODULE, "summarize", side_effect=values):
            report = MODULE.audit([], [], [], self.safety, self.training, bad_dates)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["minimum_guide_fraction"])
        self.assertFalse(report["checks"]["position_disjoint"])
        self.assertFalse(report["checks"]["player_disjoint"])
        self.assertFalse(report["checks"]["future_ordered"])


if __name__ == "__main__":
    unittest.main()
