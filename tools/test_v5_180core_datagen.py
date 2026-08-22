#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "v5_180core_datagen.py"
SPEC = importlib.util.spec_from_file_location("v5_180core_datagen", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
import aegis_v3_data as V3
import aegis_v4_data as V4


def record(game, player):
    target = 12 | (28 << 6)
    actions = sorted((target, 1 | (18 << 6)))
    base = V3.AegisV3Record(
        bitboards=(0x10, 0x42, 0x24, 0x81, 0x08, 0x10) * 2,
        move=target,
        promotion=0,
        wdl=1,
        rating=1500,
        castling=0,
        ep_file=0xFF,
        halfmove=0,
        time_class=2,
        flags=0,
        history_len=0,
        history=(0,) * 8,
        game_hash=game,
        player_hash=player,
    )
    return V4.AegisV4Record(
        base=base,
        legal_count=2,
        target_action=target,
        teacher_best_action=V4.ACTION_SENTINEL,
        policy_kind=V4.POLICY_HUMAN,
        legal_flags=0,
        legal_actions=tuple(actions) + (V4.ACTION_SENTINEL,) * 216,
        legal_regrets=(V4.REGRET_SENTINEL,) * 218,
    )


class V5180CoreDatagenTests(unittest.TestCase):
    def test_cpu_list_and_nonoverlapping_allocation(self):
        self.assertEqual(MODULE.parse_cpu_list("0-3,8,10-11"), [0, 1, 2, 3, 8, 10, 11])
        sets = MODULE.allocate_core_sets(list(range(12)), 2, 6, False)
        self.assertEqual(len(sets), 6)
        self.assertTrue(all(len(values) == 2 for values in sets))
        self.assertEqual(len({core for values in sets for core in values}), 12)

    def test_rejects_oversubscribed_layout(self):
        with self.assertRaisesRegex(ValueError, "only"):
            MODULE.allocate_core_sets(list(range(7)), 2, 4, False)

    def test_plan_is_deterministic_and_range_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "human.aegis4"
            V4.write_shard(input_path, [record(i + 1, i + 101) for i in range(5)])
            config = {
                "schema": 1,
                "cpu": {
                    "requested_cores": 4,
                    "cores_per_worker": 1,
                    "reserve_cores": 0,
                    "physical_cores_only": False,
                    "numa_aware": False,
                    "maximum_memory_fraction": 0.8,
                },
                "teacher": {
                    "engine": "/fake/engine",
                    "threads": 1,
                    "hash_mb": 1,
                    "estimated_process_mb": 1,
                    "nodes_per_action": 10,
                },
                "data": {
                    "input_globs": [str(input_path)],
                    "output_dir": str(root / "out"),
                    "records_per_task": 2,
                },
            }
            with mock.patch.object(MODULE, "available_cores", return_value=list(range(4))), mock.patch.object(
                MODULE, "memory_total_mb", return_value=1024
            ):
                first = MODULE.build_plan(config)
                second = MODULE.build_plan(config)
            self.assertEqual(first["records"], 5)
            self.assertEqual([task["count"] for task in first["tasks"]], [2, 2, 1])
            self.assertEqual(
                [task["task_id"] for task in first["tasks"]],
                [task["task_id"] for task in second["tasks"]],
            )
            command = MODULE.task_command(first, first["tasks"][0])
            self.assertIn("--input-sha256", command)
            self.assertIn("--resume", command)

    def test_checked_in_configuration_targets_180_cores(self):
        config = json.loads((TOOLS.parent / "config/v5_180core_datagen.json").read_text())
        self.assertEqual(config["cpu"]["verda_vcpus"], 180)
        self.assertEqual(config["cpu"]["requested_cores"], 176)
        self.assertEqual(config["cpu"]["reserve_cores"], 4)
        self.assertEqual(config["cpu"]["cores_per_worker"], 1)
        self.assertFalse(config["cpu"]["physical_cores_only"])
        self.assertEqual(config["teacher"]["threads"], 1)
        self.assertTrue(config["teacher"]["clear_hash_per_action"])


if __name__ == "__main__":
    unittest.main()
