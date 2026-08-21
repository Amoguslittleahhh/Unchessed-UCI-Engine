#!/usr/bin/env python3

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
MODULE_PATH = TOOLS / "unarchitectured_v1_safety.py"
SPEC = importlib.util.spec_from_file_location("unarchitectured_v1_safety", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ROOT = TOOLS.parent
POLICY = json.loads((ROOT / "config/unarchitectured_v1_safety.json").read_text())


class UnarchitecturedV1SafetyTests(unittest.TestCase):
    def controller(self):
        return MODULE.TrainingSafetyController(POLICY)

    def test_finite_normal_steps_continue(self):
        controller = self.controller()
        for value in (2.0, 1.8, 1.6, 1.5):
            self.assertTrue(controller.check_step(value, 2.0).safe)
        self.assertIsNotNone(controller.loss_ema)

    def test_nonfinite_gradient_and_loss_spike_abort(self):
        controller = self.controller()
        self.assertFalse(controller.check_step(math.nan, 1.0).safe)
        controller = self.controller()
        controller.check_step(1.0, 1.0)
        self.assertEqual(controller.check_step(100.0, 1.0).action, "abort")
        self.assertEqual(controller.check_step(1.0, 2000.0).action, "abort")

    def test_validation_cusum_or_patience_stops_without_supervision(self):
        controller = self.controller()
        self.assertTrue(controller.check_validation(1.0).safe)
        decision = controller.check_validation(1.01)
        if decision.safe:
            decision = controller.check_validation(1.02)
        self.assertEqual(decision.action, "early_stop")

    def test_atomic_heartbeat_is_self_describing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heartbeat.json"
            MODULE.write_heartbeat(path, "oracle", {"global_step": 25})
            payload = json.loads(path.read_text())
            self.assertEqual(payload["architecture"], "Unarchitectured v1")
            self.assertEqual(payload["global_step"], 25)
            self.assertIn("unix_time", payload)

    def test_checked_in_policy_is_fail_closed(self):
        self.assertTrue(POLICY["fail_closed"])
        self.assertEqual(
            POLICY["responses"]["heartbeat_stale"],
            "terminate_process_group_and_write_incident",
        )
        self.assertEqual(POLICY["training"]["maximum_consecutive_nonfinite_steps"], 0)

    def test_launcher_uses_external_watchdog_and_internal_heartbeats(self):
        launcher = (
            ROOT / "scripts/training/a100_hydra_v5_train.sh"
        ).read_text()
        trainer = (ROOT / "tools/train_hydra_oracle_v5_a100.py").read_text()
        watchdog = (ROOT / "tools/unarchitectured_v1_watchdog.py").read_text()
        self.assertIn("unarchitectured_v1_watchdog.py", launcher)
        self.assertIn('rm -f "$OUTPUT_DIR/oracle-unarchitectured-v1.heartbeat.json"', launcher)
        self.assertIn("write_heartbeat", trainer)
        self.assertIn("os.killpg", watchdog)
        self.assertIn("SIGKILL", watchdog)
        self.assertIn('"action": "child_failed"', watchdog)


if __name__ == "__main__":
    unittest.main()
