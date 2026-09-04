#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "unarchitectured_v1_watchdog.py"
SPEC = importlib.util.spec_from_file_location(
    "unarchitectured_v1_watchdog", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ROOT = TOOLS.parent
BASE_POLICY = json.loads((ROOT / "config/unarchitectured_v1_safety.json").read_text())


def policy():
    value = json.loads(json.dumps(BASE_POLICY))
    value["training"].update(
        {
            "startup_timeout_seconds": 0.2,
            "heartbeat_timeout_seconds": 0.1,
            "terminate_grace_seconds": 0.05,
            "poll_interval_seconds": 0.02,
        }
    )
    return value


class UnarchitecturedV1WatchdogTests(unittest.TestCase):
    def test_success_requires_valid_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            heartbeat = Path(directory) / "heartbeat.json"
            incident = Path(directory) / "incident.json"
            code = (
                "import json,time,pathlib;"
                f"p=pathlib.Path({str(heartbeat)!r});"
                "p.write_text(json.dumps({'architecture':'Unarchitectured v1',"
                "'unix_time':time.time(),'monotonic_time':time.monotonic()}));"
                "time.sleep(0.08)"
            )
            result = MODULE.supervise(
                [sys.executable, "-c", code], policy(), heartbeat, incident
            )
            self.assertEqual(result, 0)
            self.assertFalse(incident.exists())

    def test_zero_exit_without_heartbeat_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            heartbeat = Path(directory) / "heartbeat.json"
            incident = Path(directory) / "incident.json"
            result = MODULE.supervise(
                [sys.executable, "-c", "pass"], policy(), heartbeat, incident
            )
            self.assertEqual(result, 125)
            payload = json.loads(incident.read_text())
            self.assertEqual(payload["action"], "missing_heartbeat")

    def test_stale_heartbeat_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            heartbeat = Path(directory) / "heartbeat.json"
            incident = Path(directory) / "incident.json"
            code = (
                "import json,time,pathlib;"
                f"p=pathlib.Path({str(heartbeat)!r});"
                "p.write_text(json.dumps({'architecture':'Unarchitectured v1',"
                "'unix_time':time.time()-10,'monotonic_time':time.monotonic()-10}));"
                "time.sleep(5)"
            )
            result = MODULE.supervise(
                [sys.executable, "-c", code], policy(), heartbeat, incident
            )
            self.assertEqual(result, 124)
            payload = json.loads(incident.read_text())
            self.assertEqual(payload["action"], "terminated")
            self.assertIn("stale", payload["reason"])


if __name__ == "__main__":
    unittest.main()
