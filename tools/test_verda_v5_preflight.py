#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "verda_v5_preflight.py"
SPEC = importlib.util.spec_from_file_location("verda_v5_preflight", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VerdaV5PreflightTests(unittest.TestCase):
    def test_nvidia_csv_parser(self):
        parsed = MODULE.parse_nvidia_csv(
            "0, NVIDIA A100-SXM4-80GB, 81920, 570.12\n"
        )
        self.assertEqual(parsed[0]["index"], 0)
        self.assertEqual(parsed[0]["memory_total_mib"], 81920)
        self.assertIn("A100", parsed[0]["name"])
        self.assertIsNotNone(MODULE.SUPPORTED_GPU.search(parsed[0]["name"]))
        self.assertIsNotNone(MODULE.SUPPORTED_GPU.search("NVIDIA B300"))
        self.assertIsNotNone(MODULE.SUPPORTED_GPU.search("Tesla V100-SXM2-16GB"))

    def test_mount_probe_accepts_future_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            future = Path(directory) / "not-created" / "data"
            report = MODULE.mount_info(future)
            self.assertEqual(report["path"], str(future.resolve()))
            self.assertGreater(report["total_bytes"], 0)
            self.assertGreaterEqual(report["free_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
