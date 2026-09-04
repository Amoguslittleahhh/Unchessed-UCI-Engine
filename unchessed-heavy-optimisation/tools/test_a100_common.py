#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("a100_common.py")
SPEC = importlib.util.spec_from_file_location("a100_common_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class A100CommonTests(unittest.TestCase):
    def test_warmup_and_cosine_schedule_are_bounded(self):
        values = [MODULE.learning_rate(step, 100, 1e-3, 10) for step in range(100)]
        self.assertAlmostEqual(values[0], 1e-4)
        self.assertAlmostEqual(values[9], 1e-3)
        self.assertTrue(all(0 < value <= 1e-3 for value in values))
        self.assertGreater(values[20], values[80])

    def test_config_requires_schema_and_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"schema": 1, "hardware": {"precision": "bf16"}, "model": {"x": 1}}),
                encoding="utf-8",
            )
            model, hardware = MODULE.load_config(path, "model")
            self.assertEqual(model, {"x": 1})
            self.assertEqual(hardware["precision"], "bf16")
            with self.assertRaises(ValueError):
                MODULE.load_config(path, "missing")

    def test_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"unchessed-a100")
            self.assertEqual(
                MODULE.sha256_file(path),
                "d40fe557fc8ec0e6a84c24e5efe8282e462db9cbbe23bb1dbdefa049e4287089",
            )


if __name__ == "__main__":
    unittest.main()
