#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
MODULE_PATH = TOOLS / "verda_cpu_profile.py"
SPEC = importlib.util.spec_from_file_location("verda_cpu_profile", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ROOT = TOOLS.parent
PROFILES = json.loads((ROOT / "config/verda_cpu_profiles.json").read_text())
BASE = json.loads((ROOT / "config/v5_180core_datagen.json").read_text())


class VerdaCpuProfileTests(unittest.TestCase):
    def test_every_advertised_size_resolves(self):
        expected = {
            4: 4,
            8: 7,
            16: 15,
            32: 30,
            64: 60,
            96: 92,
            120: 116,
            180: 176,
            360: 352,
        }
        for vcpus, workers in expected.items():
            with self.subTest(vcpus=vcpus):
                resolved = MODULE.resolve(BASE, PROFILES, list(range(vcpus)))
                self.assertEqual(resolved["cpu"]["verda_vcpus"], vcpus)
                self.assertEqual(resolved["cpu"]["resolved_workers"], workers)
                self.assertEqual(resolved["cpu"]["requested_cores"], workers)
                self.assertFalse(resolved["cpu"]["physical_cores_only"])
                self.assertEqual(resolved["teacher"]["threads"], 1)

    def test_unknown_vcpu_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected one of"):
            MODULE.select_profile(PROFILES, 48)

    def test_checked_in_base_represents_180_vcpu_node(self):
        self.assertEqual(BASE["cpu"]["verda_vcpus"], 180)
        self.assertEqual(BASE["cpu"]["requested_cores"], 176)
        self.assertEqual(BASE["cpu"]["reserve_cores"], 4)
        self.assertFalse(BASE["cpu"]["physical_cores_only"])


if __name__ == "__main__":
    unittest.main()
