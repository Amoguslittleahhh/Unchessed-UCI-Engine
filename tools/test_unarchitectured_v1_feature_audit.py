#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("unarchitectured_v1_feature_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "unarchitectured_v1_feature_audit", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UnarchitecturedV1FeatureAuditTests(unittest.TestCase):
    def test_all_feature_contracts_match(self):
        report = MODULE.audit()
        failures = [name for name, passed in report["checks"].items() if not passed]
        self.assertEqual(failures, [])
        self.assertTrue(report["passed"])
        self.assertEqual(report["expected"]["direct_threat_dimensions"], 32_400)
        self.assertEqual(report["expected"]["xray_hyperedge_dimensions"], 13_824)


if __name__ == "__main__":
    unittest.main()
