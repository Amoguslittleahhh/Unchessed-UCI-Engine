#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("unarchitectured_v1_architecture_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "unarchitectured_v1_architecture_audit", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UnarchitecturedV1ArchitectureAuditTests(unittest.TestCase):
    def test_all_cross_config_contracts_pass(self):
        report = MODULE.audit()
        failed = [name for name, passed in report["checks"].items() if not passed]
        self.assertEqual(failed, [])
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["oracle_parameters_by_profile"],
            [878_114_575, 501_835_855, 230_537_295, 58_412_431, 29_144_367],
        )


if __name__ == "__main__":
    unittest.main()
