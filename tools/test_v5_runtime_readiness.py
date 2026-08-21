#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("v5_runtime_readiness.py")
SPEC = importlib.util.spec_from_file_location("v5_runtime_readiness", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V5RuntimeReadinessTests(unittest.TestCase):
    def test_repository_reports_partial_package_but_missing_forward_runtime(self):
        report = MODULE.readiness()
        self.assertFalse(report["ready_for_engine_candidate_training"])
        self.assertTrue(all(value["exists"] for value in report["checks"].values()))
        self.assertTrue(report["capabilities"]["container_format"])
        self.assertFalse(report["capabilities"]["scalar_neural_forward"])
        self.assertFalse(report["capabilities"]["runtime_safety_suite"])

    def test_complete_fixture_becomes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in MODULE.REQUIREMENTS.values():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("fixture", encoding="utf-8")
            capability_path = root / MODULE.CAPABILITIES
            capability_path.parent.mkdir(parents=True, exist_ok=True)
            capability_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        **{name: True for name in MODULE.REQUIRED_CAPABILITIES},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(MODULE.readiness(root)["ready_for_engine_candidate_training"])


if __name__ == "__main__":
    unittest.main()
