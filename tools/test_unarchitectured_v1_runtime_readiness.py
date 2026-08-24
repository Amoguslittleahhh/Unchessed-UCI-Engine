#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("unarchitectured_v1_runtime_readiness.py")
SPEC = importlib.util.spec_from_file_location("unarchitectured_v1_runtime_readiness", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UnarchitecturedV1RuntimeReadinessTests(unittest.TestCase):
    def test_repository_reports_forward_runtime_but_missing_safety_gate(self):
        report = MODULE.readiness()
        self.assertFalse(report["ready_for_engine_candidate_training"])
        self.assertTrue(all(value["exists"] for value in report["checks"].values()))
        self.assertTrue(report["capabilities"]["container_format"])
        self.assertTrue(report["capabilities"]["scalar_neural_forward"])
        self.assertTrue(report["capabilities"]["exported_reference_vectors"])
        self.assertTrue(report["capabilities"]["quantized_neural_forward"])
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

    def test_npu_dispatch_is_flagged_experimental_and_off(self):
        """NPU dispatch is unimplemented; the flag must say so."""
        report = MODULE.readiness()
        self.assertIn("npu_dispatch", MODULE.EXPERIMENTAL_CAPABILITIES)
        self.assertFalse(report["experimental_capabilities"]["npu_dispatch"])
        self.assertEqual(report["experimental_violations"], [])

        capabilities = json.loads(
            (Path(MODULE.ROOT) / MODULE.CAPABILITIES).read_text(encoding="utf-8")
        )
        self.assertFalse(capabilities["npu_dispatch"])
        self.assertEqual(
            capabilities["npu_dispatch_status"], "experimental-unimplemented"
        )
        # The rationale must travel with the flag, so nobody re-litigates it
        # from scratch or flips it on assuming it just needs wiring up.
        self.assertIn("NOT IMPLEMENTED", capabilities["npu_dispatch_notes"])

    def test_enabling_npu_dispatch_without_implementation_fails_closed(self):
        """Flipping the flag on must break readiness, not silently pass.

        This is the actual protection: even with every required capability
        forced true, an unimplemented experimental capability keeps the tree
        from reporting itself ready.
        """
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
                        "npu_dispatch": True,
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.readiness(root)
            self.assertFalse(
                report["ready_for_engine_candidate_training"],
                "an unimplemented experimental capability must fail closed",
            )
            self.assertEqual(report["experimental_violations"], ["npu_dispatch"])


if __name__ == "__main__":
    unittest.main()
