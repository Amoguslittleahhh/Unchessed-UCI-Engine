"""Tests for cloud preflight — no torch."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnue_cloud_runtime import CLOUD_GO_TOKEN, cloud_flags, preflight_errors


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        for k in (
            "GO_CLOUD", "REQUIRE_CLOUD_GO", "PERSONA_ACTIVE", "UNARCH_HINT",
            "EPOCH_CAP", "USE_AMP",
        ):
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_local_train_without_go_token_is_ok(self):
        err = preflight_errors(10_000, "cpu", go_required=False)
        self.assertEqual(err, [])

    def test_cloud_without_token_is_blocked(self):
        err = preflight_errors(10_000, "cuda", go_required=True)
        self.assertTrue(any("GO_CLOUD" in e for e in err))

    def test_cloud_with_token_passes(self):
        os.environ["GO_CLOUD"] = CLOUD_GO_TOKEN
        err = preflight_errors(10_000, "cuda", go_required=True)
        self.assertEqual(err, [])

    def test_persona_off_is_blocked(self):
        os.environ["PERSONA_ACTIVE"] = "0"
        err = preflight_errors(10_000, "cpu", go_required=False)
        self.assertTrue(any("PERSONA" in e for e in err))

    def test_unarch_hint_on_is_blocked(self):
        os.environ["UNARCH_HINT"] = "1"
        err = preflight_errors(10_000, "cpu", go_required=False)
        self.assertTrue(any("UNARCH" in e for e in err))

    def test_too_few_records(self):
        err = preflight_errors(10, "cpu", go_required=False)
        self.assertTrue(any("refusing to train" in e or "1000" in e for e in err))

    def test_too_many_records(self):
        err = preflight_errors(500_000_001, "cuda", go_required=False)
        self.assertTrue(any("SAFE_MAX" in e for e in err))

    def test_defaults_keep_persona_on_hint_off(self):
        f = cloud_flags()
        self.assertTrue(f["persona_active"])
        self.assertFalse(f["unarch_hint"])
        self.assertTrue(f["allow_tf32"])
        self.assertTrue(f["use_amp"])
        self.assertFalse(f["torch_compile"])


if __name__ == "__main__":
    unittest.main()
