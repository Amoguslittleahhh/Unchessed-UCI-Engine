#!/usr/bin/env python3
"""Tests for the int8 activation calibration study.

The study's headline output is a *negative* result, which makes it unusually
important to test the machinery: a quantizer that silently did nothing would
also report "no drift" and could be mistaken for a positive finding. These
tests check that each scheme genuinely quantizes, that the error metric
responds in the right direction, and that the committed artifact records the
negative conclusion it actually measured.

Runs without torch where possible; the tensor-level tests skip if torch is
absent, the artifact tests always run.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

ARTIFACT = ROOT / "benchmarks" / "unarchitectured-metal" / "int8-activation-calibration.json"

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class QuantizerBehaviour(unittest.TestCase):
    def setUp(self):
        from analyse_int8_activation_calibration import quantize_dequantize

        self.qdq = quantize_dequantize
        torch.manual_seed(0)

    def test_every_scheme_actually_quantizes(self):
        """A no-op quantizer would make the whole study report false success."""
        from analyse_int8_activation_calibration import SCHEMES

        x = torch.randn(1, 64, 256)
        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                out = self.qdq(x, scheme)
                self.assertEqual(out.shape, x.shape)
                # It must change the values (it is lossy) ...
                self.assertGreater((out - x).abs().max().item(), 0.0)
                # ... but stay close, or it isn't a quantizer, it's noise.
                rel = (out - x).abs().max().item() / x.abs().max().item()
                self.assertLess(rel, 0.25)

    def test_output_lands_on_a_representable_grid(self):
        """Round-tripped values must be integer multiples of some scale."""
        x = torch.randn(1, 8, 64)
        out = self.qdq(x, "per_token_symmetric")
        scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
        levels = out / scale
        self.assertLess((levels - levels.round()).abs().max().item(), 1e-4)
        self.assertLessEqual(levels.abs().max().item(), 127.0 + 1e-4)

    def test_group_size_one_is_lossless_and_larger_groups_are_worse(self):
        """Sanity-check the direction of the error, not just its presence."""
        x = torch.randn(1, 4, 64)
        err = {}
        for gs in (1, 8, 64):
            out = self.qdq(x, "per_group_symmetric", group_size=gs)
            err[gs] = (out - x).abs().max().item()
        self.assertAlmostEqual(err[1], 0.0, places=6)
        self.assertLessEqual(err[8], err[64] + 1e-9)

    def test_per_group_handles_non_multiple_widths(self):
        """Padding bug here would silently truncate a real activation."""
        x = torch.randn(1, 3, 70)
        out = self.qdq(x, "per_group_symmetric", group_size=32)
        self.assertEqual(out.shape, x.shape)

    def test_outlier_makes_symmetric_worse_and_percentile_better(self):
        """The motivating hypothesis for the percentile scheme, tested."""
        x = torch.full((1, 1, 256), 0.01)
        x[0, 0, 0] = 10.0  # one loud channel
        sym = self.qdq(x, "per_token_symmetric")
        pct = self.qdq(x, "percentile", percentile=99.0)
        # Symmetric scaling crushes the quiet channels to zero.
        quiet_sym = (sym[0, 0, 1:] - x[0, 0, 1:]).abs().max().item()
        quiet_pct = (pct[0, 0, 1:] - x[0, 0, 1:]).abs().max().item()
        self.assertGreater(quiet_sym, quiet_pct)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class ComparisonMetric(unittest.TestCase):
    def test_identical_outputs_report_zero_drift(self):
        from analyse_int8_activation_calibration import compare

        out = {
            "logits": torch.tensor([[1.0, 2.0, 0.5]]),
            "evidence": torch.tensor([[1.0, 2.0]]),
            "representation": torch.tensor([[0.1, 0.2]]),
        }
        r = compare(out, out, 3)
        self.assertEqual(r["max_logit_drift"], 0.0)
        self.assertTrue(r["best_move_preserved"])

    def test_detects_a_changed_best_move(self):
        from analyse_int8_activation_calibration import compare

        base = {
            "logits": torch.tensor([[1.0, 2.0, 0.5]]),
            "evidence": torch.tensor([[1.0]]),
            "representation": torch.tensor([[0.1]]),
        }
        cand = {
            "logits": torch.tensor([[3.0, 2.0, 0.5]]),
            "evidence": torch.tensor([[1.0]]),
            "representation": torch.tensor([[0.1]]),
        }
        r = compare(base, cand, 3)
        self.assertFalse(r["best_move_preserved"])
        self.assertEqual(r["baseline_best_index"], 1)
        self.assertEqual(r["candidate_best_index"], 0)

    def test_only_legal_moves_are_compared(self):
        """Padding slots must not contribute drift."""
        from analyse_int8_activation_calibration import compare

        base = {
            "logits": torch.tensor([[1.0, 2.0, 999.0]]),
            "evidence": torch.tensor([[1.0]]),
            "representation": torch.tensor([[0.1]]),
        }
        cand = {
            "logits": torch.tensor([[1.0, 2.0, -999.0]]),
            "evidence": torch.tensor([[1.0]]),
            "representation": torch.tensor([[0.1]]),
        }
        r = compare(base, cand, 2)
        self.assertEqual(r["max_logit_drift"], 0.0)


class CommittedArtifact(unittest.TestCase):
    """The artifact must record the negative result, not a hopeful one."""

    @classmethod
    def setUpClass(cls):
        if not ARTIFACT.exists():
            raise unittest.SkipTest(f"{ARTIFACT} not generated")
        cls.data = json.loads(ARTIFACT.read_text())

    def test_gate_is_the_projects_real_tolerance(self):
        self.assertEqual(self.data["parity_gate"], 5e-3)

    def test_no_whole_model_scheme_passes(self):
        """If one ever does, this test should fail and be re-examined."""
        for fixture, entry in self.data["fixtures"].items():
            for scheme, r in entry["schemes"].items():
                with self.subTest(fixture=fixture, scheme=scheme):
                    self.assertFalse(
                        r["passes_gate"],
                        f"{scheme} now passes on {fixture} -- re-run the study",
                    )

    def test_rejected_baseline_is_reproduced(self):
        """per_token_symmetric was already rejected; it must still fail."""
        for fixture, entry in self.data["fixtures"].items():
            r = entry["schemes"]["per_token_symmetric"]
            self.assertGreater(r["max_logit_drift"], 5e-3)

    def test_calibration_sweep_shows_the_tradeoff(self):
        sweep = self.data["calibration_sweep"]
        self.assertGreaterEqual(len(sweep), 3)
        for row in sweep:
            with self.subTest(n=row["calibration_positions"]):
                self.assertFalse(row["generalizes"])
                self.assertGreater(row["holdout_over_gate"], 0)

    def test_more_calibration_shrinks_the_int8_set(self):
        """The central tradeoff: safety costs coverage."""
        sweep = {r["calibration_positions"]: r for r in self.data["calibration_sweep"]}
        self.assertIn(0, sweep)
        self.assertIn(80, sweep)
        self.assertLess(sweep[80]["mac_fraction_int8"], sweep[0]["mac_fraction_int8"])
        self.assertLess(sweep[80]["holdout_over_gate"], sweep[0]["holdout_over_gate"])

    def test_conclusion_is_recorded_as_negative(self):
        self.assertIn("NEGATIVE", self.data["conclusion"])

    def test_holdout_is_disjoint_from_calibration(self):
        """Calibration reads the corpus tail, holdout the head."""
        source = (TOOLS / "analyse_int8_activation_calibration.py").read_text()
        self.assertIn("_rows[-args.calibrate :]", source)
        self.assertIn("rows[: args.validate]", source)


class ToolIsSelfContained(unittest.TestCase):
    def test_help_runs_without_torch_or_a_checkpoint(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, str(TOOLS / "analyse_int8_activation_calibration.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("int8", result.stdout.lower())

    def test_documents_that_it_needs_torch(self):
        source = (TOOLS / "analyse_int8_activation_calibration.py").read_text()
        self.assertIn("Requires `torch`", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
