#!/usr/bin/env python3
"""Regression tests for train_nnue_xt_a100.py's two pre-GPU-spend bugs found
by the smoke test: an undefined `batch` in ThreatIndexer.indices(), and
phase_stack computation hardcoding 8 heads regardless of the model's actual
phase_stacks config (which selfcheck deliberately reduces for speed)."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for a100_common
MODULE_PATH = Path(__file__).with_name("train_nnue_xt_a100.py")
SPEC = importlib.util.spec_from_file_location("train_nnue_xt_a100_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainNnueXtA100Tests(unittest.TestCase):
    def test_selfcheck_runs_end_to_end_on_cpu(self):
        # This is exactly the pre-flight gate that must pass before any real
        # GPU time is spent: synthetic data, reduced (2-head) phase_stacks,
        # one full forward+backward+optimizer step.
        class Args:
            config = str(Path(__file__).with_name("..") / "config" / "a100_hybrid_training.json")

        args = Args()
        args.config = str(
            Path(__file__).resolve().parents[1] / "config" / "a100_hybrid_training.json"
        )
        MODULE.selfcheck(args)  # raises/asserts internally; no exception == pass

    def test_threat_indexer_indices_does_not_reference_undefined_batch(self):
        import torch

        device = torch.device("cpu")
        indexer = MODULE.ThreatIndexer(device)
        records = MODULE.synthetic_records(8)
        import numpy as np

        bitboards = torch.from_numpy(
            np.ascontiguousarray(records["bb"]).view(np.int64)
        )
        bits = MODULE.unpack_planes(bitboards)
        # Must not raise NameError: name 'batch' is not defined.
        indices, offsets, counts = indexer.indices(bits, bitboards)
        self.assertEqual(offsets.shape[0], bits.shape[0] + 1)

    def test_make_batch_respects_non_default_phase_stacks(self):
        import torch

        device = torch.device("cpu")
        fixed = MODULE.constants(device)
        indexer = MODULE.ThreatIndexer(device)
        records = MODULE.synthetic_records(64, seed=11)
        for phase_stacks in (2, 4, 8):
            batch = MODULE.make_batch(records, device, fixed, indexer, phase_stacks)
            phase_stack = batch[10]
            self.assertTrue(int(phase_stack.max()) < phase_stacks)
            self.assertTrue(int(phase_stack.min()) >= 0)


if __name__ == "__main__":
    unittest.main()
