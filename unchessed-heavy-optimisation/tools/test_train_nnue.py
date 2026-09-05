"""Integration tests for train_nnue.py's best-checkpoint export.

Skipped when torch is absent — torch is not in requirements-dev.txt on
purpose (see that file). The control-loop behavior without torch is in
test_nnue_train_control.py and always runs.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError as error:
    torch = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"torch not installed ({_IMPORT_ERROR})")
class TrainNnueBestCheckpointTests(unittest.TestCase):
    def test_grouped_position_split_keeps_duplicates_together(self):
        import train_nnue as t

        records = t.synth_records(200, t.np.random.default_rng(11))
        records[100:120] = records[0:20]
        val, train = t.grouped_position_split(records)
        val_boards = {bytes(records[i]["bb"].tobytes()) for i in val}
        train_boards = {bytes(records[i]["bb"].tobytes()) for i in train}
        self.assertTrue(val_boards.isdisjoint(train_boards))
        self.assertGreater(len(val), 0)
        self.assertGreater(len(train), 0)

    def test_train_exports_best_epoch_not_last(self):
        import train_nnue as t

        rng = t.np.random.default_rng(0)
        records = t.synth_records(1200, rng)
        fd, shard = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        fd, out = tempfile.mkstemp(suffix=".nnue.bin")
        os.close(fd)
        records.tofile(shard)

        # Predetermined val-MAE sequence: best at epoch 2, then three
        # misses. With default patience=3 the run must stop at epoch 5
        # and export epoch 2, not epoch 5.
        maes = [20.0, 10.0, 10.5, 11.0, 12.0, 13.0]
        losses = [0.02] * len(maes)
        state = {"i": 0}

        def fake_eval(model, batch_iter):
            for _ in batch_iter:
                pass
            i = state["i"]
            state["i"] += 1
            return losses[i], maes[i]

        env_patch = patch.dict(
            os.environ,
            {"EARLY_STOP_PATIENCE": "3", "EARLY_STOP_MIN_DELTA": "0.1", "BATCH_SIZE": "256"},
            clear=False,
        )
        try:
            with env_patch, patch.object(t, "evaluate_iter", fake_eval), patch.object(
                t, "BATCH_SIZE", 256
            ):
                result = t.train([shard], out, epochs=10)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["last_epoch"], 5)
            self.assertTrue(result["stopped_early"])
            self.assertAlmostEqual(result["best_val_mae"], 10.0)
            with open(out, "rb") as f:
                blob = f.read()
            self.assertEqual(blob[:8], t.MAGIC)
            version, ft_in, acc = struct.unpack("<III", blob[8:20])
            self.assertEqual(version, t.VERSION)
            self.assertEqual(ft_in, t.FT_IN)
            self.assertEqual(acc, t.ACC)
            self.assertEqual(len(blob), t.HEADER_SIZE + t.PAYLOAD_FLOATS * 4)
        finally:
            os.unlink(shard)
            os.unlink(out)


if __name__ == "__main__":
    unittest.main()
