"""NNUE training-loop controls: early stopping and the production LR schedule.

Pure Python, no torch. The trainer (`train_nnue.py`) and its tests both
import this so the control logic can be checked without a GPU (or even
without torch installed).

Why this file exists: the reviewer follow-up
(`docs/nnue-v4-retrain-data-scaling-finding.md`) trained at three data
scales and SPRT'd each net against the shipped default. Every run
exported the *last* epoch while val-MAE was already climbing — the
trainer had no best-checkpoint export and no early stop. That is the
same class of failure as F-01/F-03 in
`docs/full-scale-bug-audit-2026-08-21.md` (copied epoch count; worsening
validation burning the rest of the run). These helpers are the fix, and
the tests in `test_nnue_train_control.py` pin the exact behavior.
"""

from __future__ import annotations

import os


DEFAULT_EARLY_STOP_PATIENCE = 3
DEFAULT_EARLY_STOP_MIN_DELTA_CP = 0.1
DEFAULT_BASE_LR = 1e-3
DEFAULT_LR_GAMMA = 0.3
LR_DROP_FRACTIONS = (0.6, 0.8)


def recipe_from_env():
    """Read EARLY_STOP_PATIENCE / EARLY_STOP_MIN_DELTA from the environment.

    patience 0 disables early stop (still tracks the best metric so the
    caller can export the best checkpoint). Negative values are rejected.
    """
    patience = int(os.environ.get("EARLY_STOP_PATIENCE", str(DEFAULT_EARLY_STOP_PATIENCE)))
    min_delta = float(
        os.environ.get("EARLY_STOP_MIN_DELTA", str(DEFAULT_EARLY_STOP_MIN_DELTA_CP))
    )
    if patience < 0:
        raise ValueError(f"EARLY_STOP_PATIENCE must be >= 0, got {patience}")
    if min_delta < 0:
        raise ValueError(f"EARLY_STOP_MIN_DELTA must be >= 0, got {min_delta}")
    return patience, min_delta


def lr_drop_epochs(max_epochs):
    """0-based epoch indices at which LR drops, matching train_nnue.py's
    historical `int(epochs * 0.6)` / `int(epochs * 0.8)` schedule.

    For the production 15-epoch cap this is epochs 10 and 13 (1-based),
    i.e. indices 9 and 12.
    """
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
    return tuple(int(max_epochs * f) for f in LR_DROP_FRACTIONS)


def lr_at_epoch(epoch_index, max_epochs, base_lr=DEFAULT_BASE_LR, gamma=DEFAULT_LR_GAMMA):
    """Learning rate for a 0-based epoch under the production step-decay.

    Kept as a function of the *cap* (max_epochs), not of how long the run
    actually lasts. If early-stop fires before the first drop, LR never
    drops — that is intentional: we already started overfitting at the
    high LR, so a late fine-tuning phase is not what we want.
    """
    drop1, drop2 = lr_drop_epochs(max_epochs)
    n_drops = (epoch_index >= drop1) + (epoch_index >= drop2)
    return base_lr * (gamma ** n_drops)


class EarlyStop:
    """Lower-is-better early stop with a minimum-delta and a patience.

    patience=0 means "never stop", but `update` still reports `is_best`
    so the trainer can export the best checkpoint of a full run.
    """

    def __init__(self, patience, min_delta=DEFAULT_EARLY_STOP_MIN_DELTA_CP):
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.best_epoch = None  # 1-based
        self.bad = 0
        self.epoch = 0  # 1-based, incremented on every update

    def update(self, metric):
        """Observe one epoch's metric.

        Returns (is_best, should_stop). `metric` is val-MAE in centipawns.
        """
        if metric != metric:  # NaN
            raise ValueError("early-stop metric is NaN")
        self.epoch += 1
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.best_epoch = self.epoch
            self.bad = 0
            return True, False
        self.bad += 1
        should_stop = self.patience > 0 and self.bad >= self.patience
        return False, should_stop
