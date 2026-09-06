#!/usr/bin/env python3
"""Autonomous fail-closed safety controller for Unarchitectured Metal training."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class SafetyDecision:
    action: str
    reason: str = ""

    @property
    def safe(self) -> bool:
        return self.action == "continue"


class TrainingSafetyController:
    def __init__(self, policy: dict):
        if policy.get("schema") != 1 or not policy.get("fail_closed"):
            raise ValueError("Unarchitectured Metal requires fail-closed safety schema 1")
        self.policy = policy
        self.config = policy["training"]
        self.loss_ema = None
        self.best_validation = float("inf")
        self.stale_validations = 0
        self.validation_cusum = 0.0
        self.last_decision = SafetyDecision("continue")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def check_step(self, loss: float, gradient_norm: float) -> SafetyDecision:
        if not math.isfinite(loss) or not math.isfinite(gradient_norm):
            return self._decide("abort", "non-finite loss or gradient norm")
        if gradient_norm > self.config["maximum_gradient_norm"]:
            return self._decide(
                "abort",
                f"gradient norm {gradient_norm:.4g} exceeds "
                f"{self.config['maximum_gradient_norm']:.4g}",
            )
        if self.loss_ema is not None:
            limit = max(1e-8, abs(self.loss_ema)) * self.config["maximum_loss_to_ema_ratio"]
            if abs(loss) > limit:
                return self._decide(
                    "abort",
                    f"absolute loss {abs(loss):.6g} exceeds EMA spike limit {limit:.6g}",
                )
        decay = self.config["loss_ema_decay"]
        self.loss_ema = loss if self.loss_ema is None else decay * self.loss_ema + (1 - decay) * loss
        return self._decide("continue")

    def check_validation(self, value: float) -> SafetyDecision:
        if not math.isfinite(value):
            return self._decide("abort", "non-finite validation metric")
        delta = self.config["validation_min_delta"]
        if value < self.best_validation - delta:
            self.best_validation = value
            self.stale_validations = 0
            self.validation_cusum = 0.0
            return self._decide("continue")
        self.stale_validations += 1
        degradation = max(
            0.0,
            value - self.best_validation - self.config["validation_cusum_drift"],
        )
        self.validation_cusum += degradation
        if self.validation_cusum >= self.config["validation_cusum_threshold"]:
            return self._decide(
                "early_stop",
                f"validation degradation CUSUM {self.validation_cusum:.6g} crossed threshold",
            )
        if self.stale_validations >= self.config["validation_patience"]:
            return self._decide(
                "early_stop",
                f"validation failed to improve for {self.stale_validations} epochs",
            )
        return self._decide("continue")

    def _decide(self, action: str, reason: str = "") -> SafetyDecision:
        self.last_decision = SafetyDecision(action, reason)
        return self.last_decision

    def snapshot(self) -> dict:
        return {
            "loss_ema": self.loss_ema,
            "best_validation": self.best_validation,
            "stale_validations": self.stale_validations,
            "validation_cusum": self.validation_cusum,
            "last_decision": dataclasses.asdict(self.last_decision),
        }


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_heartbeat(path: str | Path, phase: str, payload: dict) -> None:
    atomic_json(
        path,
        {
            "schema": 1,
            "architecture": "Unarchitectured Metal",
            "phase": phase,
            "unix_time": time.time(),
            "monotonic_time": time.monotonic(),
            **payload,
        },
    )
