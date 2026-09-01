"""Replica of OpponentModel Elo detector (adapt.rs) for misfire tests."""
from __future__ import annotations

import math


def elo_sample(cp_loss: float) -> float:
    return max(400.0, min(3200.0, 2950.0 - 850.0 * math.log(1.0 + cp_loss / 20.0)))


class OpponentModel:
    def __init__(self):
        self.mean = 1500.0
        self.weight = 1.0
        self.samples = 0
        self.last_cp_loss = None
        self.is_computer = False
        self.declared_elo = None
        self.var_accum = 90_000.0
        self.suspicion = 0.0
        self.prev_mean = 1500.0
        self.low_loss_streak = 0

    def seed(self, kind: str, elo: int | None, name: str = ""):
        self.is_computer = kind == "computer"
        self.declared_elo = elo
        if self.is_computer:
            table = {
                "stockfish": 3600,
                "maia": 1600,
                "rubichess": 3300,
            }
            known = next((v for k, v in table.items() if k in name.lower()), None)
            seed = elo or known or 2800
            self.mean = float(seed)
            self.weight = 6.0
            self.prev_mean = self.mean
        elif elo:
            self.mean = 1500.0 * 0.5 + elo * 0.5
            self.prev_mean = self.mean

    def observe(self, cp_loss: int, difficulty_weight: float = 1.0):
        self.prev_mean = self.mean
        self.last_cp_loss = cp_loss
        opening = 0.5 if self.samples < 8 else 1.0
        w = min(2.0, max(0.05, difficulty_weight * opening))
        sample = elo_sample(cp_loss)
        dev = sample - self.mean
        self.var_accum = 0.88 * self.var_accum + 0.12 * dev * dev
        self.mean = (self.mean * self.weight + sample * w) / (self.weight + w)
        self.weight = min(14.0, self.weight + w)
        self.weight *= 0.985
        self.samples += 1
        if cp_loss <= 40:
            self.low_loss_streak += 1
        else:
            self.low_loss_streak = 0

    def observe_time(self, used_ms: int, had_choice: bool):
        strong = self.last_cp_loss is not None and self.last_cp_loss <= 60
        in_opening = self.samples < 8
        if used_ms < 300 and strong and had_choice and not in_opening:
            self.suspicion += 1.0
        elif used_ms > 1500:
            self.suspicion = max(0.0, self.suspicion - 0.5)

    def volatility(self) -> int:
        return int(round(math.sqrt(self.var_accum)))

    def estimate(self) -> int:
        return int(round(self.mean))

    def engine_suspect(self) -> bool:
        if self.is_computer:
            return self.mean >= 2400.0
        if self.suspicion >= 4.0:
            return True
        if self.declared_elo is not None:
            return False
        return (
            self.weight >= 11.0
            and self.samples >= 16
            and self.mean >= 2500.0
            and self.low_loss_streak >= 12
        )
