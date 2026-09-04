#!/usr/bin/env python3
"""Misfire tests for the live Elo detector at high-level play."""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo_detector import OpponentModel, elo_sample


def play(cp_losses, times=None, seed_kind=None, seed_elo=None, seed_name=""):
    m = OpponentModel()
    if seed_kind:
        m.seed(seed_kind, seed_elo, seed_name)
    for i, cp in enumerate(cp_losses):
        m.observe(cp, 1.0)
        if times is not None:
            m.observe_time(times[i], True)
    return m


class CurveTests(unittest.TestCase):
    def test_zero_loss_is_near_ceiling(self):
        self.assertGreater(elo_sample(0), 2900)
        self.assertLess(elo_sample(300), 1200)


class HighLevelMisfireTests(unittest.TestCase):
    def test_maia_computer_is_match_not_full(self):
        m = OpponentModel()
        m.seed("computer", 1600, "Maia 2")
        self.assertFalse(m.engine_suspect())

    def test_stockfish_computer_is_full(self):
        m = OpponentModel()
        m.seed("computer", 3644, "Stockfish 16")
        self.assertTrue(m.engine_suspect())

    def test_ten_clean_opening_moves_do_not_flag(self):
        m = play([6] * 10)
        self.assertFalse(m.engine_suspect())

    def test_declared_2500_human_clean_game_does_not_flag(self):
        m = play([10] * 24, seed_kind="human", seed_elo=2500, seed_name="IM")
        self.assertFalse(m.engine_suspect())
        self.assertGreater(m.estimate(), 2000)

    def test_opening_premoves_do_not_clock_flag(self):
        m = play([8] * 7, times=[80] * 7)
        self.assertFalse(m.engine_suspect())

    def test_middlegame_instant_engine_does_flag(self):
        cps = [8] * 8 + [5] * 4
        times = [2000] * 8 + [80] * 4
        m = play(cps, times)
        self.assertTrue(m.engine_suspect())

    def test_anonymous_long_perfect_low_vol_flags(self):
        m = play([4] * 20)
        self.assertTrue(m.engine_suspect())


class BandSimulationTests(unittest.TestCase):
    def test_human_bands_misfire_rate(self):
        rng = random.Random(20260901)
        # typical human ACP loss by rating (rough, from Maia/lichess folklore)
        bands = {
            1200: 90,
            1600: 55,
            2000: 35,
            2200: 25,
            2500: 18,
        }
        report = {}
        for elo, mean_cp in bands.items():
            flags = 0
            n = 200
            for _ in range(n):
                cps = [max(0, int(rng.gauss(mean_cp, mean_cp * 0.6))) for _ in range(24)]
                times = [int(rng.uniform(400, 8000)) for _ in range(24)]
                m = play(cps, times, seed_kind="human", seed_elo=elo, seed_name="player")
                if m.engine_suspect():
                    flags += 1
            report[str(elo)] = {"n": n, "engine_flags": flags, "rate": flags / n}
            self.assertEqual(flags, 0, f"human {elo} misfires: {flags}/{n}")
        dest = Path(__file__).resolve().parents[1] / "artifacts" / "elo-detector-misfire.json"
        dest.write_text(json.dumps({"human_declared_bands": report, "seed": 20260901}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
