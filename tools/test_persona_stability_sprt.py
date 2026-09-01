"""Tests for persona dwell/EMA replica — no torch."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from persona_stability_sprt import PersonaState, decide_legacy, flip_rate, run_legacy, run_stable


class LegacyContractTests(unittest.TestCase):
    def test_defend_hysteresis(self):
        self.assertNotEqual(decide_legacy(-120, 20, "Match", False, False), "Defend")
        self.assertEqual(decide_legacy(-120, 20, "Defend", False, False), "Defend")

    def test_clinch_band(self):
        self.assertEqual(decide_legacy(0, 40, "Match", False, False), "Clinch")


class StableDwellTests(unittest.TestCase):
    def test_one_clinch_vote_stays_match(self):
        s = PersonaState()
        # before move 28 CLINCH cannot fire; establish MATCH
        self.assertEqual(s.update(5, 20, False, False, confidence=200), "Match")
        # first CLINCH-eligible ply is only a vote
        self.assertEqual(s.update(8, 40, False, False, confidence=200), "Match")

    def test_blunder_punish_is_immediate(self):
        s = PersonaState()
        s.update(80, 20, False, False)
        self.assertEqual(s.update(90, 20, True, False), "Punish")

    def test_flip_rate_not_higher_than_legacy_on_noisy_trace(self):
        evals = [0, 40, -10, 55, 5, -30, 20, 8, 70, -5] * 4
        blunders = [False] * len(evals)
        a = run_legacy(evals, blunders)
        b = run_stable(evals, blunders)
        self.assertLessEqual(flip_rate(b), flip_rate(a) + 1e-9)


class ArtifactTests(unittest.TestCase):
    def test_artifact(self):
        p = Path(__file__).resolve().parents[1] / "artifacts" / "persona-stability-sprt.json"
        if not p.exists():
            self.skipTest("run persona_stability_sprt.py first")
        data = json.loads(p.read_text())
        self.assertGreater(data["flip_rate_reduction"], 0.3)
        self.assertTrue(data["engine_contract"]["adaptive_default"])


if __name__ == "__main__":
    unittest.main()
