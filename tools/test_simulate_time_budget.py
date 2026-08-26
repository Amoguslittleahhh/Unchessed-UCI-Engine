#!/usr/bin/env python3
"""Tests for tools/simulate_time_budget.py.

The transcription is pinned two ways:

  * the exact (soft, hard) values that the Rust test
    `budget_speeds_up_as_clock_drains` (unchessed-core/src/search.rs)
    asserts inequalities for — if the Rust formula ever changes, these
    values must be re-derived and updated here deliberately;
  * the movetime path and the situation factor, which have no Rust test
    and are pinned by hand computation from the code.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from simulate_time_budget import budget, final_soft, situation_factor  # noqa: E402


class TestBudgetTranscription:
    """Exact values implied by the Rust test's fixtures (inc=0, no movestogo)."""

    def test_rust_test_fixture_values(self):
        # Rust: soft_for(180_000) / (20_000) / (5_000) / (1_000), inc = 0.
        assert budget(180_000, 0)[0] == 6_000
        assert budget(20_000, 0)[0] == 666
        assert budget(5_000, 0)[0] == 83
        assert budget(1_000, 0)[0] == 16

    def test_rust_test_inequalities(self):
        # The assertions as written in search.rs.
        full, mid, low, panic = (
            budget(180_000, 0)[0],
            budget(20_000, 0)[0],
            budget(5_000, 0)[0],
            budget(1_000, 0)[0],
        )
        assert full > mid and mid > low and low >= panic
        assert low / 5_000.0 < full / 180_000.0 * 0.8
        assert panic <= 40
        assert budget(300, 0)[1] <= 300 - 60 + 5

    def test_rust_test_fixture_hards(self):
        assert budget(180_000, 0)[1] == 36_000
        assert budget(300, 0)[1] == 18

    def test_soft_never_exceeds_hard_and_floors(self):
        for t in (100, 300, 1_000, 5_000, 20_000, 100_000):
            for inc in (0, 50, 600):
                soft, hard = budget(t, inc)
                assert 3 <= soft <= hard
                assert hard >= 5

    def test_movetime_path(self):
        # t = movetime - 25, floored at 5.
        assert budget(500, 0, movetime=500) == (475, 475)
        assert budget(500, 0, movetime=25) == (5, 5)
        assert budget(500, 0, movetime=20) == (5, 5)

    def test_low_clock_tiers_shrink_fraction(self):
        # 5+0.05 profile: the soft fraction of the clock must not grow as
        # the clock drains (that is the whole point of the urgency tiers).
        assert 100.0 * budget(10_000, 50)[0] / 10_000 < 100.0 * budget(300_000, 50)[0] / 300_000
        assert 100.0 * budget(3_000, 50)[0] / 3_000 < 100.0 * budget(300_000, 50)[0] / 300_000

    def test_panic_mode_uses_increment(self):
        # t < 2000: soft = min(base, t/35+inc/2, t/60+inc/2, max(inc/2, 30)).
        assert budget(1_000, 50)[0] == 30   # max(25, 30) wins at 30
        assert budget(1_000, 100)[0] == 50  # max(50, 30) = 50 is reachable
        assert budget(1_000, 200)[0] == 100
        assert budget(1_000, 0)[0] == 16    # tiers below the floor win


class TestSituationFactor:
    def test_width_clamps(self):
        assert situation_factor(2, False) == 0.75  # 0.694 clamped up
        assert abs(situation_factor(5, False) - (0.65 + 5 / 45.0)) < 1e-12
        assert situation_factor(30, False) == 1.3  # 1.317 clamped down
        assert situation_factor(100, False) == 1.3

    def test_in_check_multiplies_after_width_clamp(self):
        # Rust clamps the width to 1.3, then multiplies by 1.25 -> 1.625,
        # which is NOT re-clamped.
        assert abs(situation_factor(30, True) - 1.3 * 1.25) < 1e-12

    def test_final_soft_clamps_to_soft_hard_range(self):
        soft, hard = budget(300_000, 50)
        final = final_soft(soft, hard, 30, False)
        assert 3 <= final <= hard
        assert final == int(soft * 1.3)


class TestCli:
    def test_help_runs_standalone(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "simulate_time_budget.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0
        assert "time management" in out.stdout.lower() or "budget" in out.stdout.lower()

    def test_table_rows_match_library(self):
        out = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "simulate_time_budget.py"),
                "--t", "300000,1000",
                "--inc", "50,50",
                "--legal-count", "30",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0
        lines = [l for l in out.stdout.splitlines() if l.strip() and not l.strip().startswith("time ms")]
        for line, t, inc in zip(lines, (300_000, 1_000), (50, 50)):
            soft, hard = budget(t, inc)
            final = final_soft(soft, hard, 30, False)
            assert f"{soft:>6}" in line
            assert f"{hard:>7}" in line
            assert f"{final:>10}" in line

    def test_only_stdlib_imports(self):
        import ast

        src = (TOOLS / "simulate_time_budget.py").read_text()
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        assert mods == {"__future__", "argparse", "sys"}, mods
