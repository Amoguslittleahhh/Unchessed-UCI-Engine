#!/usr/bin/env python3
"""Tests for tools/check_search_param_consistency.py.

Strategy: the tool is a pure text-level cross-check, so the negative
controls are one-character mutations of copies of the three Rust files in a
scratch repo — every mutation class the linter exists to catch must flip
the exit code and be named.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
TOOL = TOOLS / "check_search_param_consistency.py"

FILES = (
    "unchessed-core/src/search.rs",
    "unchessed-core/src/uci.rs",
    "unchessed-core/src/eval.rs",
)


def make_repo(tmp: Path) -> Path:
    for rel in FILES:
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    return tmp


def run_tool(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--repo", str(repo), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestRealRepo:
    def test_is_consistent(self):
        out = run_tool(REPO)
        assert out.returncode == 0, out.stdout + out.stderr
        data = json.loads(out.stdout)
        assert data["failures"] == []
        assert data["checked"] == 20
        # spot-check three of the pinned values
        assert data["params"]["rfpmargin"]["struct_default"] == 90
        assert data["params"]["futilitymargin"]["clamp"] == [30, 400]
        assert data["params"]["unarchitecturedmintime"]["struct_default"] == 30_000

    def test_every_spin_param_has_all_three_sides(self):
        out = run_tool(REPO)
        data = json.loads(out.stdout)
        for name, rec in data["params"].items():
            if (rec.get("advertised") or {}).get("type") == "spin":
                assert rec["struct_default"] is not None, name
                assert rec["clamp"] is not None, name


class TestDriftDetection:
    def test_advertised_default_drift(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        p.write_text(src.replace(
            "option name AspirationDelta type spin default 25 min 5 max 200",
            "option name AspirationDelta type spin default 30 min 5 max 200",
        ))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("aspirationdelta" in f for f in json.loads(out.stdout)["failures"])

    def test_struct_default_drift(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/search.rs"
        src = p.read_text()
        assert "aspiration_delta: 25," in src
        p.write_text(src.replace("aspiration_delta: 25,", "aspiration_delta: 26,"))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("aspirationdelta" in f for f in json.loads(out.stdout)["failures"])

    def test_clamp_vs_advertisement_mismatch(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        p.write_text(src.replace(
            "option name ProbCutMargin type spin default 200 min 50 max 400",
            "option name ProbCutMargin type spin default 200 min 50 max 500",
        ))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("probcutmargin" in f for f in json.loads(out.stdout)["failures"])

    def test_default_outside_advertised_bounds(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        p.write_text(src.replace(
            "option name AspirationDelta type spin default 25 min 5 max 200",
            "option name AspirationDelta type spin default 25 min 30 max 200",
        ))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("outside" in f and "aspirationdelta" in f
                   for f in json.loads(out.stdout)["failures"])

    def test_mintime_clamp_drift(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        p.write_text(src.replace(
            "milliseconds.clamp(1_000, 600_000)",
            "milliseconds.clamp(1_000, 300_000)",
        ))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("unarchitecturedmintime" in f for f in json.loads(out.stdout)["failures"])

    def test_mintime_struct_literal_drift(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        p.write_text(src.replace(
            "unarchitectured_min_time_ms: 30_000,",
            "unarchitectured_min_time_ms: 10_000,",
        ))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("unarchitecturedmintime" in f for f in json.loads(out.stdout)["failures"])

    def test_check_option_drift(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        p.write_text(src.replace(
            "option name ProbcutSeeFilter type check default false",
            "option name ProbcutSeeFilter type check default true",
        ))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("probcutseefilter" in f for f in json.loads(out.stdout)["failures"])

    def test_missing_handler_clamp(self, tmp_path):
        repo = make_repo(tmp_path)
        p = repo / "unchessed-core/src/uci.rs"
        src = p.read_text()
        # remove the rookpct clamp entirely
        needle = "\"rookpct\" => {\n            if let Ok(v) = value.parse::<i32>() {\n                opt.eval_params.rook_pct = v.clamp(0, 200);"
        replacement = "\"rookpct\" => {\n            if let Ok(v) = value.parse::<i32>() {\n                opt.eval_params.rook_pct = v.min(200);"
        assert needle in src
        p.write_text(src.replace(needle, replacement))
        out = run_tool(repo)
        assert out.returncode == 1
        assert any("rookpct" in f for f in json.loads(out.stdout)["failures"])


class TestCli:
    def test_help_runs_standalone(self):
        out = subprocess.run(
            [sys.executable, str(TOOL), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0
        assert "UCI" in out.stdout

    def test_only_stdlib_imports(self):
        import ast

        src = TOOL.read_text()
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        assert mods == {"__future__", "argparse", "json", "re", "sys", "pathlib"}, mods
