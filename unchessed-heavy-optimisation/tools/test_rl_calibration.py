from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import rl_calibration as cal


ROOT = Path(__file__).resolve().parent


def test_replay_is_legal_and_reproducible():
    first, counts = cal.generate(seed=17, games=20)
    second, counts2 = cal.generate(seed=17, games=20)
    assert first == second
    assert counts == counts2
    assert counts["games_completed"] == 20
    assert counts["legal_mask_violations"] == 0
    assert first
    assert all(record.action in record.legal_actions for record in first)


def test_value_learning_improves_heldout_loss():
    report = cal.run(seed=17, games=100, updates=25, learning_rate=0.05)
    assert report["heldout_loss_decreased"] is True
    assert report["heldout_loss_after"] < report["heldout_loss_before"]
    assert report["no_elo_claim"] is True


def test_cli_writes_machine_readable_report(tmp_path):
    output = tmp_path / "calibration.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "rl_calibration.py"), "--games", "20", "--json", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "unchessed.rl-calibration.v1"
    assert report["replay_sha256"]
    assert "heldout_loss_after" in proc.stdout


def test_invalid_configuration_is_rejected():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "rl_calibration.py"), "--games", "1"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "games >= 2" in proc.stderr
