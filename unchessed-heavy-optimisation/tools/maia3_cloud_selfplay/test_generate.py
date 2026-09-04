#!/usr/bin/env python3
"""Tests for tools/maia3_cloud_selfplay/generate.py.

Hermetic parts (core maps, worker-count logic) run anywhere; the
self-play mini run needs the maia3 ONNX model (set MAIA3_ONNX, or it
is found at the /tmp/smi staging path used by the sandbox) and
skips otherwise.

Run:  /tmp/venv/bin/python -m pytest tools/maia3_cloud_selfplay/test_generate.py -q
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import generate  # noqa: E402

MODEL_CANDIDATES = [
    os.environ.get("MAIA3_ONNX", ""),
    "/tmp/smi/simple_maia3_inference/maia3_simplified.onnx",
]
MODEL = next((Path(m) for m in MODEL_CANDIDATES if m and Path(m).exists()),
             None)


# ----------------------------------------------------------------------
# core maps (180/360-vCPU pinning logic)
# ----------------------------------------------------------------------

def test_core_maps_360_mixed():
    maps = generate._core_maps(175, 0, True, "auto", ncores=360)
    assert maps is not None and len(maps) == 175
    for i, cores in enumerate(maps):
        assert cores == [i * 2, i * 2 + 1]
    # the parent keeps the last 10 cores
    assert max(max(m) for m in maps) < 360 - 10


def test_core_maps_360_maia_only():
    maps = generate._core_maps(350, 0, False, "auto", ncores=360)
    assert maps is not None and len(maps) == 350
    assert maps[7] == [7]
    assert max(max(m) for m in maps) < 360 - 10


def test_core_maps_disabled_cases():
    assert generate._core_maps(10, 0, True, "off", ncores=360) is None
    assert generate._core_maps(2, 1, False, "auto", ncores=360) is None
    # not enough free cores (10-core sandbox, 12 workers x 1)
    assert generate._core_maps(12, 0, False, "auto", ncores=10) is None


# ----------------------------------------------------------------------
# end-to-end mini self-play (needs the maia3 model)
# ----------------------------------------------------------------------

@pytest.mark.skipif(MODEL is None,
                    reason="maia3 ONNX model not available "
                           "(set MAIA3_ONNX)")
def test_mini_selfplay_generate_validate(tmp_path):
    out = tmp_path / "mini"
    cmd = [sys.executable, str(HERE / "generate.py"), "generate",
           "--model", str(MODEL), "--engines", "maia3", "--out", str(out),
           "--games", "12", "--workers", "2", "--max-ply", "60",
           "--seed", "20260828", "--cpu-affinity", "auto"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["games"] == 12
    assert manifest["skipped_games"] == 0
    assert manifest["label_rows"] > 0
    report = json.loads((out / "calibration.json").read_text())
    assert report["error_count"] == 0
    assert report["games_checked"] == 12
    # every row is a calibrated maia3 row
    for s in manifest["shards"]:
        assert s["pgn_sha256"] and s["labels_sha256"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
