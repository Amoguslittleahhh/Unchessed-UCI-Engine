#!/usr/bin/env python3
"""Fail closed before paid v5 training when engine integration is unavailable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQUIREMENTS = {
    "quantized_exporter": "tools/export_unarchitectured_v1.py",
    "package_inspector": "tools/inspect_unarchitectured_v1.py",
    "rust_scalar_runtime": "unchessed-core/src/unarchitectured_v1.rs",
    "quantization_drift_gate": "tools/validate_unarchitectured_v1_quantization.py",
    "runtime_safety_tests": "tools/test_unarchitectured_v1_runtime.py",
}


def readiness(root=ROOT):
    checks = {
        name: {"path": path, "exists": (root / path).is_file()}
        for name, path in REQUIREMENTS.items()
    }
    return {
        "schema": 1,
        "ready_for_engine_candidate_training": all(
            item["exists"] for item in checks.values()
        ),
        "checks": checks,
        "warning": (
            "A training checkpoint is not engine-loadable without exporter, "
            "inspector, scalar runtime, quantization drift, and safety gates."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = readiness()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not report["ready_for_engine_candidate_training"]:
        raise SystemExit(
            "v5 runtime pipeline is incomplete; set ALLOW_RESEARCH_CHECKPOINT_ONLY=1 "
            "only if a non-engine-loadable research checkpoint is intentional"
        )


if __name__ == "__main__":
    main()
