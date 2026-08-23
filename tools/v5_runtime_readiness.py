#!/usr/bin/env python3
"""Fail closed while the Unarchitectured v1 runtime safety gate is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQUIREMENTS = {
    "checkpoint_exporter": "tools/export_unarchitectured_v1.py",
    "package_inspector": "tools/inspect_unarchitectured_v1.py",
    "rust_package_loader": "unchessed-core/src/unarchitectured_v1.rs",
    "runtime_package_tests": "tools/test_unarchitectured_v1_runtime.py",
}
CAPABILITIES = "config/unarchitectured_v1_runtime_capabilities.json"
REQUIRED_CAPABILITIES = (
    "container_format",
    "checkpoint_exporter",
    "package_inspector",
    "rust_package_loader",
    "tensor_quantization_drift",
    "scalar_neural_forward",
    "quantized_neural_forward",
    "exported_reference_vectors",
    "runtime_safety_suite",
)


def readiness(root=ROOT):
    root = Path(root)
    checks = {
        name: {"path": path, "exists": (root / path).is_file()}
        for name, path in REQUIREMENTS.items()
    }
    capability_path = root / CAPABILITIES
    capabilities = (
        json.loads(capability_path.read_text(encoding="utf-8"))
        if capability_path.is_file()
        else {}
    )
    capability_checks = {
        name: bool(capabilities.get(name)) for name in REQUIRED_CAPABILITIES
    }
    ready = all(item["exists"] for item in checks.values()) and all(
        capability_checks.values()
    )
    return {
        "schema": 1,
        "ready_for_engine_candidate_training": ready,
        "checks": checks,
        "capability_file": CAPABILITIES,
        "capabilities": capability_checks,
        "warning": (
            "Package and neural-forward infrastructure are insufficient until "
            "the runtime safety capability also passes."
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
        raise SystemExit("Unarchitectured v1 runtime safety is incomplete")


if __name__ == "__main__":
    main()
