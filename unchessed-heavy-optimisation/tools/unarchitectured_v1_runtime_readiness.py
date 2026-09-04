#!/usr/bin/env python3
"""Fail closed while the Unarchitectured v1 runtime safety gate is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQUIREMENTS = {
    "canonical_data_module": "tools/unarchitectured_v1_data.py",
    "canonical_student_trainer": "tools/train_unarchitectured_v1_student_a100.py",
    "canonical_oracle_trainer": "tools/train_unarchitectured_v1_a100.py",
    "canonical_teacher_worker": "tools/unarchitectured_v1_uci_teacher_worker.py",
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
# Capabilities that must remain FALSE until the corresponding work is actually
# implemented and gated. These are experimental placeholders: flipping one on
# without the implementation (and its SPRT) is a regression, so readiness
# reports it explicitly rather than leaving it unmentioned.
EXPERIMENTAL_CAPABILITIES = ("npu_dispatch",)


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
    # Experimental capabilities are asserted false. If one is ever flipped true
    # without an implementation behind it, readiness fails closed rather than
    # silently reporting the engine as ready.
    experimental = {
        name: bool(capabilities.get(name)) for name in EXPERIMENTAL_CAPABILITIES
    }
    experimental_violations = sorted(
        name for name, enabled in experimental.items() if enabled
    )
    ready = (
        all(item["exists"] for item in checks.values())
        and all(capability_checks.values())
        and not experimental_violations
    )
    return {
        "schema": 1,
        "ready_for_engine_candidate_training": ready,
        "checks": checks,
        "capability_file": CAPABILITIES,
        "capabilities": capability_checks,
        "experimental_capabilities": experimental,
        "experimental_violations": experimental_violations,
        "warning": (
            "Package and neural-forward infrastructure are insufficient until "
            "the runtime safety capability also passes. Experimental "
            "capabilities (e.g. npu_dispatch) must remain false until a real, "
            "SPRT-gated implementation exists."
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
