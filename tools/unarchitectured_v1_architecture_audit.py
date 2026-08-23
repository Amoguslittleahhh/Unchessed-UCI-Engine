#!/usr/bin/env python3
"""Audit canonical Unarchitectured v1 contracts present in this repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import unarchitectured_v1_package as package

ARCHITECTURE = "config/unarchitectured_v1.json"
SAFETY = "config/unarchitectured_v1_safety.json"
CAPABILITIES = "config/unarchitectured_v1_runtime_capabilities.json"
RUNTIME = "unchessed-core/src/aegis_v4_runtime.rs"
ARTIFACT = "artifacts/unarchitectured-v1-final.unarchv1"
CANONICAL_SCRIPTS = (
    "tools/unarchitectured_v1_base_data.py",
    "tools/unarchitectured_v1_data.py",
    "tools/unarchitectured_v1_uci_teacher_worker.py",
    "tools/train_unarchitectured_v1_student_a100.py",
    "tools/train_unarchitectured_v1_a100.py",
    "tools/calibrate_unarchitectured_v1_throughput.py",
    "tools/reference_forward_unarchitectured_v1.py",
    "tools/unarchitectured_v1_runtime_readiness.py",
)
LEGACY_ENTRYPOINTS = (
    "tools/aegis_v3_data.py",
    "tools/aegis_v4_data.py",
    "tools/v5_uci_teacher_worker.py",
    "tools/train_chessformer_v4_a100.py",
    "tools/train_hydra_oracle_v5_a100.py",
    "tools/reference_forward_aegis_v4.py",
    "tools/v5_runtime_readiness.py",
)


def load_json(root: Path, relative_path: str):
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def audit(root=ROOT):
    """Return deterministic checks without depending on uncommitted tooling."""

    root = Path(root)
    architecture = load_json(root, ARCHITECTURE)
    safety = load_json(root, SAFETY)
    capabilities = load_json(root, CAPABILITIES)
    runtime = architecture["runtime_student"]
    exits = runtime["exit_layers"]
    widths = runtime["matryoshka_widths"]

    checks = {
        "canonical_name": architecture["name"] == "Unarchitectured v1",
        "canonical_magic_config": architecture["runtime_file_magic"] == "UNARCHV1",
        "canonical_magic_package": package.MAGIC == b"UNARCHV1",
        "student_exit_pairing": len(exits) == len(widths),
        "student_full_exit": exits[-1] == runtime["layers"]
        and widths[-1] == runtime["d_model"],
        "student_head_divisibility": all(
            width % runtime["heads"] == 0 for width in widths
        ),
        "legal_action_vocabulary": runtime["policy_action_vocabulary"]
        == 64 * 64 * runtime["promotion_classes"],
        "maximum_legal_moves": runtime["maximum_legal_moves"] == 218,
        "runtime_storage_int8": runtime["runtime_storage_bits"] == 8,
        "alpha_beta_authoritative": runtime["alpha_beta_authoritative"] is True,
        "safety_fail_closed": safety["fail_closed"] is True,
        "runtime_capability_schema": capabilities["architecture"]
        == "Unarchitectured v1",
        "scalar_forward_available": capabilities["scalar_neural_forward"] is True,
        "quantized_forward_available": capabilities["quantized_neural_forward"]
        is True,
        "reference_vectors_available": capabilities["exported_reference_vectors"]
        is True,
        "runtime_safety_still_blocked": capabilities["runtime_safety_suite"]
        is False,
        "runtime_source_present": (root / RUNTIME).is_file(),
        "real_artifact_present": (root / ARTIFACT).is_file(),
        "canonical_scripts_present": all(
            (root / path).is_file() for path in CANONICAL_SCRIPTS
        ),
        "legacy_entrypoints_absent": all(
            not (root / path).exists() for path in LEGACY_ENTRYPOINTS
        ),
    }
    return {
        "schema": 1,
        "architecture": "Unarchitectured v1",
        "passed": all(checks.values()),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise SystemExit("architecture audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
