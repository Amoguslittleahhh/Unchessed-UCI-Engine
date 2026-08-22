#!/usr/bin/env python3
"""Cross-check every canonical Unarchitectured v1 architecture contract."""

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
import verda_gpu_profile


def close_to_one(values):
    return abs(sum(values) - 1.0) <= 1e-12


def audit(root=ROOT):
    architecture = json.loads((root / "config/unarchitectured_v1.json").read_text())
    training = json.loads(
        (root / "config/unarchitectured_v1_training.json").read_text()
    )
    student_file = json.loads(
        (root / "config/unarchitectured_v1_student.json").read_text()
    )
    profiles = json.loads((root / "config/verda_gpu_profiles.json").read_text())
    registry = json.loads((root / "config/architecture_registry.json").read_text())
    safety = json.loads(
        (root / "config/unarchitectured_v1_safety.json").read_text()
    )
    capabilities = json.loads(
        (root / "config/unarchitectured_v1_runtime_capabilities.json").read_text()
    )
    runtime = architecture["runtime_student"]
    student = student_file["chessformer"]
    mapping = {
        "d_model": "d_model",
        "layers": "layers",
        "heads": "heads",
        "ffn": "ffn",
        "exit_layers": "exit_layers",
        "matryoshka_widths": "matryoshka_widths",
        "policy_adapter_rank": "policy_adapter_rank",
        "history_width": "history_width",
        "history_plies": "history_plies",
        "time_classes": "time_classes",
        "legal_regret_width": "regret_width",
    }
    checks = {
        "canonical_registry": registry["canonical"]["id"] == "unarchitectured-v1",
        "canonical_magic_registry": registry["canonical"]["runtime_magic"]
        == "UNARCHV1",
        "canonical_magic_config": architecture["runtime_file_magic"] == "UNARCHV1",
        "canonical_magic_package": package.MAGIC == b"UNARCHV1",
        "training_architecture_link": training["architecture_config"]
        == "config/unarchitectured_v1.json",
        "training_student_link": training["student_distillation"]["student_config"]
        == "config/unarchitectured_v1_student.json",
        "student_exit_pairing": len(runtime["exit_layers"])
        == len(runtime["matryoshka_widths"]),
        "student_full_exit": runtime["exit_layers"][-1] == runtime["layers"]
        and runtime["matryoshka_widths"][-1] == runtime["d_model"],
        "student_head_divisibility": all(
            width % runtime["heads"] == 0 for width in runtime["matryoshka_widths"]
        ),
        "legal_action_vocabulary": runtime["policy_action_vocabulary"]
        == 64 * 64 * runtime["promotion_classes"],
        "maximum_legal_moves": runtime["maximum_legal_moves"] == 218,
        "oracle_loss_normalized": close_to_one(
            training["oracle"]["loss_weights"].values()
        ),
        "distillation_loss_normalized": close_to_one(
            training["student_distillation"]["loss_weights"].values()
        ),
        "safety_fail_closed": safety["fail_closed"] is True,
        "runtime_capability_schema": capabilities["architecture"]
        == "Unarchitectured v1",
        "scalar_forward_available": capabilities["scalar_neural_forward"] is True,
        "reference_vectors_available": capabilities["exported_reference_vectors"]
        is True,
        "quantized_forward_still_blocked": capabilities["quantized_neural_forward"]
        is False,
    }
    for architecture_key, student_key in mapping.items():
        checks[f"student_{architecture_key}"] = (
            runtime[architecture_key] == student[student_key]
        )
    profile_counts = []
    for profile in profiles["profiles"]:
        resolved = json.loads(json.dumps(training))
        verda_gpu_profile.deep_update(resolved["hardware"], profile.get("hardware", {}))
        verda_gpu_profile.deep_update(resolved["oracle"], profile.get("oracle", {}))
        oracle = resolved["oracle"]
        profile_counts.append(verda_gpu_profile.oracle_parameter_count(oracle))
        checks[f"profile_{profile['id']}_board_heads"] = (
            oracle["d_model"] % oracle["board_heads"] == 0
        )
        checks[f"profile_{profile['id']}_decoder_heads"] = (
            oracle["d_model"] % oracle["decoder_heads"] == 0
        )
        checks[f"profile_{profile['id']}_minimum_data"] = (
            oracle["minimum_optimizer_steps_per_epoch"] > 0
            and oracle["minimum_validation_records"] > 0
        )
    checks["profile_capacity_order"] = profile_counts == sorted(
        profile_counts, reverse=True
    )
    return {
        "schema": 1,
        "architecture": "Unarchitectured v1",
        "passed": all(checks.values()),
        "checks": checks,
        "oracle_parameters_by_profile": profile_counts,
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
