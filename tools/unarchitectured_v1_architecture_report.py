#!/usr/bin/env python3
"""Calculate the canonical Unarchitectured v1 architecture budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import hydra_v5_architecture_report as experimental


def build_report(config, student, profiles, training):
    if config.get("schema") != 1 or config.get("name") != "Unarchitectured v1":
        raise ValueError("canonical Unarchitectured schema/name mismatch")
    if config.get("runtime_file_magic") != "UNARCHV1":
        raise ValueError("canonical Unarchitectured v1 runtime magic mismatch")
    canonical_student = {"chessformer": config["runtime_student"]}
    if student is not None:
        for key in ("d_model", "layers", "exit_layers", "matryoshka_widths"):
            if student["chessformer"].get(key) != canonical_student["chessformer"].get(key):
                raise ValueError(f"external student config drifts from canonical key {key}")
    legacy = dict(config)
    legacy["schema"] = 5
    report = experimental.build_report(legacy, canonical_student, profiles, training)
    report["schema"] = 1
    report["name"] = config["name"]
    report["runtime_file_magic"] = config["runtime_file_magic"]
    report["lineage"] = config["lineage"]
    report["autonomous_safety"] = config["autonomous_safety"]
    report["training_efficiency"] = config["training_efficiency"]
    report["feature_extraction"] = config["feature_extraction"]
    report["experimental_predecessor"] = "Hydra Apex v5"
    return report


def markdown(report):
    legacy = experimental.markdown(report)
    legacy = legacy.replace(
        "# Unchessed Hydra Apex v5 calculated budget",
        "# Unarchitectured v1 calculated budget",
        1,
    )
    return legacy + (
        "\nHydra v1-v5 and Apex v1 are experimental lineage labels. "
        "Unarchitectured v1 is canonical, but remains untrained and default-off until all "
        "export, runtime, calibration, and SPRT gates pass.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unarchitectured_v1.json")
    parser.add_argument("--student-config", default="config/unarchitectured_v1_student.json")
    parser.add_argument("--profiles", default="config/verda_gpu_profiles.json")
    parser.add_argument("--training-config", default="config/unarchitectured_v1_training.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    student = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    training = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    report = build_report(config, student, profiles, training)
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown(report)
    if args.check:
        if args.json is None or args.markdown is None:
            raise SystemExit("--check requires output paths")
        if args.json.read_text(encoding="utf-8") != json_text:
            raise SystemExit(f"generated output differs: {args.json}")
        if args.markdown.read_text(encoding="utf-8") != markdown_text:
            raise SystemExit(f"generated output differs: {args.markdown}")
    else:
        print(json_text, end="")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json_text, encoding="utf-8")
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown_text, encoding="utf-8")


if __name__ == "__main__":
    main()
