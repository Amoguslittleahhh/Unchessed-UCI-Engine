#!/usr/bin/env python3
"""Calculate the canonical Unchessed Apex v1 architecture budget."""

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
    if config.get("schema") != 1 or config.get("name") != "Unchessed Apex v1":
        raise ValueError("canonical Apex schema/name mismatch")
    if config.get("runtime_file_magic") != "UNCHAPX1":
        raise ValueError("canonical Apex v1 runtime magic mismatch")
    legacy = dict(config)
    legacy["schema"] = 5
    report = experimental.build_report(legacy, student, profiles, training)
    report["schema"] = 1
    report["name"] = config["name"]
    report["runtime_file_magic"] = config["runtime_file_magic"]
    report["lineage"] = config["lineage"]
    report["experimental_predecessor"] = "Hydra Apex v5"
    return report


def markdown(report):
    legacy = experimental.markdown(report)
    legacy = legacy.replace(
        "# Unchessed Hydra Apex v5 calculated budget",
        "# Unchessed Apex v1 calculated budget",
        1,
    )
    return legacy + (
        "\nThe Hydra v1-v5 names are experimental lineage labels. Apex v1 is the "
        "canonical architecture name, but remains untrained and default-off until all "
        "export, runtime, calibration, and SPRT gates pass.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unchessed_apex_v1.json")
    parser.add_argument("--student-config", default="config/unchessed_hydra_v4.json")
    parser.add_argument("--profiles", default="config/verda_gpu_profiles.json")
    parser.add_argument("--training-config", default="config/apex_v1_training.json")
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
