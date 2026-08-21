#!/usr/bin/env python3
"""Inspect and integrity-check a binary UNARCHV1 package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from unarchitectured_v1_package import DTYPE_NAMES, FLAG_METADATA, read_package


def inspect(path):
    path = Path(path)
    package = read_package(path)
    tensors = [section for section in package.sections if not section.flags & FLAG_METADATA]
    metadata = package.metadata
    checks = {
        "architecture": metadata.get("architecture") == "Unarchitectured v1",
        "format": metadata.get("format") == "UNARCHV1",
        "calibration_present": bool(metadata.get("calibration")),
        "tensor_count": metadata.get("tensor_count") == len(tensors),
        "unique_sections": len({section.name for section in package.sections})
        == len(package.sections),
    }
    return {
        "schema": 1,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_uuid": package.model_uuid.hex(),
        "metadata": metadata,
        "checks": checks,
        "passed": all(checks.values()),
        "sections": [
            {
                "name": section.name,
                "dtype": DTYPE_NAMES[section.dtype],
                "shape": section.shape,
                "bytes": len(section.data),
                "scale": section.scale,
                "zero_point": section.zero_point,
                "flags": section.flags,
            }
            for section in package.sections
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = inspect(args.package)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not report["passed"]:
        raise SystemExit("UNARCHV1 package inspection failed")


if __name__ == "__main__":
    main()
