#!/usr/bin/env python3
"""Measure tensor reconstruction drift for a UNARCHV1 package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from unarchitectured_v1_package import DTYPE_F32, DTYPE_I8, FLAG_METADATA, read_package


def validate(checkpoint_path, package_path, maximum_rmse_ratio=0.02):
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError("drift validation requires PyTorch and NumPy") from error
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    package = read_package(package_path)
    sections = {
        section.name: section
        for section in package.sections
        if not section.flags & FLAG_METADATA
    }
    missing = sorted(set(state) - set(sections))
    extra = sorted(set(sections) - set(state))
    tensors = []
    failed = []
    for name in sorted(set(state) & set(sections)):
        reference = state[name].detach().cpu().float().contiguous().numpy()
        section = sections[name]
        if section.dtype == DTYPE_I8:
            restored = np.frombuffer(section.data, dtype=np.int8).astype(np.float32)
            restored = restored.reshape(section.shape) * section.scale
        elif section.dtype == DTYPE_F32:
            restored = np.frombuffer(section.data, dtype="<f4").reshape(section.shape)
        else:
            failed.append(name)
            continue
        difference = restored - reference
        rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
        reference_rms = float(np.sqrt(np.mean(reference * reference))) if reference.size else 0.0
        ratio = rmse / max(reference_rms, 1e-12)
        maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
        passed = math.isfinite(ratio) and ratio <= maximum_rmse_ratio
        if not passed:
            failed.append(name)
        tensors.append(
            {
                "name": name,
                "dtype": "int8" if section.dtype == DTYPE_I8 else "float32",
                "rmse": rmse,
                "reference_rms": reference_rms,
                "rmse_ratio": ratio,
                "maximum_absolute_error": maximum,
                "passed": passed,
            }
        )
    return {
        "schema": 1,
        "checkpoint": str(checkpoint_path),
        "package": str(package_path),
        "maximum_rmse_ratio": maximum_rmse_ratio,
        "missing_tensors": missing,
        "extra_tensors": extra,
        "failed_tensors": failed,
        "passed": not missing and not extra and not failed,
        "tensors": tensors,
        "warning": "tensor reconstruction gate only; end-to-end neural forward drift is not yet implemented",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("--maximum-rmse-ratio", type=float, default=0.02)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.checkpoint, args.package, args.maximum_rmse_ratio
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not report["passed"]:
        raise SystemExit("UNARCHV1 tensor quantization drift exceeded threshold")


if __name__ == "__main__":
    main()
