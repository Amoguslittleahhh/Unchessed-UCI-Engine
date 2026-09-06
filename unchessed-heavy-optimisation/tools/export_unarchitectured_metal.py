#!/usr/bin/env python3
"""Export a calibrated Unarchitectured Metal student checkpoint to UNARCHV1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from unarchitectured_metal_package import (
    DTYPE_F32,
    DTYPE_I8,
    FLAG_QUANTIZED,
    Section,
    atomic_write,
    build_package,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(checkpoint_path, architecture_path, allow_legacy=False):
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError("export requires PyTorch and NumPy on the training host") from error
    checkpoint_path = Path(checkpoint_path)
    architecture_path = Path(architecture_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_format = str(checkpoint.get("format", ""))
    if not checkpoint_format.startswith("UNARCHV1_") and not allow_legacy:
        raise ValueError(
            f"checkpoint format {checkpoint_format!r} is not canonical Unarchitectured Metal"
        )
    if "calibration" not in checkpoint:
        raise ValueError("student checkpoint is not calibrated")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no model state dictionary")
    sections = []
    quantized = float32 = 0
    for name in sorted(state):
        tensor = state[name]
        if not hasattr(tensor, "detach"):
            raise TypeError(f"state entry {name!r} is not a tensor")
        array = tensor.detach().cpu().float().contiguous().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"state tensor {name!r} contains non-finite values")
        use_int8 = array.ndim >= 2 and array.size >= 256
        if use_int8:
            maximum = float(np.max(np.abs(array)))
            scale = maximum / 127.0 if maximum else 1.0
            encoded = np.rint(array / scale).clip(-127, 127).astype(np.int8)
            sections.append(
                Section(
                    name=name,
                    dtype=DTYPE_I8,
                    shape=tuple(array.shape),
                    data=encoded.tobytes(order="C"),
                    scale=scale,
                    flags=FLAG_QUANTIZED,
                )
            )
            quantized += 1
        else:
            encoded = np.asarray(array, dtype="<f4")
            sections.append(
                Section(
                    name=name,
                    dtype=DTYPE_F32,
                    shape=tuple(array.shape),
                    data=encoded.tobytes(order="C"),
                )
            )
            float32 += 1
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if architecture.get("runtime_file_magic") != "UNARCHV1":
        raise ValueError("architecture config is not canonical UNARCHV1")
    metadata = {
        "architecture": "Unarchitectured Metal",
        "format": "UNARCHV1",
        "checkpoint_format": checkpoint_format,
        "checkpoint_sha256": sha256(checkpoint_path),
        "architecture_sha256": sha256(architecture_path),
        "calibration": checkpoint["calibration"],
        "student_config": checkpoint.get("student_config"),
        "tensor_count": len(sections),
        "quantized_tensor_count": quantized,
        "float32_tensor_count": float32,
        "warning": "generic tensor package; scalar neural forward remains a separate gate",
    }
    return build_package(sections, metadata), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--architecture", default="config/unarchitectured_metal.json", type=Path
    )
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    blob, metadata = export_checkpoint(
        args.checkpoint, args.architecture, args.allow_legacy
    )
    atomic_write(args.output, blob)
    report = {
        **metadata,
        "output": str(args.output),
        "output_bytes": len(blob),
        "output_sha256": hashlib.sha256(blob).hexdigest(),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
