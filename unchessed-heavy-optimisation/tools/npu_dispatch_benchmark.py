#!/usr/bin/env python3
"""Measure real Intel AI Boost NPU dispatch/inference latency via OpenVINO.

docs/npu-viability-285h.md's NPU dispatch cost (~1ms) was a literature
estimate because the sandbox that wrote it had no NPU. This machine has a
real Core Ultra 9 285H with a working NPU, so this script replaces that
estimate with a measurement.

Two model sizes are benchmarked:

- "tiny": a single 256x1 linear layer, matching the NNUE output layer's
  shape. Isolates pure per-call dispatch overhead from compute time.
- "student": a stack of 256x256 linear+ReLU layers sized to roughly match
  Unarchitectured v1's ~4.2M runtime parameter count. Approximates a real
  per-move forward pass's dispatch-plus-compute cost, not just dispatch.

Both are run on CPU and NPU for comparison. Output is machine-readable JSON
so the numbers can be pasted into a doc without transcription.
"""

from __future__ import annotations

import argparse
import json
import time


def build_tiny_model(np, ov):
    import openvino.opset14 as ops

    x = ops.parameter([1, 256], np.float32, name="x")
    w = ops.constant(np.random.randn(256, 1).astype(np.float32) * 0.01)
    y = ops.matmul(x, w, False, False)
    return ov.Model([y], [x], "tiny_linear_256x1")


def build_student_model(np, ov, layers: int = 8, width: int = 256):
    import openvino.opset14 as ops

    x = ops.parameter([1, width], np.float32, name="x")
    h = x
    for i in range(layers):
        w = ops.constant((np.random.randn(width, width) * (1.0 / width**0.5)).astype(np.float32))
        h = ops.matmul(h, w, False, False)
        h = ops.relu(h)
    return ov.Model([h], [x], f"student_{layers}x{width}")


def bench(np, compiled, input_shape, calls: int, warmup: int):
    request = compiled.create_infer_request()
    x = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(warmup):
        request.infer({0: x})
    samples = []
    for _ in range(calls):
        started = time.perf_counter()
        request.infer({0: x})
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    n = len(samples)
    return {
        "calls": calls,
        "mean_ms": sum(samples) / n,
        "median_ms": samples[n // 2],
        "p10_ms": samples[n // 10],
        "p90_ms": samples[(n * 9) // 10],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--devices", default="CPU,NPU")
    args = parser.parse_args()

    # Deferred so --help works standalone even without openvino/numpy
    # installed, matching this repo's other canonical CLIs.
    import numpy as np
    import openvino as ov

    core = ov.Core()
    available = set(core.available_devices)
    devices = [d for d in args.devices.split(",") if d in available]
    missing = [d for d in args.devices.split(",") if d not in available]

    models = {
        "tiny_256x1": (build_tiny_model(np, ov), (1, 256)),
        "student_8x256": (build_student_model(np, ov, 8, 256), (1, 256)),
        # ~64 x 256x256 matmuls =~ 4.2M multiply-add params, roughly matching
        # Unarchitectured v1's runtime student parameter count, to see
        # whether NPU throughput offsets its fixed dispatch overhead at
        # closer-to-real compute scale (this is still a flat MLP stand-in,
        # not the real attention/GAB architecture -- real compute is
        # structured differently, so this bounds the question rather than
        # answering it exactly).
        "student_64x256": (build_student_model(np, ov, 64, 256), (1, 256)),
    }

    report = {
        "available_devices": sorted(available),
        "requested_missing": missing,
        "full_device_names": {
            d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices
        },
        "results": {},
    }

    for model_name, (model, shape) in models.items():
        report["results"][model_name] = {}
        for device in devices:
            compiled = core.compile_model(model, device)
            report["results"][model_name][device] = bench(
                np, compiled, shape, args.calls, args.warmup
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
