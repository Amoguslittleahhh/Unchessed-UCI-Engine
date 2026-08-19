#!/usr/bin/env python3
"""Inspect and validate an UNCHNNUE weight file without loading the engine."""

import argparse
import json
import math
import mmap
import os
import struct

SCHEMES = {
    1: {"name": "Flat768", "inputs": 768, "accumulator": 256, "head_multiplier": 2},
    2: {"name": "HalfKA", "inputs": 45056, "accumulator": 256, "head_multiplier": 4},
    3: {"name": "HalfKAv2_hm", "inputs": 22528, "accumulator": 256, "head_multiplier": 2},
}


def inspect(path):
    size = os.path.getsize(path)
    with open(path, "rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if size < 20 or data[:8] != b"UNCHNNUE":
            raise ValueError("not an UNCHNNUE file")
        version, inputs, accumulator = struct.unpack_from("<III", data, 8)
        if version not in SCHEMES:
            raise ValueError(f"unsupported version {version}")
        scheme = SCHEMES[version]
        if inputs != scheme["inputs"] or accumulator != scheme["accumulator"]:
            raise ValueError(
                f"dimensions {inputs}x{accumulator} do not match {scheme['name']} "
                f"({scheme['inputs']}x{scheme['accumulator']})"
            )
        parameters = (
            inputs * accumulator
            + accumulator
            + scheme["head_multiplier"] * accumulator
            + 1
        )
        expected_size = 20 + 4 * parameters
        if size != expected_size:
            raise ValueError(f"size {size} does not match expected {expected_size}")

        non_finite = 0
        maximum = 0.0
        transformer_values = inputs * accumulator
        nonzero_rows = [False] * inputs
        for index, (value,) in enumerate(struct.iter_unpack("<f", data[20:])):
            if not math.isfinite(value):
                non_finite += 1
            else:
                maximum = max(maximum, abs(value))
                if index < transformer_values and value != 0.0:
                    nonzero_rows[index // accumulator] = True
        if non_finite:
            raise ValueError(f"file contains {non_finite} non-finite values")
        all_zero_rows = nonzero_rows.count(False)

        return {
            "path": path,
            "size_bytes": size,
            "version": version,
            "scheme": scheme["name"],
            "inputs": inputs,
            "accumulator": accumulator,
            "parameters": parameters,
            "expected_size_bytes": expected_size,
            "max_abs_value": maximum,
            "all_zero_feature_rows": all_zero_rows,
            "quantized_runtime_bytes": inputs * accumulator * 2 + accumulator * 2,
            "validation": "PASS",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    args = parser.parse_args()
    print(json.dumps(inspect(args.file), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
