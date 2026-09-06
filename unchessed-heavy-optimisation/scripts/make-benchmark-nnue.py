#!/usr/bin/env python3
"""Create a deterministic small-scale-valid v1 UNCHNNUE file for throughput tests."""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

FT_IN = 768
ACC = 256


def values(count: int, scale: float, phase: float):
    for i in range(count):
        yield scale * math.sin((i + 1) * 0.0017 + phase)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make-benchmark-nnue.py OUTPUT")
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    ft_w = list(values(FT_IN * ACC, 0.002, 0.0))
    ft_b = list(values(ACC, 0.01, 0.3))
    out_w = list(values(2 * ACC, 0.01, 0.7))
    out_b = [0.0]
    payload = b"".join(
        (
            b"UNCHNNUE",
            struct.pack("<III", 1, FT_IN, ACC),
            struct.pack(f"<{len(ft_w)}f", *ft_w),
            struct.pack(f"<{len(ft_b)}f", *ft_b),
            struct.pack(f"<{len(out_w)}f", *out_w),
            struct.pack("<f", *out_b),
        )
    )
    output.write_bytes(payload)
    print(f"{output} {len(payload)} bytes")


if __name__ == "__main__":
    main()
