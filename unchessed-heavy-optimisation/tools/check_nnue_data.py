#!/usr/bin/env python3
"""Sanity-check NNUE training sample files. Usage: check_nnue_data.py <bin...>"""
import sys

import numpy as np

REC = np.dtype([("bb", "<u8", 12), ("score", "<i2"), ("wdl", "u1"), ("pad", "u1", 5)])

total = 0
for path in sys.argv[1:]:
    d = np.fromfile(path, dtype=REC)
    total += len(d)
    if len(d) == 0:
        print(f"{path}: EMPTY")
        continue
    kings_ok = (
        (np.bitwise_count(d["bb"][:, 5]) == 1) & (np.bitwise_count(d["bb"][:, 11]) == 1)
    ).mean()
    overlap = 0
    occ_all = np.zeros(len(d), dtype=np.uint64)
    for p in range(12):
        overlap += int((occ_all & d["bb"][:, p]).any())
        occ_all |= d["bb"][:, p]
    s = d["score"].astype(np.int64)
    wdl = np.bincount(d["wdl"], minlength=3)
    print(
        f"{path}: n={len(d)} kings_ok={kings_ok:.4f} plane_overlap={overlap} "
        f"score(min={s.min()} max={s.max()} mean={s.mean():.1f} absmean={np.abs(s).mean():.0f}) "
        f"wdl L/D/W={wdl[0]}/{wdl[1]}/{wdl[2]} pad_zero={bool((d['pad'] == 0).all())}"
    )
print(f"TOTAL {total}")
