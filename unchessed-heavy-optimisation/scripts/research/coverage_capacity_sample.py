"""Build two matched, equal-size NNUE training sets from a shard pool:
RAW (uniform random sample) and BALANCED (stratified evenly across the
8 piece-count buckets used by the shipped v4 output head), for the
coverage-vs-capacity diagnostic in docs/reinforcement/13-nnue-ceiling.md.

Usage: coverage_capacity_sample.py <pool_dir_with_w*.bin> <n_total> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np

REC = np.dtype([("bb", "<u8", 12), ("score", "<i2"), ("wdl", "u1"), ("pad", "u1", 5)])


def load_pool(pool_dir: Path) -> np.ndarray:
    parts = []
    for f in sorted(pool_dir.glob("w*.bin")):
        parts.append(np.fromfile(f, dtype=REC))
    return np.concatenate(parts)


def piece_count_bucket(bb: np.ndarray) -> np.ndarray:
    # bb: [n, 12] uint64. Popcount via numpy's bit_count (numpy>=2.0) with a
    # manual fallback for older numpy.
    if hasattr(np, "bit_count"):
        counts = np.bit_count(bb).sum(axis=1)
    else:
        counts = np.zeros(bb.shape[0], dtype=np.int64)
        for col in range(bb.shape[1]):
            v = bb[:, col].copy()
            c = np.zeros(bb.shape[0], dtype=np.int64)
            while v.any():
                c += (v & 1).astype(np.int64)
                v >>= np.uint64(1)
            counts += c
    bucket = np.clip((counts - 1) // 4, 0, 7)
    return bucket


def main():
    pool_dir = Path(sys.argv[1])
    n_total = int(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pool(pool_dir)
    n = len(data)
    buckets = piece_count_bucket(data["bb"])
    print(f"pool: {n} records")
    for b in range(8):
        print(f"  bucket {b}: {int((buckets == b).sum())} records")

    rng = np.random.default_rng(20260904)

    # RAW: uniform random sample, no stratification.
    raw_idx = rng.choice(n, size=min(n_total, n), replace=False)
    raw = data[raw_idx]
    raw.tofile(out_dir / "raw.bin")
    print(f"RAW: wrote {len(raw)} records -> {out_dir / 'raw.bin'}")

    # BALANCED: equal draw from each of the 8 buckets, oversampling with
    # replacement only for buckets smaller than the equal share.
    per_bucket = n_total // 8
    balanced_parts = []
    for b in range(8):
        bucket_idx = np.flatnonzero(buckets == b)
        if len(bucket_idx) == 0:
            print(f"  WARNING: bucket {b} is empty, skipping (balanced set will be smaller)")
            continue
        replace = len(bucket_idx) < per_bucket
        chosen = rng.choice(bucket_idx, size=per_bucket, replace=replace)
        balanced_parts.append(data[chosen])
        if replace:
            print(f"  bucket {b}: only {len(bucket_idx)} available, sampled {per_bucket} WITH replacement")
    balanced = np.concatenate(balanced_parts)
    rng.shuffle(balanced)
    balanced.tofile(out_dir / "balanced.bin")
    print(f"BALANCED: wrote {len(balanced)} records -> {out_dir / 'balanced.bin'}")


if __name__ == "__main__":
    main()
