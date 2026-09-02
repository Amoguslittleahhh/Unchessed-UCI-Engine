#!/usr/bin/env python3
"""Overwrite NNUE shard search-score labels without changing boards.

The 108M SPRT and the Bayes-floor analysis both say more of the same
5000-node HCE labels will not move val-MAE. The next experiment is
*stronger labels on the same positions*: a sidecar of i16 scores (one
per record, STM cp), produced by a deeper HCE search or by
self-distillation from a shipped net. This tool:

  * reads 104-byte shards (same layout as train_nnue.py)
  * writes a new shard with boards/WDL/pad copied and `score` replaced
  * reports MAE / Pearson between old and new scores so label-noise can
    be separated from architecture capacity

It does not run search or inference. Feeding it the original scores is
a no-op (MAE 0). It refuses mismatched lengths and nonzero pad.

Usage:
  nnue_relabel_existing.py --help
  nnue_relabel_existing.py compare <old.bin> <new_scores.i16>
  nnue_relabel_existing.py apply   <old.bin> <new_scores.i16> <out.bin>
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

REC_SIZE = 104
BB_BYTES = 12 * 8
SCORE_OFF = BB_BYTES  # 96
WDL_OFF = 98
PAD_OFF = 99


def iter_records(path: str):
    size = os.path.getsize(path)
    if size % REC_SIZE != 0:
        raise SystemExit(
            f"ERROR: {path} size {size} is not a multiple of {REC_SIZE}"
        )
    if size == 0:
        raise SystemExit(f"ERROR: {path} is empty")
    with open(path, "rb") as f:
        n = 0
        while True:
            blob = f.read(REC_SIZE)
            if not blob:
                break
            score = struct.unpack_from("<h", blob, SCORE_OFF)[0]
            wdl = blob[WDL_OFF]
            pad = blob[PAD_OFF:]
            if wdl > 2 or pad != b"\x00" * 5:
                raise SystemExit(
                    f"ERROR: {path} does not look like an NNUE sample file "
                    "(wdl>2 or nonzero padding)"
                )
            n += 1
            yield blob, score
    if n == 0:
        raise SystemExit(f"ERROR: {path} is empty")


def load_scores(path: str) -> list[int]:
    size = os.path.getsize(path)
    if size % 2 != 0:
        raise SystemExit(f"ERROR: {path} is not packed i16 (odd size {size})")
    with open(path, "rb") as f:
        raw = f.read()
    n = size // 2
    return list(struct.unpack("<" + "h" * n, raw))


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    num = sum(x * y for x, y in zip(da, db))
    denom = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    if denom == 0.0:
        return float("nan")
    return num / denom


def report(old: list[int], new: list[int]) -> dict:
    n = len(old)
    diffs = [float(n_) - float(o) for o, n_ in zip(old, new)]
    mae = sum(abs(d) for d in diffs) / n
    rms = math.sqrt(sum(d * d for d in diffs) / n)
    changed = sum(1 for o, n_ in zip(old, new) if o != n_)
    return {
        "n": n,
        "mae_cp": mae,
        "rms_cp": rms,
        "changed": changed,
        "frac_changed": changed / n,
        "pearson": pearson([float(x) for x in old], [float(x) for x in new]),
        "old_mean": sum(old) / n,
        "new_mean": sum(new) / n,
    }


def read_old_scores(shard: str) -> tuple[list[bytes], list[int]]:
    blobs: list[bytes] = []
    scores: list[int] = []
    for blob, score in iter_records(shard):
        blobs.append(blob)
        scores.append(score)
    return blobs, scores


def cmd_compare(shard: str, scores_path: str) -> int:
    _blobs, old = read_old_scores(shard)
    new = load_scores(scores_path)
    if len(new) != len(old):
        raise SystemExit(
            f"ERROR: {scores_path} has {len(new)} scores, shard has {len(old)} records"
        )
    stats = report(old, new)
    print(
        "n={n} mae={mae_cp:.2f}cp rms={rms_cp:.2f}cp "
        "changed={changed}/{n} ({frac_changed:.4f}) pearson={pearson:.4f} "
        "old_mean={old_mean:.1f} new_mean={new_mean:.1f}".format(**stats)
    )
    return 0


def cmd_apply(shard: str, scores_path: str, out_path: str) -> int:
    blobs, old = read_old_scores(shard)
    new = load_scores(scores_path)
    if len(new) != len(old):
        raise SystemExit(
            f"ERROR: {scores_path} has {len(new)} scores, shard has {len(old)} records"
        )
    stats = report(old, new)
    with open(out_path, "wb") as f:
        for blob, score in zip(blobs, new):
            rec = bytearray(blob)
            struct.pack_into("<h", rec, SCORE_OFF, int(score))
            f.write(rec)
    print(
        "wrote {out} ({bytes} bytes) n={n} mae={mae_cp:.2f}cp "
        "pearson={pearson:.4f}".format(
            out=out_path, bytes=os.path.getsize(out_path), **stats
        )
    )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="print old-vs-new score stats, no write")
    c.add_argument("shard")
    c.add_argument("scores")
    a = sub.add_parser("apply", help="write a new shard with replaced scores")
    a.add_argument("shard")
    a.add_argument("scores")
    a.add_argument("out")
    args = p.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args.shard, args.scores)
    return cmd_apply(args.shard, args.scores, args.out)


if __name__ == "__main__":
    sys.exit(main())
