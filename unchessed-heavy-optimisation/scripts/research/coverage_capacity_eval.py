"""Per-piece-count-bucket held-out MAE for a trained NNUE net, for the
coverage-vs-capacity diagnostic (docs/reinforcement/13-nnue-ceiling.md).
Reuses tools/train_nnue.py's own validated, export-consistent manual
feature functions (the ones its own selfcheck cross-checks the fast
path against) rather than re-deriving the index scheme.

Usage: coverage_capacity_eval.py <net.bin> <heldout.bin>
"""
import struct
import sys

import numpy as np

sys.path.insert(0, "tools")
import train_nnue as tn  # noqa: E402


def load_net(path):
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == b"UNCHNNUE"
    version, ft_in, acc = struct.unpack_from("<III", blob, 8)
    assert version == 4, version
    off = 20
    ftw = np.frombuffer(blob, "<f4", ft_in * acc, off).reshape(ft_in, acc)
    off += ft_in * acc * 4
    ftb = np.frombuffer(blob, "<f4", acc, off)
    off += acc * 4
    outw = np.frombuffer(blob, "<f4", 8 * 2 * acc, off).reshape(8, 2 * acc)
    off += 8 * 2 * acc * 4
    outb = np.frombuffer(blob, "<f4", 8, off)
    return ftw, ftb, outw, outb


def screlu(x):
    v = np.clip(x, 0, 1)
    return v * v


def eval_one(ftw, ftb, outw, outb, bb_row):
    bits = tn._manual_bits(bb_row)
    nstm_bits = tn._manual_nstm_bits(bits)
    stm_feats = tn._manual_halfka_v2_hm(bits)
    nstm_feats = tn._manual_halfka_v2_hm(nstm_bits)
    a = ftw[stm_feats].sum(axis=0) + ftb
    b = ftw[nstm_feats].sum(axis=0) + ftb
    h = np.concatenate([screlu(a), screlu(b)])
    pieces = int(bits.sum())
    bucket = min((pieces - 1) // 4, 7)
    raw = float(h @ outw[bucket] + outb[bucket])
    return raw * 400.0, bucket


def main():
    net_path, heldout_path = sys.argv[1], sys.argv[2]
    ftw, ftb, outw, outb = load_net(net_path)
    data = np.fromfile(heldout_path, dtype=tn.REC)
    n = len(data)

    abs_err = np.zeros(n)
    buckets = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pred_cp, bucket = eval_one(ftw, ftb, outw, outb, data["bb"][i])
        abs_err[i] = abs(pred_cp - float(data["score"][i]))
        buckets[i] = bucket

    print(f"n={n} overall_mae={abs_err.mean():.2f}cp")
    for bkt in range(8):
        mask = buckets == bkt
        cnt = int(mask.sum())
        if cnt == 0:
            print(f"  bucket {bkt}: 0 records")
            continue
        print(f"  bucket {bkt}: n={cnt} mae={abs_err[mask].mean():.2f}cp")


if __name__ == "__main__":
    main()
