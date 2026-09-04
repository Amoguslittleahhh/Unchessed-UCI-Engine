#!/usr/bin/env python3
"""Train the Maia-style per-rating human policy nets (v2).

Input: bucketN.bin files from unchessed-datagen, record v2 (104 bytes):
  12 x u64 LE bitboards (side-to-move normalized) + u16 move (from | to<<6)
  + u16 rating + u8 castling rights + u8 ep file (0xFF none) + u8 flags
  (1 castle, 2 en passant, 4 promotion) + u8 pad.

Features (780): 768 piece-plane bits + 4 castling-rights bits + 8 ep-file
one-hots. Model per bucket: 780 -> hidden ReLU -> 4096 from-to logits.

Reports validation top-1 overall AND on the special-rule move classes
(castling / en passant / promotions) so rule handling is measurable.

Usage: train_policy.py <samples_dir> <out.bin> [epochs] [hidden] [max_per_bucket]
"""
import sys
import struct
import time

import numpy as np
import torch
import torch.nn as nn

BUCKETS = [(0, 1299), (1300, 1599), (1600, 1899), (1900, 4000)]
REC = np.dtype(
    [
        ("bb", "<u8", 12),
        ("mv", "<u2"),
        ("rating", "<u2"),
        ("castle", "u1"),
        ("ep", "u1"),
        ("flags", "u1"),
        ("pad", "u1"),
    ]
)
INPUT = 780


def load_bucket(path, max_n):
    data = np.fromfile(path, dtype=REC)
    if len(data) > max_n:
        rng = np.random.default_rng(2024)
        data = data[rng.choice(len(data), max_n, replace=False)]
    return data


def make_features(sel):
    n = len(sel)
    bits = np.unpackbits(
        sel["bb"].view(np.uint8).reshape(n, 96), axis=1, bitorder="little"
    ).astype(np.float32)
    castle = np.unpackbits(
        sel["castle"].reshape(n, 1), axis=1, bitorder="little"
    )[:, :4].astype(np.float32)
    ep = np.zeros((n, 8), dtype=np.float32)
    has_ep = sel["ep"] != 0xFF
    ep[np.arange(n)[has_ep], sel["ep"][has_ep]] = 1.0
    return np.concatenate([bits, castle, ep], axis=1)


def batches(data, idx, bs):
    for s in range(0, len(idx), bs):
        sel = data[idx[s : s + bs]]
        x = torch.from_numpy(make_features(sel))
        y = torch.from_numpy(sel["mv"].astype(np.int64))
        yield x, y


def evaluate(model, data, idx, bs=4096):
    """(overall, castle, ep, promo) top-1 accuracies on idx."""
    hits = np.zeros(4)
    tots = np.zeros(4)
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            sel = data[idx[s : s + bs]]
            x = torch.from_numpy(make_features(sel))
            y = sel["mv"].astype(np.int64)
            pred = model(x).argmax(dim=1).numpy()
            ok = pred == y
            f = sel["flags"]
            for k, mask in enumerate(
                [np.ones(len(sel), bool), f & 1 > 0, f & 2 > 0, f & 4 > 0]
            ):
                hits[k] += ok[mask].sum()
                tots[k] += mask.sum()
    return [h / t if t > 0 else float("nan") for h, t in zip(hits, tots)]


def train_bucket(path, epochs, hidden, max_n):
    data = load_bucket(path, max_n)
    n = len(data)
    n_val = max(1000, min(150_000, n // 40))
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = nn.Sequential(nn.Linear(INPUT, hidden), nn.ReLU(), nn.Linear(hidden, 4096))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    for ep in range(epochs):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.4 ** ep)
        rng.shuffle(train_idx)
        running = steps = 0
        for x, y in batches(data, train_idx, 1024):
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        acc = evaluate(model, data, val_idx)
        print(
            f"  epoch {ep + 1}: loss {running / max(steps,1):.3f} "
            f"val-top1 {acc[0]*100:.1f}% [castle {acc[1]*100:.1f}% "
            f"ep {acc[2]*100:.1f}% promo {acc[3]*100:.1f}%] "
            f"({time.time() - t0:.0f}s, {len(train_idx)} samples)",
            flush=True,
        )
    return model, evaluate(model, data, val_idx)


def main():
    samples_dir, out_path = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 320
    max_n = int(sys.argv[5]) if len(sys.argv) > 5 else 5_000_000

    torch.manual_seed(42)
    models = []
    for i, (lo, hi) in enumerate(BUCKETS):
        print(f"bucket {i} ({lo}-{hi}):", flush=True)
        model, acc = train_bucket(f"{samples_dir}/bucket{i}.bin", epochs, hidden, max_n)
        models.append((lo, hi, model, acc))

    with open(out_path, "wb") as f:
        f.write(b"UNCHMAIA")
        f.write(struct.pack("<IIIII", 2, len(models), INPUT, hidden, 4096))
        for lo, hi, model, _ in models:
            f.write(struct.pack("<II", lo, hi))
            lin1, lin2 = model[0], model[2]
            f.write(lin1.weight.detach().numpy().astype("<f4").tobytes())  # [hidden][INPUT]
            f.write(lin1.bias.detach().numpy().astype("<f4").tobytes())
            f.write(lin2.weight.detach().numpy().astype("<f4").tobytes())  # [4096][hidden]
            f.write(lin2.bias.detach().numpy().astype("<f4").tobytes())
    print(f"wrote {out_path}")
    for (lo, hi, _, acc) in models:
        print(
            f"  bucket {lo}-{hi}: top1 {acc[0]*100:.1f}% castle {acc[1]*100:.1f}% "
            f"ep {acc[2]*100:.1f}% promo {acc[3]*100:.1f}%"
        )


if __name__ == "__main__":
    main()
