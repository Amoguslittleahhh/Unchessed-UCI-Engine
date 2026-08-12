#!/usr/bin/env python3
"""Train the first NNUE eval net for Unchessed AI (CPU, PyTorch).

Input: shard .bin files of 104-byte records, side-to-move normalized:
  12 x u64 LE bitboards (planes 0-5 = mover P,N,B,R,Q,K; 6-11 = opponent;
  board vertically flipped when black was to move, so bit i = square i with
  a1=0..h8=63 in the mover-up frame)
  + i16 LE search score in centipawns (stm pov)
  + u8 wdl (2 = mover won, 1 = draw, 0 = lost) + 5 pad bytes.

Features: STM perspective index = plane*64 + sq (768). NSTM perspective =
swap plane groups (own<->opp) AND flip squares vertically (sq^56), done in
numpy as u64.byteswap() + plane reorder [6..11, 0..5].

Model: shared EmbeddingBag(768 -> 256, sum) feature transformer + bias,
SCReLU (clamp(x,0,1)^2) on both accumulators, concat [stm, nstm] (512)
-> Linear(512, 1). Raw output unit is cp/400: the Rust engine computes
eval_cp = raw * 400. Training loss: |sigmoid(raw) - target|^2.5 (not plain
MSE), target = 0.7*sigmoid(cp/400) + 0.3*(wdl/2).

Export: b"UNCHNNUE", u32 version=1, u32 ft_in=768, u32 acc=256,
ft weights [768][256] f32, ft bias [256] f32, out weights [512] f32
(STM half first, matching the concat order), out bias [1] f32. All LE.

Usage: train_nnue.py selfcheck
       train_nnue.py <out.bin> <epochs> <shard1.bin> [shard2.bin ...]

Device: auto-detects CUDA; override with DEVICE=cpu|cuda|cuda:0 env var.
Batch size: override with BATCH_SIZE env var (default 16384; a GPU can
usually take this much higher, e.g. 65536+, for better throughput).
"""
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(
    os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16384))

REC = np.dtype(
    [
        ("bb", "<u8", 12),
        ("score", "<i2"),
        ("wdl", "u1"),
        ("pad", "u1", 5),
    ]
)
FT_IN = 768
ACC = 256
MAGIC = b"UNCHNNUE"
HEADER_SIZE = 8 + 3 * 4
PAYLOAD_FLOATS = FT_IN * ACC + ACC + 2 * ACC + 1
# NSTM perspective: use opponent planes as "own" and vice versa.
PLANE_SWAP = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])


def screlu(x):
    v = x.clamp(0.0, 1.0)
    return v * v


def wdl_loss(raw, target):
    # |sigmoid(raw) - target|^2.5, not plain MSE (exponent 2) -- matches
    # Stockfish's nnue-pytorch recipe, which weights positions near the
    # 50%-win boundary (the most decision-relevant ones) more heavily than
    # squared error alone does. Computed as diff^2 * sqrt(|diff|) rather
    # than diff ** 2.5 directly, since the latter is NaN for negative diff
    # under float exponentiation.
    #
    # The epsilon inside the sqrt is load-bearing, not cosmetic: sqrt(x)'s
    # gradient is 1/(2*sqrt(x)), which is infinite at x=0. Autograd doesn't
    # algebraically simplify diff^2 * sqrt(|diff|) before differentiating,
    # so even though the TRUE combined derivative is a well-defined 0 as
    # diff -> 0, the product rule evaluates as inf * 0 = NaN at exactly
    # diff == 0 -- and across a large batch, at least one sample landing
    # exactly (or numerically) at diff == 0 is close to inevitable. That
    # single NaN then poisons the whole batch's gradient via .mean()'s
    # backward, corrupting every weight after one optimizer step. Clamping
    # the sqrt's argument away from exactly 0 keeps the gradient finite
    # (bounded by 1/(2*sqrt(eps))) with a negligible accuracy cost, since
    # diff^2 is also -> 0 in that same regime.
    diff = torch.sigmoid(raw) - target
    return (diff * diff * (diff.abs() + 1e-8).sqrt()).mean()


class Nnue(nn.Module):
    def __init__(self):
        super().__init__()
        # include_last_offset=True: offsets has n+1 entries, last = total nnz.
        self.ft = nn.EmbeddingBag(FT_IN, ACC, mode="sum", include_last_offset=True)
        self.ft_bias = nn.Parameter(torch.zeros(ACC))
        self.out = nn.Linear(2 * ACC, 1, bias=True)
        # Default EmbeddingBag init is N(0,1); with ~30 active features the
        # sum would saturate SCReLU immediately. Small uniform init instead.
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)

    def forward(self, stm_idx, stm_off, nstm_idx, nstm_off):
        acc_stm = self.ft(stm_idx, stm_off) + self.ft_bias
        acc_nstm = self.ft(nstm_idx, nstm_off) + self.ft_bias
        h = torch.cat([screlu(acc_stm), screlu(acc_nstm)], dim=1)
        return self.out(h).squeeze(1)  # raw output ~ cp/400


def perspective_bits(sel):
    """(stm_bits, nstm_bits), each uint8 [n, 768], bit p*64+s set iff active."""
    n = len(sel)
    bb = np.ascontiguousarray(sel["bb"])
    stm = np.unpackbits(bb.view(np.uint8).reshape(n, 96), axis=1, bitorder="little")
    # byteswap of a u64 == vertical flip of a bitboard (sq -> sq^56)
    flipped = np.ascontiguousarray(bb.byteswap()[:, PLANE_SWAP])
    nstm = np.unpackbits(
        flipped.view(np.uint8).reshape(n, 96), axis=1, bitorder="little"
    )
    return stm, nstm


def flat_indices(bits):
    """Variable-length exact bags: flat feature indices + offsets [n+1]."""
    rows, cols = np.nonzero(bits)  # row-major -> cols grouped per sample
    offsets = np.zeros(len(bits) + 1, dtype=np.int64)
    np.cumsum(bits.sum(axis=1, dtype=np.int64), out=offsets[1:])
    return torch.from_numpy(cols.astype(np.int64)), torch.from_numpy(offsets)


def make_batch(sel):
    stm_bits, nstm_bits = perspective_bits(sel)
    stm_idx, stm_off = flat_indices(stm_bits)
    nstm_idx, nstm_off = flat_indices(nstm_bits)
    score = sel["score"].astype(np.float32)
    target = 0.7 / (1.0 + np.exp(-score / 400.0)) + 0.3 * (
        sel["wdl"].astype(np.float32) / 2.0
    )
    return (
        stm_idx.to(DEVICE, non_blocking=True),
        stm_off.to(DEVICE, non_blocking=True),
        nstm_idx.to(DEVICE, non_blocking=True),
        nstm_off.to(DEVICE, non_blocking=True),
        torch.from_numpy(target.astype(np.float32)).to(DEVICE, non_blocking=True),
        torch.from_numpy(score).to(DEVICE, non_blocking=True),
    )


def batches(data, idx, bs):
    for s in range(0, len(idx), bs):
        yield make_batch(data[idx[s : s + bs]])


def evaluate(model, data, idx, bs=BATCH_SIZE):
    """(val MSE loss, val MAE in centipawns)."""
    se = ae = n = 0.0
    with torch.no_grad():
        for si, so, ni, no, target, score in batches(data, idx, bs):
            raw = model(si, so, ni, no)
            se += ((torch.sigmoid(raw) - target) ** 2).sum().item()
            ae += (raw * 400.0 - score).abs().sum().item()
            n += len(target)
    return se / max(n, 1), ae / max(n, 1)


def export_net(model, path):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", 1, FT_IN, ACC))
        # EmbeddingBag.weight is [768][256] = one row per feature: write as-is.
        f.write(model.ft.weight.detach().cpu().numpy().astype("<f4").tobytes())
        f.write(model.ft_bias.detach().cpu().numpy().astype("<f4").tobytes())
        # out.weight is [1, 512]; [:256] = STM half, [256:] = NSTM half,
        # exactly the concat order in forward().
        f.write(
            model.out.weight.detach().cpu().numpy().astype("<f4").reshape(-1).tobytes()
        )
        f.write(model.out.bias.detach().cpu().numpy().astype("<f4").tobytes())


def train(shards, out_path, epochs):
    # Read shard sizes up front and allocate the full array once, then read
    # each shard directly into its slice -- avoids ever holding both the
    # per-shard arrays AND their concatenation in memory simultaneously
    # (np.concatenate on a list of already-loaded parts doubles peak RSS
    # for the whole run, since Python has no reason to free `parts` while
    # it's still a live local variable -- at 100M+ records that's an extra
    # ~11GB retained uselessly for the entire multi-hour training run).
    counts = []
    for p in shards:
        size = os.path.getsize(p)
        if size % REC.itemsize != 0:
            print(f"warning: {p} size {size} not a multiple of {REC.itemsize}, "
                  f"trailing bytes ignored", flush=True)
        counts.append(size // REC.itemsize)
    n = sum(counts)
    if n < 1000:
        raise SystemExit(f"ERROR: only {n} records total — refusing to train")
    data = np.empty(n, dtype=REC)
    offset = 0
    for p, cnt in zip(shards, counts):
        part = np.fromfile(p, dtype=REC, count=cnt)
        # guard against ingesting Maia-policy shards (same 104-byte size,
        # different fields): NNUE records have wdl<=2 and zero padding
        if len(part) and (part["wdl"].max() > 2 or (part["pad"] != 0).any()):
            raise SystemExit(f"ERROR: {p} does not look like an NNUE sample file "
                             f"(wdl>2 or nonzero padding) — wrong shard?")
        print(f"loaded {len(part)} records from {p}", flush=True)
        data[offset : offset + cnt] = part
        offset += cnt
    del part
    n_val = min(200_000, max(1, n // 50))  # 200k, or 2% if smaller
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"total {n} records: {len(train_idx)} train / {n_val} val", flush=True)
    print(f"device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else "")
          + f", batch size: {BATCH_SIZE}", flush=True)

    model = Nnue().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    drop1, drop2 = int(epochs * 0.6), int(epochs * 0.8)

    for ep in range(epochs):
        lr = 1e-3 * 0.3 ** ((ep >= drop1) + (ep >= drop2))
        for g in opt.param_groups:
            g["lr"] = lr
        train_idx = rng.permutation(train_idx)
        t0 = time.time()
        running = steps = 0
        for si, so, ni, no, target, _ in batches(data, train_idx, BATCH_SIZE):
            opt.zero_grad()
            loss = wdl_loss(model(si, so, ni, no), target)
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        t_train = time.time() - t0
        val_loss, val_mae = evaluate(model, data, val_idx)
        print(
            f"epoch {ep + 1}/{epochs}: lr {lr:.1e} "
            f"train-loss {running / max(steps, 1):.6f} "
            f"val-loss {val_loss:.6f} val-mae {val_mae:.1f}cp "
            f"({len(train_idx) / max(t_train, 1e-9):.0f} samples/s, "
            f"{t_train:.0f}s)",
            flush=True,
        )

    export_net(model, out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def synth_records(n, rng):
    """Random sparse boards, exactly one king per side (planes 5 and 11)."""
    recs = np.zeros(n, dtype=REC)
    for i in range(n):
        squares = rng.permutation(64)
        bits = np.zeros((12, 64), dtype=np.uint8)
        bits[5, squares[0]] = 1  # mover king
        bits[11, squares[1]] = 1  # opponent king
        for s in squares[2 : 2 + int(rng.integers(2, 29))]:
            p = int(rng.integers(0, 10))  # non-king planes 0-4 / 6-10
            bits[p if p < 5 else p + 1, s] = 1
        packed = np.packbits(bits, axis=1, bitorder="little")  # [12, 8]
        recs["bb"][i] = packed.reshape(12, 8).copy().view("<u8").ravel()
    recs["score"] = rng.integers(-1200, 1201, n).astype(np.int16)
    recs["wdl"] = rng.integers(0, 3, n).astype(np.uint8)
    return recs


def selfcheck():
    rng = np.random.default_rng(1337)
    torch.manual_seed(1337)
    data = synth_records(1000, rng)
    print(f"selfcheck device: {DEVICE}", flush=True)

    model = Nnue().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = np.arange(len(data))
    for _ in range(3):
        si, so, ni, no, target, _ = make_batch(data[idx])
        opt.zero_grad()
        loss = wdl_loss(model(si, so, ni, no), target)
        loss.backward()
        opt.step()
    print(f"selfcheck: 3 training steps done, last loss {loss.item():.6f}",
          flush=True)

    import tempfile

    fd, path = tempfile.mkstemp(suffix=".nnue.bin")
    os.close(fd)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}: {name}", flush=True)
        ok = ok and cond

    try:
        export_net(model, path)
        with open(path, "rb") as f:
            blob = f.read()
        expected = HEADER_SIZE + PAYLOAD_FLOATS * 4
        check(f"file size {len(blob)} == {expected}", len(blob) == expected)
        check("magic UNCHNNUE", blob[:8] == MAGIC)
        version, ft_in, acc = struct.unpack("<III", blob[8:HEADER_SIZE])
        check("version == 1", version == 1)
        check("ft_in == 768", ft_in == FT_IN)
        check("acc == 256", acc == ACC)

        off = HEADER_SIZE
        ftw = np.frombuffer(blob, "<f4", FT_IN * ACC, off).reshape(FT_IN, ACC)
        off += FT_IN * ACC * 4
        ftb = np.frombuffer(blob, "<f4", ACC, off)
        off += ACC * 4
        outw = np.frombuffer(blob, "<f4", 2 * ACC, off)
        off += 2 * ACC * 4
        outb = np.frombuffer(blob, "<f4", 1, off)

        # Manual numpy forward from the exported arrays vs model().
        max_diff = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(10):
                sel = data[i : i + 1]
                stm_bits, nstm_bits = perspective_bits(sel)
                a = ftw[np.nonzero(stm_bits[0])[0]].sum(axis=0) + ftb
                b = ftw[np.nonzero(nstm_bits[0])[0]].sum(axis=0) + ftb
                va, vb = np.clip(a, 0, 1), np.clip(b, 0, 1)
                h = np.concatenate([va * va, vb * vb])
                manual = float(h @ outw + outb[0])
                si, so, ni, no, _, _ = make_batch(sel)
                got = float(model(si, so, ni, no)[0])
                max_diff = max(max_diff, abs(manual - got))
        check(f"numpy forward matches model (max diff {max_diff:.2e} <= 1e-4)",
              max_diff <= 1e-4)
    finally:
        os.unlink(path)

    print("selfcheck: ALL PASS" if ok else "selfcheck: FAILED", flush=True)
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip(), flush=True)
        sys.exit(2)
    if sys.argv[1] == "selfcheck":
        selfcheck()
        return
    if len(sys.argv) < 4:
        print("usage: train_nnue.py selfcheck | "
              "train_nnue.py <out.bin> <epochs> <shard1.bin> [shard2.bin ...]",
              flush=True)
        sys.exit(2)
    out_path = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    shards = sys.argv[3:]
    torch.manual_seed(42)
    train(shards, out_path, epochs)


if __name__ == "__main__":
    main()
