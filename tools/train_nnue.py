#!/usr/bin/env python3
"""Train the NNUE eval net for Unchessed AI (v3: HalfKA king-relative
features, GPU-resident pipeline). PyTorch, CPU or CUDA.

Input: shard .bin files of 104-byte records, side-to-move normalized:
  12 x u64 LE bitboards (planes 0-5 = mover P,N,B,R,Q,K; 6-11 = opponent;
  board vertically flipped when black was to move, so bit i = square i with
  a1=0..h8=63 in the mover-up frame)
  + i16 LE search score in centipawns (stm pov)
  + u8 wdl (2 = mover won, 1 = draw, 0 = lost) + 5 pad bytes.

Features (v3, HalfKA-style, king-relative, no mirroring/bucketing yet):
  feature = king_sq*704 + piece_idx*64 + sq, where king_sq is the OWN king's
  square (the anchor, never itself an active feature), and piece_idx enum-
  erates the 11 non-own-king planes in order [ownP,N,B,R,Q, oppP,N,B,R,Q,K].
  FT_IN = 64 * 11 * 64 = 45056. STM perspective computed directly from the
  mover-perspective planes; NSTM perspective derived from the same unpacked
  bit-planes via a plane-group swap + rank flip (sq -> sq^56), entirely in
  torch so it can run on DEVICE (see unpack_planes/both_perspectives below)
  instead of numpy -- this is the "GPU-resident pipeline" change: the only
  CPU-bound step left is reading shard bytes off disk into the preloaded
  in-RAM array; everything from raw bb bytes onward (bit unpacking, feature-
  index construction, target computation) runs as tensor ops on DEVICE.

Model: shared EmbeddingBag(45056 -> 256, sum) feature transformer + bias.
Output head concatenates BOTH SCReLU (clamp(x,0,1)^2) and plain ClippedReLU
(clamp(x,0,1)) of each perspective's accumulator (SFNNv5 trick) -> 4*256 =
1024 -> Linear(1024, 1). Raw output unit is cp/400: the Rust engine computes
eval_cp = raw * 400. Training loss: |sigmoid(raw) - target|^2.5 (not plain
MSE), target = 0.7*sigmoid(cp/400) + 0.3*(wdl/2).

Export: b"UNCHNNUE", u32 version=2, u32 ft_in=45056, u32 acc=256,
ft weights [45056][256] f32, ft bias [256] f32, out weights [1024] f32
(order: STM SCReLU, STM ClippedReLU, NSTM SCReLU, NSTM ClippedReLU), out
bias [1] f32. All LE.

NOTE: this is a breaking format change from v1 (flat-768 features, 512-wide
out layer, version=1). The Rust engine's NNUE loader/inference code
(unchessed-core) MUST be updated to match this feature scheme (king-relative
indexing, not flat plane*64+sq) AND the new version/out-layer size before a
net trained with this script can be loaded and played -- this script alone
does not make that pairing automatic. Existing v1 unchessed-nnue.bin stays
loadable/playable until that Rust-side update lands; don't overwrite it with
a v2 export until the loader is ready.

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
VERSION = 2
N_KING_SQ = 64
KEEP_PLANES = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11])  # exclude own king (plane 5)
N_PLANES_KEPT = len(KEEP_PLANES)  # 11
N_PIECE_SQ = N_PLANES_KEPT * 64  # 704
FT_IN = N_KING_SQ * N_PIECE_SQ  # 45056
ACC = 256
MAGIC = b"UNCHNNUE"
HEADER_SIZE = 8 + 3 * 4
PAYLOAD_FLOATS = FT_IN * ACC + ACC + 4 * ACC + 1
# NSTM perspective: use opponent planes as "own" and vice versa.
PLANE_SWAP = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])

# Constant tensors for the torch-native (DEVICE-resident) feature pipeline.
_BIT_SHIFTS = torch.arange(64, dtype=torch.int64, device=DEVICE)
_PLANE_SWAP_T = torch.tensor(PLANE_SWAP, dtype=torch.int64, device=DEVICE)
_KEEP_PLANES_T = torch.tensor(KEEP_PLANES, dtype=torch.int64, device=DEVICE)


def screlu(x):
    v = x.clamp(0.0, 1.0)
    return v * v


def crelu(x):
    return x.clamp(0.0, 1.0)


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
        self.out = nn.Linear(4 * ACC, 1, bias=True)
        # Default EmbeddingBag init is N(0,1); with ~28 active features the
        # sum would saturate SCReLU immediately. Small uniform init instead.
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)

    def forward(self, stm_idx, stm_off, nstm_idx, nstm_off):
        acc_stm = self.ft(stm_idx, stm_off) + self.ft_bias
        acc_nstm = self.ft(nstm_idx, nstm_off) + self.ft_bias
        # SFNNv5 trick: concat plain ClippedReLU alongside SCReLU so the
        # output layer sees both the squared and linear activation shape.
        h = torch.cat(
            [screlu(acc_stm), crelu(acc_stm), screlu(acc_nstm), crelu(acc_nstm)],
            dim=1,
        )
        return self.out(h).squeeze(1)  # raw output ~ cp/400


def unpack_planes(bb):
    """bb: [n, 12] int64 tensor on DEVICE (raw bitboard bits, reinterpreted
    from uint64 -- sign is irrelevant, only used for bitwise ops below).
    Returns [n, 12, 64] uint8 one-hot per plane/square, bit i -> square i
    (a1=0..h8=63). Runs entirely as tensor ops so it executes on DEVICE
    (GPU) when available instead of CPU-bound numpy unpackbits."""
    shifted = bb.unsqueeze(-1) >> _BIT_SHIFTS  # [n, 12, 64]
    return (shifted & 1).to(torch.uint8)


def both_perspectives(bits):
    """bits: [n, 12, 64] mover-perspective one-hot planes (0-5 own P..K,
    6-11 opp same). Returns (stm_bits, nstm_bits). NSTM is derived from the
    already-unpacked STM bits via a plane-group swap + rank flip (sq ->
    sq^56 == reversing the 8-square rank groups), which is algebraically
    equivalent to byte-swapping the raw u64 before unpacking but stays in
    tensor-land so it can run on DEVICE."""
    n = bits.shape[0]
    nstm = (
        bits[:, _PLANE_SWAP_T, :]
        .reshape(n, 12, 8, 8)
        .flip(dims=(2,))
        .reshape(n, 12, 64)
    )
    return bits, nstm


def halfka_indices(bits):
    """bits: [n, 12, 64] one-hot planes, one perspective (own king = plane
    5). HalfKA feature index = king_sq*704 + piece_idx*64 + sq over the 11
    non-own-king planes. Returns flat feature indices + offsets [n+1]
    (int64 tensors on DEVICE) for EmbeddingBag's variable-length bags."""
    n = bits.shape[0]
    king_sq = bits[:, 5, :].argmax(dim=1)  # [n], exactly one bit set
    non_king = bits[:, _KEEP_PLANES_T, :]  # [n, 11, 64]
    flat = non_king.reshape(n, -1)  # [n, 704]
    nz = flat.nonzero(as_tuple=False)  # [nnz, 2] -> (row, col)
    rows, cols = nz[:, 0], nz[:, 1]
    feat = king_sq[rows] * N_PIECE_SQ + cols
    counts = flat.sum(dim=1)
    offsets = torch.zeros(n + 1, dtype=torch.int64, device=bits.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return feat.to(torch.int64), offsets


def _features_and_target(bb, score, wdl):
    """bb: [n, 12] int64 tensor on DEVICE (raw bitboard bits, reinterpreted
    from uint64). score: [n] int-ish tensor on DEVICE. wdl: [n] int-ish
    tensor on DEVICE. Shared by both the host-resident (make_batch) and
    GPU-resident (make_batch_resident) data paths below."""
    bits = unpack_planes(bb)
    stm_bits, nstm_bits = both_perspectives(bits)
    stm_idx, stm_off = halfka_indices(stm_bits)
    nstm_idx, nstm_off = halfka_indices(nstm_bits)
    score_f = score.to(torch.float32)
    target = 0.7 * torch.sigmoid(score_f / 400.0) + 0.3 * (wdl.to(torch.float32) / 2.0)
    return stm_idx, stm_off, nstm_idx, nstm_off, target, score_f


def make_batch(sel):
    """Host-resident path: sel is a structured-array slice living in host
    RAM (numpy). Ships just this batch's raw bytes to DEVICE."""
    bb_np = np.ascontiguousarray(sel["bb"]).view(np.int64)  # bit-reinterpret only
    bb = torch.from_numpy(bb_np).to(DEVICE, non_blocking=True)
    score = torch.from_numpy(np.ascontiguousarray(sel["score"])).to(
        DEVICE, non_blocking=True
    )
    wdl = torch.from_numpy(np.ascontiguousarray(sel["wdl"])).to(
        DEVICE, non_blocking=True
    )
    return _features_and_target(bb, score, wdl)


def make_batch_resident(bb_all, score_all, wdl_all, idx_t):
    """GPU-resident path: bb_all/score_all/wdl_all are the FULL dataset,
    already living on DEVICE as tensors; idx_t is an index tensor also on
    DEVICE. No host<->device transfer and no CPU-side numpy fancy-indexing
    happen here at all -- both the gather and the feature extraction run as
    DEVICE tensor ops. Used when the whole dataset fits comfortably in GPU
    memory (raw records are only 104 bytes each, so even 500M+ records is a
    two-digit-GB footprint -- cheap relative to a modern training GPU's
    VRAM), to avoid the repeated per-epoch host RAM traffic and numpy
    indexing cost of re-touching the same host array 15+ times."""
    bb = bb_all.index_select(0, idx_t)
    score = score_all.index_select(0, idx_t)
    wdl = wdl_all.index_select(0, idx_t)
    return _features_and_target(bb, score, wdl)


def batches(data, idx, bs):
    for s in range(0, len(idx), bs):
        yield make_batch(data[idx[s : s + bs]])


def batches_resident(bb_all, score_all, wdl_all, idx_t, bs):
    for s in range(0, len(idx_t), bs):
        yield make_batch_resident(bb_all, score_all, wdl_all, idx_t[s : s + bs])


def evaluate_iter(model, batch_iter):
    """(val MSE loss, val MAE in centipawns), given any iterable of batches
    (either batches() or batches_resident())."""
    se = ae = n = 0.0
    with torch.no_grad():
        for si, so, ni, no, target, score in batch_iter:
            raw = model(si, so, ni, no)
            se += ((torch.sigmoid(raw) - target) ** 2).sum().item()
            ae += (raw * 400.0 - score).abs().sum().item()
            n += len(target)
    return se / max(n, 1), ae / max(n, 1)


def export_net(model, path):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, FT_IN, ACC))
        # EmbeddingBag.weight is [45056][256] = one row per feature: write as-is.
        f.write(model.ft.weight.detach().cpu().numpy().astype("<f4").tobytes())
        f.write(model.ft_bias.detach().cpu().numpy().astype("<f4").tobytes())
        # out.weight is [1, 1024]; order matches forward()'s concat:
        # [STM SCReLU, STM ClippedReLU, NSTM SCReLU, NSTM ClippedReLU].
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
    val_idx_np, train_idx_np = perm[:n_val], perm[n_val:]
    print(f"total {n} records: {len(train_idx_np)} train / {n_val} val", flush=True)
    print(f"device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else "")
          + f", batch size: {BATCH_SIZE}, ft_in: {FT_IN}", flush=True)

    # GPU-resident mode: on CUDA, ship the WHOLE dataset to DEVICE once (raw
    # records are only 104 bytes each, so even several hundred million
    # records is a two-digit-GB footprint -- cheap relative to a training
    # GPU's VRAM) and free the host copy. Every epoch's shuffle/gather then
    # runs as a DEVICE tensor op instead of repeated host<->device transfer
    # plus CPU-bound numpy fancy-indexing of the same array 15+ times. On
    # CPU there's no separate memory pool to move into, so this is skipped
    # and the original host-resident path (data[idx] numpy slicing) is used.
    gpu_resident = DEVICE.type == "cuda"
    if gpu_resident:
        print(f"GPU-resident mode: transferring full dataset "
              f"({data.nbytes / 1e9:.1f} GB) to {DEVICE}...", flush=True)
        bb_all = torch.from_numpy(np.ascontiguousarray(data["bb"]).view(np.int64)).to(DEVICE)
        score_all = torch.from_numpy(np.ascontiguousarray(data["score"])).to(DEVICE)
        wdl_all = torch.from_numpy(np.ascontiguousarray(data["wdl"])).to(DEVICE)
        del data
        train_idx = torch.from_numpy(train_idx_np).to(DEVICE)
        val_idx = torch.from_numpy(val_idx_np).to(DEVICE)
    else:
        train_idx, val_idx = train_idx_np, val_idx_np

    model = Nnue().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    drop1, drop2 = int(epochs * 0.6), int(epochs * 0.8)

    for ep in range(epochs):
        lr = 1e-3 * 0.3 ** ((ep >= drop1) + (ep >= drop2))
        for g in opt.param_groups:
            g["lr"] = lr
        if gpu_resident:
            train_idx = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
            train_iter = batches_resident(bb_all, score_all, wdl_all, train_idx, BATCH_SIZE)
        else:
            train_idx = rng.permutation(train_idx)
            train_iter = batches(data, train_idx, BATCH_SIZE)
        t0 = time.time()
        running = steps = 0
        for si, so, ni, no, target, _ in train_iter:
            opt.zero_grad()
            loss = wdl_loss(model(si, so, ni, no), target)
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        t_train = time.time() - t0
        if gpu_resident:
            val_iter = batches_resident(bb_all, score_all, wdl_all, val_idx, BATCH_SIZE)
        else:
            val_iter = batches(data, val_idx, BATCH_SIZE)
        val_loss, val_mae = evaluate_iter(model, val_iter)
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


def _manual_bits(bb_row):
    """bb_row: [12] uint64 array. Plain-Python independent bit-unpack, used
    only by selfcheck's cross-verification below -- deliberately NOT
    sharing code with unpack_planes/halfka_indices, so a shared bug in the
    fast torch training-path implementation wouldn't hide from the check."""
    bits = np.zeros((12, 64), dtype=np.uint8)
    for p in range(12):
        v = int(bb_row[p])
        for s in range(64):
            if (v >> s) & 1:
                bits[p, s] = 1
    return bits


def _manual_nstm_bits(bits):
    nstm = np.zeros((12, 64), dtype=np.uint8)
    for p in range(12):
        for s in range(64):
            nstm[p, s] = bits[PLANE_SWAP[p], s ^ 56]
    return nstm


def _manual_halfka(bits):
    king_sq = int(np.nonzero(bits[5])[0][0])
    feats = []
    piece_idx = 0
    for p in range(12):
        if p == 5:
            continue
        for s in range(64):
            if bits[p, s]:
                feats.append(king_sq * N_PIECE_SQ + piece_idx * 64 + s)
        piece_idx += 1
    return feats


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
        check(f"version == {VERSION}", version == VERSION)
        check(f"ft_in == {FT_IN}", ft_in == FT_IN)
        check(f"acc == {ACC}", acc == ACC)

        off = HEADER_SIZE
        ftw = np.frombuffer(blob, "<f4", FT_IN * ACC, off).reshape(FT_IN, ACC)
        off += FT_IN * ACC * 4
        ftb = np.frombuffer(blob, "<f4", ACC, off)
        off += ACC * 4
        outw = np.frombuffer(blob, "<f4", 4 * ACC, off)
        off += 4 * ACC * 4
        outb = np.frombuffer(blob, "<f4", 1, off)

        def screlu_np(x):
            v = np.clip(x, 0, 1)
            return v * v

        def crelu_np(x):
            return np.clip(x, 0, 1)

        # Manual numpy forward from the exported arrays vs model(), using
        # the independent _manual_* HalfKA reimplementation above.
        max_diff = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(10):
                sel = data[i : i + 1]
                bits = _manual_bits(sel["bb"][0])
                nstm_bits = _manual_nstm_bits(bits)
                stm_feats = _manual_halfka(bits)
                nstm_feats = _manual_halfka(nstm_bits)
                a = ftw[stm_feats].sum(axis=0) + ftb
                b = ftw[nstm_feats].sum(axis=0) + ftb
                h = np.concatenate(
                    [screlu_np(a), crelu_np(a), screlu_np(b), crelu_np(b)]
                )
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
