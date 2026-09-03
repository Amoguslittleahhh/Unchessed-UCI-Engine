#!/usr/bin/env python3
"""Train the NNUE eval net for Unchessed AI (v4: HalfKAv2_hm-style features
-- 32-bucket horizontal king mirroring, own-king included as an active
feature, factorized/virtual embedding table during training). PyTorch, CPU
or CUDA.

Input: shard .bin files of 104-byte records, side-to-move normalized:
  12 x u64 LE bitboards (planes 0-5 = mover P,N,B,R,Q,K; 6-11 = opponent;
  board vertically flipped when black was to move, so bit i = square i with
  a1=0..h8=63 in the mover-up frame)
  + i16 LE search score in centipawns (stm pov)
  + u8 wdl (2 = mover won, 1 = draw, 0 = lost) + 5 pad bytes.

Why v4, and why NOT the v3 (unmirrored HalfKA + SFNNv5 concat) architecture:
v3 was SPRT-gated at -70.3 Elo vs v1 (decisive fail). A 100-epoch rerun
showed val-mae bottoming at epoch ~13 then rising steadily while train-loss
kept falling -- textbook overfitting, not undertraining. Root-cause research
(cross-checked against the actual Stockfish and nnue-pytorch source, not
just literature) identified three concrete, verifiable deviations from the
reference HalfKAv2_hm design: (1) no horizontal mirroring (2x too many
rows), (2) own king excluded as an active feature (contrary to Stockfish's
PS_KING category, which both W_KING and B_KING map into), (3) no feature
factorization (the "virtual weight" mechanism nnue-pytorch uses specifically
to give sparse per-bucket rows a denser gradient signal during training).
v4 fixes exactly these three -- and DROPS v3's other, unrelated change (the
SFNNv5 SCReLU+ClippedReLU output-head concat), reverting to v1's plain
512-wide SCReLU-only output head, so this is a clean, single-variable test
of the feature-scheme fix against v1, not another bundle of changes.

Feature scheme (matches nnue-pytorch's HalfKAv2_hm^ exactly, ported from
model/modules/features/halfka_v2_hm.py in official-stockfish/nnue-pytorch):
  - 12 piece-type planes per perspective (own P,N,B,R,Q,K then opp P,N,B,R,
    Q,K, matching this project's existing bb layout), remapped internally to
    nnue-pytorch's own/opp-interleaved p_idx order [ownP,oppP,ownN,oppN,...,
    ownK,oppK] (p_idx 0-11) so the king pair lands consecutively at the end,
    which is what the export merge step needs.
  - Horizontal mirroring: if the (own) king is on files a-d, the whole board
    is file-flipped (sq -> sq^7) before feature computation, so the king is
    always on files e-h.
  - 32 king buckets (KING_BUCKETS table below, identical to Stockfish's and
    nnue-pytorch's), indexed by the ORIENTED (post-mirror) king square.
  - Training-time (factorized) feature index = bucket*768 + p_idx*64 + sq,
    over a [24576, ACC] "main" table (32 buckets x 768 = 12 piece types x 64
    squares), PLUS a shared [768, ACC] "virtual" table indexed by p_idx*64+sq
    alone (no bucket) -- every position's virtual-table lookups are updated
    on every step regardless of king position, giving frequently-starved
    main-table rows a dense auxiliary gradient signal during training. The
    two tables' outputs are summed at the accumulator.
  - Export-time coalesce: virtual weights are added into the main table
    (self.weight + self.virtual_weight.repeat(32, 1)), then the 12-piece-type
    layout is merged down to 11 (own-king and opp-king planes, p_idx 10/11,
    share one 64-wide block in the export format: opp-king's full row, with
    ONE entry overwritten by the own-king's row at the king's own square) --
    this exactly matches nnue-pytorch's get_export_weights(). The exported
    .nnue-style file has NO concept of factorization or 12-vs-11 planes; it
    is a plain [22528, ACC] table (32 buckets x 704 = 11 piece types x 64
    squares), matching what the Rust inference side implements directly.

Model: acc_persp = ft_main(idx) + ft_virtual(vidx) + ft_bias (training only;
ft_main/ft_virtual are coalesced into one table at export). SCReLU
(clamp(x,0,1)^2) on both accumulators, concat [stm, nstm] (512, same as v1)
-> piece-count-bucketed output head: 8 parallel output rows, one per band of
4 pieces. bucket = clamp((pieces - 1) / 4, 0, 7), pieces = popcount of all
12 planes (legal positions have 2..=32 pieces -> buckets 0..=7); the head is
Linear(512, 8) indexed by the position's bucket (MoE-style expert selection,
see docs/research-notes-moe-2507.11181.md). Raw output unit is cp/400.
Training loss: |sigmoid(raw) - target|^2.5,
target = 0.7*sigmoid(cp/400) + 0.3*(wdl/2).

Export: b"UNCHNNUE", u32 version=4, u32 ft_in=22528, u32 acc=256,
ft weights [22528][256] f32 (coalesced + 11-piece-merged), ft bias [256] f32,
out weights [8][512] f32 (per bucket: STM half first, then NSTM half),
out bias [8] f32. All LE. (File version 3 = identical features with a single
non-bucketed output head; both remain loadable by the Rust runtime.)

Usage: train_nnue.py selfcheck
       train_nnue.py <out.bin> <epochs> <shard1.bin> [shard2.bin ...]

`<epochs>` is a safety cap, not a "must complete" count. The trainer
exports the **best-val-MAE checkpoint**, not the last epoch, and stops
early when val-MAE has not improved by EARLY_STOP_MIN_DELTA cp for
EARLY_STOP_PATIENCE epochs (defaults 0.1 / 3; patience 0 disables
early-stop but still exports best). See docs/nnue-v4-training-recipe.md.

Device: auto-detects CUDA; override with DEVICE=cpu|cuda|cuda:0 env var.
Batch size: override with BATCH_SIZE env var (default 16384; production
CPU recipe uses 65536, A100 recipe uses 131072).
"""
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnue_train_control import EarlyStop, lr_at_epoch, recipe_from_env
from nnue_cloud_runtime import apply_torch_speed, cloud_flags, preflight_errors

DEVICE = torch.device(
    os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16384))
_CLOUD = cloud_flags()
_SPEED_NOTES = apply_torch_speed(torch, DEVICE)

REC = np.dtype(
    [
        ("bb", "<u8", 12),
        ("score", "<i2"),
        ("wdl", "u1"),
        ("pad", "u1", 5),
    ]
)
VERSION = 4
N_SQ = 64
N_PT = 12  # training-time piece types (own+opp king kept separate)
N_PLANES = N_SQ * N_PT  # 768
N_BUCKETS = 32
FT_IN_MAIN = N_PLANES * N_BUCKETS  # 24576, training-time "real" (per-bucket) table
FT_IN_VIRTUAL = N_PLANES  # 768, training-time shared/factorized table
N_PT_EXPORT = 11  # own+opp king merged into one category at export
N_PIECE_SQ_EXPORT = N_PT_EXPORT * N_SQ  # 704
FT_IN = N_BUCKETS * N_PIECE_SQ_EXPORT  # 22528, the exported/inference table
ACC = 256
N_OUT_BUCKETS = 8  # piece-count output-bucket (expert) count
MAGIC = b"UNCHNNUE"
HEADER_SIZE = 8 + 3 * 4
PAYLOAD_FLOATS = FT_IN * ACC + ACC + N_OUT_BUCKETS * 2 * ACC + N_OUT_BUCKETS
# NSTM perspective: use opponent planes as "own" and vice versa.
PLANE_SWAP = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])

# Our bb layout is [ownP,N,B,R,Q,K, oppP,N,B,R,Q,K] (planes 0-11). Remap to
# nnue-pytorch's own/opp-interleaved p_idx order [ownP,oppP,ownN,oppN,ownB,
# oppB,ownR,oppR,ownQ,oppQ,ownK,oppK] so the king pair (p_idx 10,11) lands
# consecutively at the end -- required by the export merge logic below.
# PIDX_TO_PLANE[p_idx] = which of our planes 0-11 supplies that p_idx.
PIDX_TO_PLANE = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])

# 32 king buckets, identical to Stockfish's src/nnue/features/half_ka_v2_hm.h
# and nnue-pytorch's model/modules/features/halfka_v2_hm.py KingBuckets
# table. Indexed by the ORIENTED (post-mirror) king square 0-63; only
# entries for files e-h (index%8 >= 4) are valid, since mirroring guarantees
# the king never lands on a-d after orientation.
KING_BUCKETS = np.array(
    [
        -1, -1, -1, -1, 31, 30, 29, 28,
        -1, -1, -1, -1, 27, 26, 25, 24,
        -1, -1, -1, -1, 23, 22, 21, 20,
        -1, -1, -1, -1, 19, 18, 17, 16,
        -1, -1, -1, -1, 15, 14, 13, 12,
        -1, -1, -1, -1, 11, 10, 9, 8,
        -1, -1, -1, -1, 7, 6, 5, 4,
        -1, -1, -1, -1, 3, 2, 1, 0,
    ]
)
# Inverse: INVERSE_KING_BUCKETS[bucket] = the oriented king square, needed
# at export time to know which square within the merged king block belongs
# to the own king for that bucket.
INVERSE_KING_BUCKETS = np.zeros(N_BUCKETS, dtype=np.int64)
for _sq, _bucket in enumerate(KING_BUCKETS):
    if _bucket >= 0:
        INVERSE_KING_BUCKETS[_bucket] = _sq

# Constant tensors for the torch-native (DEVICE-resident) feature pipeline.
_PLANE_SWAP_T = torch.tensor(PLANE_SWAP, dtype=torch.int64, device=DEVICE)
_PIDX_TO_PLANE_T = torch.tensor(PIDX_TO_PLANE, dtype=torch.int64, device=DEVICE)
_KING_BUCKETS_T = torch.tensor(KING_BUCKETS, dtype=torch.int64, device=DEVICE)


def screlu(x):
    v = x.clamp(0.0, 1.0)
    return v * v


def wdl_loss(raw, target, weight=None):
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
    per_sample = diff * diff * (diff.abs() + 1e-8).sqrt()
    if weight is None:
        return per_sample.mean()
    return (per_sample * weight).sum() / weight.sum()


# Predeclared inverse-pool-frequency weights (docs/reinforcement/32-nnue-soft-reweighting.md),
# derived from the 27M-record 178M-corpus pool counts, clipped to [0.25x, 20x] and
# normalized so the RAW-sampled average weight is close to 1. Off by default --
# only applied when NNUE_BUCKET_WEIGHTS=1, so the existing recipe is untouched.
BUCKET_WEIGHT = torch.tensor(
    [20.0, 20.0, 11.7, 2.0, 0.87, 0.58, 0.45, 0.44], dtype=torch.float32, device=DEVICE
)


class Nnue(nn.Module):
    def __init__(self):
        super().__init__()
        # include_last_offset=True: offsets has n+1 entries, last = total nnz.
        self.ft_main = nn.EmbeddingBag(FT_IN_MAIN, ACC, mode="sum", include_last_offset=True)
        self.ft_virtual = nn.EmbeddingBag(FT_IN_VIRTUAL, ACC, mode="sum", include_last_offset=True)
        self.ft_bias = nn.Parameter(torch.zeros(ACC))
        self.out = nn.Linear(2 * ACC, N_OUT_BUCKETS, bias=True)
        # Small uniform init for the main (sparse, per-bucket) table --
        # with ~28 active features the EmbeddingBag sum would saturate
        # SCReLU immediately under a larger init. Virtual table starts at
        # exactly zero (matches nnue-pytorch: it's a pure additive
        # regularizer during training, contributing nothing until learned).
        nn.init.uniform_(self.ft_main.weight, -0.05, 0.05)
        nn.init.zeros_(self.ft_virtual.weight)

    def accumulate(self, idx, off, vidx):
        return self.ft_main(idx, off) + self.ft_virtual(vidx, off) + self.ft_bias

    def forward(self, stm_idx, stm_off, stm_vidx, nstm_idx, nstm_off,
                nstm_vidx, bucket):
        acc_stm = self.accumulate(stm_idx, stm_off, stm_vidx)
        acc_nstm = self.accumulate(nstm_idx, nstm_off, nstm_vidx)
        h = torch.cat([screlu(acc_stm), screlu(acc_nstm)], dim=1)
        raw = self.out(h)  # [n, N_OUT_BUCKETS] -- one row per bucket
        return raw.gather(1, bucket.reshape(-1, 1)).squeeze(1)  # ~ cp/400


def unpack_planes(bb):
    """bb: [n, 12] int64 tensor on DEVICE (raw bitboard bits, reinterpreted
    from uint64 -- sign is irrelevant, only used for bitwise ops below).
    Returns [n, 12, 64] uint8 one-hot per plane/square, bit i -> square i
    (a1=0..h8=63)."""
    shifts = torch.arange(64, dtype=torch.int64, device=bb.device)
    shifted = bb.unsqueeze(-1) >> shifts  # [n, 12, 64]
    return (shifted & 1).to(torch.uint8)


def both_perspectives(bits):
    """bits: [n, 12, 64] mover-perspective one-hot planes (0-5 own P..K,
    6-11 opp same). Returns (stm_bits, nstm_bits). NSTM is derived from the
    already-unpacked STM bits via a plane-group swap + rank flip (sq ->
    sq^56 == reversing the 8-square rank groups)."""
    n = bits.shape[0]
    nstm = (
        bits[:, _PLANE_SWAP_T, :]
        .reshape(n, 12, 8, 8)
        .flip(dims=(2,))
        .reshape(n, 12, 64)
    )
    return bits, nstm


def halfka_v2_hm_indices(bits):
    """bits: [n, 12, 64] one-hot planes (our layout), one perspective (own
    king = plane 5). Returns (main_idx, virtual_idx, offsets) -- flat int64
    tensors on DEVICE for the two EmbeddingBags in Nnue.accumulate(). See
    module docstring for the full feature-scheme description."""
    n = bits.shape[0]
    king_sq = bits[:, 5, :].argmax(dim=1)  # [n], own king, own-perspective
    mirror = (king_sq % 8) < 4  # [n] bool: king on a-d files -> mirror

    # File-mirror the whole board (sq -> sq^7 flips the file: sq=rank*8+file,
    # XOR 7 flips the low 3 bits) for rows that need it.
    mirrored = bits.reshape(n, 12, 8, 8).flip(dims=(3,)).reshape(n, 12, 64)
    bits_oriented = torch.where(mirror.view(n, 1, 1), mirrored, bits)
    king_sq_oriented = torch.where(mirror, king_sq ^ 7, king_sq)
    bucket = _KING_BUCKETS_T[king_sq_oriented]  # [n], always valid post-mirror

    # Reorder planes 0-11 into nnue-pytorch's p_idx convention (own/opp
    # interleaved per piece type, king pair last) before flattening.
    bits_pidx = bits_oriented[:, _PIDX_TO_PLANE_T, :]  # [n, 12, 64]
    flat = bits_pidx.reshape(n, -1)  # [n, 768], col = p_idx*64 + sq

    nz = flat.nonzero(as_tuple=False)  # [nnz, 2] -> (row, col)
    rows, cols = nz[:, 0], nz[:, 1]
    virtual_idx = cols
    main_idx = bucket[rows] * N_PLANES + cols
    counts = flat.sum(dim=1)
    offsets = torch.zeros(n + 1, dtype=torch.int64, device=bits.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return main_idx.to(torch.int64), virtual_idx.to(torch.int64), offsets


def _features_and_target(bb, score, wdl):
    """bb: [n, 12] int64 tensor on DEVICE. score/wdl: [n] int-ish tensors on
    DEVICE. Shared by both the host-resident and GPU-resident data paths."""
    bits = unpack_planes(bb)
    stm_bits, nstm_bits = both_perspectives(bits)
    stm_idx, stm_vidx, stm_off = halfka_v2_hm_indices(stm_bits)
    nstm_idx, nstm_vidx, nstm_off = halfka_v2_hm_indices(nstm_bits)
    # piece-count output bucket (matches the Rust runtime's
    # Nnue::output_bucket): legal positions always hold both kings.
    pieces = bits.sum(dim=(1, 2))  # [n] int64, >= 2
    bucket = ((pieces - 1) // 4).clamp(max=N_OUT_BUCKETS - 1)
    score_f = score.to(torch.float32)
    target = 0.7 * torch.sigmoid(score_f / 400.0) + 0.3 * (wdl.to(torch.float32) / 2.0)
    return stm_idx, stm_off, stm_vidx, nstm_idx, nstm_off, nstm_vidx, target, score_f, bucket


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
    happen here at all."""
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
        for si, so, sv, ni, no, nv, target, score, bkt in batch_iter:
            raw = model(si, so, sv, ni, no, nv, bkt)
            se += ((torch.sigmoid(raw) - target) ** 2).sum().item()
            ae += (raw * 400.0 - score).abs().sum().item()
            n += len(target)
    return se / max(n, 1), ae / max(n, 1)


def coalesced_export_weights(model):
    """Merge virtual weights into the main table (factorization coalesce,
    matching nnue-pytorch's `self.weight + self.virtual_weight.repeat(32, 1)`
    exactly), then remap the 12-piece-type training layout down to the
    11-piece-type export layout (own-king and opp-king planes, p_idx 10/11,
    merged into one 64-wide block: opp-king's row everywhere, except the
    entry at the king's own square which holds the own-king row) --
    matching nnue-pytorch's get_export_weights(). Returns a [22528, ACC]
    float32 numpy array."""
    with torch.no_grad():
        coalesced = model.ft_main.weight + model.ft_virtual.weight.repeat(N_BUCKETS, 1)
    coalesced = coalesced.cpu().numpy().astype("<f4")
    export = np.zeros((FT_IN, ACC), dtype="<f4")
    for b in range(N_BUCKETS):
        src = b * N_PLANES
        dst = b * N_PIECE_SQ_EXPORT
        export[dst : dst + 640] = coalesced[src : src + 640]  # p_idx 0-9 (10 non-king types)
        own_king_src = src + 10 * 64
        opp_king_src = src + 11 * 64
        dst_king = dst + 10 * 64
        ksq = int(INVERSE_KING_BUCKETS[b])
        export[dst_king : dst_king + 64] = coalesced[opp_king_src : opp_king_src + 64]
        export[dst_king + ksq] = coalesced[own_king_src + ksq]
    return export


def export_net(model, path):
    ft_export = coalesced_export_weights(model)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, FT_IN, ACC))
        f.write(ft_export.tobytes())
        f.write(model.ft_bias.detach().cpu().numpy().astype("<f4").tobytes())
        # out.weight is [N_OUT_BUCKETS, 512]; per bucket [:256] = STM half,
        # [256:] = NSTM half, exactly the concat order in forward().
        f.write(
            model.out.weight.detach().cpu().numpy().astype("<f4").reshape(-1).tobytes()
        )
        f.write(model.out.bias.detach().cpu().numpy().astype("<f4").tobytes())


def train(shards, out_path, epochs):
    # Read shard sizes up front and allocate the full array once, then read
    # each shard directly into its slice -- avoids ever holding both the
    # per-shard arrays AND their concatenation in memory simultaneously.
    counts = []
    for p in shards:
        size = os.path.getsize(p)
        if size % REC.itemsize != 0:
            print(f"warning: {p} size {size} not a multiple of {REC.itemsize}, "
                  f"trailing bytes ignored", flush=True)
        counts.append(size // REC.itemsize)
    n = sum(counts)
    go_required = os.environ.get("GO_CLOUD") is not None or os.environ.get("REQUIRE_CLOUD_GO") == "1"
    blockers = preflight_errors(n, DEVICE.type, go_required=go_required)
    if blockers:
        raise SystemExit("ERROR: cloud/train preflight failed:\n  - " + "\n  - ".join(blockers))
    if n < 1000:
        raise SystemExit(f"ERROR: only {n} records total — refusing to train")
    data = np.empty(n, dtype=REC)
    offset = 0
    for p, cnt in zip(shards, counts):
        part = np.fromfile(p, dtype=REC, count=cnt)
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
          + f", batch size: {BATCH_SIZE}, ft_in (export): {FT_IN}", flush=True)

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
    if _CLOUD["torch_compile"] and DEVICE.type == "cuda":
        model = torch.compile(model)
        print("torch.compile enabled (opt-in)", flush=True)
    adam_kwargs = {"lr": 1e-3}
    if _CLOUD["fused_adam"] and DEVICE.type == "cuda":
        try:
            opt = torch.optim.Adam(model.parameters(), fused=True, **adam_kwargs)
            print("Adam fused=True", flush=True)
        except TypeError:
            opt = torch.optim.Adam(model.parameters(), **adam_kwargs)
    else:
        opt = torch.optim.Adam(model.parameters(), **adam_kwargs)
    use_amp = _CLOUD["use_amp"] and DEVICE.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp:
        print("AMP autocast bfloat16", flush=True)
    if _SPEED_NOTES:
        print("speed: " + ", ".join(_SPEED_NOTES), flush=True)
    print(
        f"persona_active={_CLOUD['persona_active']} unarch_hint={_CLOUD['unarch_hint']} "
        f"(Adaptive stays on; hint stays off)",
        flush=True,
    )
    patience, min_delta = recipe_from_env()
    stopper = EarlyStop(patience, min_delta)
    use_bucket_weights = os.environ.get("NNUE_BUCKET_WEIGHTS") == "1"
    if use_bucket_weights:
        print(f"per-bucket loss weighting enabled: {BUCKET_WEIGHT.tolist()}", flush=True)
    n_train = len(train_idx)
    print(
        f"recipe: max_epochs={epochs} early_stop_patience={patience} "
        f"min_delta={min_delta}cp batch={BATCH_SIZE} device={DEVICE} "
        f"n_train={n_train}",
        flush=True,
    )

    best_state = None
    stopped_early = False
    last_epoch = 0
    metrics_path = os.environ.get("METRICS_JSONL", "")
    if metrics_path:
        open(metrics_path, "w").close()
    for ep in range(epochs):
        lr = lr_at_epoch(ep, epochs)
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
        for si, so, sv, ni, no, nv, target, _, bkt in train_iter:
            opt.zero_grad(set_to_none=True)
            w = BUCKET_WEIGHT[bkt] if use_bucket_weights else None
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = wdl_loss(model(si, so, sv, ni, no, nv, bkt), target, w)
            else:
                loss = wdl_loss(model(si, so, sv, ni, no, nv, bkt), target, w)
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
        last_epoch = ep + 1
        samples_seen = last_epoch * n_train
        is_best, should_stop = stopper.update(val_mae)
        mark = " *best*" if is_best else f" (no-improve {stopper.bad}/{patience or 'off'})"
        print(
            f"epoch {last_epoch}/{epochs}: lr {lr:.1e} "
            f"train-loss {running / max(steps, 1):.6f} "
            f"val-loss {val_loss:.6f} val-mae {val_mae:.1f}cp "
            f"samples-seen {samples_seen} "
            f"({n_train / max(t_train, 1e-9):.0f} samples/s, "
            f"{t_train:.0f}s){mark}",
            flush=True,
        )
        if metrics_path and _CLOUD["jsonl_metrics"]:
            import json as _json
            with open(metrics_path, "a") as mf:
                mf.write(_json.dumps({
                    "epoch": last_epoch,
                    "lr": lr,
                    "train_loss": running / max(steps, 1),
                    "val_loss": val_loss,
                    "val_mae_cp": val_mae,
                    "samples_seen": samples_seen,
                    "is_best": is_best,
                    "persona_active": True,
                }) + "\n")
        if is_best:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if should_stop:
            stopped_early = True
            print(
                f"early-stop at epoch {last_epoch}: no val-mae improvement "
                f">= {min_delta}cp for {patience} epochs "
                f"(best epoch {stopper.best_epoch} at {stopper.best:.1f}cp)",
                flush=True,
            )
            break

    if best_state is None:
        raise SystemExit("ERROR: training produced no checkpoint")
    model.load_state_dict(best_state)
    export_net(model, out_path)
    print(
        f"wrote {out_path} ({os.path.getsize(out_path)} bytes) "
        f"best-epoch {stopper.best_epoch}/{last_epoch} "
        f"best-mae {stopper.best:.1f}cp stopped-early={stopped_early}",
        flush=True,
    )
    return {
        "best_epoch": stopper.best_epoch,
        "best_val_mae": stopper.best,
        "last_epoch": last_epoch,
        "stopped_early": stopped_early,
        "n_train": n_train,
        "n_val": n_val,
    }


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
    sharing code with unpack_planes/halfka_v2_hm_indices, so a shared bug
    wouldn't hide from the check."""
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


def _manual_halfka_v2_hm(bits):
    """Independent (plain Python) reimplementation of the export-time
    feature list: king bucket + mirroring + 12->11 piece merge, all done by
    hand without reusing any of the fast torch path's code."""
    king_sq = int(np.nonzero(bits[5])[0][0])
    mirror = (king_sq % 8) < 4
    if mirror:
        oriented = np.zeros_like(bits)
        for p in range(12):
            for s in range(64):
                oriented[p, s ^ 7] = bits[p, s]
        king_sq_o = king_sq ^ 7
    else:
        oriented = bits
        king_sq_o = king_sq
    bucket = int(KING_BUCKETS[king_sq_o])
    assert bucket >= 0

    feats = []
    for our_plane in range(12):
        p_idx = int(np.nonzero(PIDX_TO_PLANE == our_plane)[0][0])
        if p_idx >= 10:
            continue  # king planes handled separately below (merged block)
        for s in range(64):
            if oriented[our_plane, s]:
                feats.append(bucket * N_PIECE_SQ_EXPORT + p_idx * 64 + s)
    # Merged king block: opp king's own square (p_idx=11's active bit) plus
    # the own king's fixed contribution at its own (oriented) square.
    opp_king_plane = int(PIDX_TO_PLANE[11])
    opp_king_sq = int(np.nonzero(oriented[opp_king_plane])[0][0])
    feats.append(bucket * N_PIECE_SQ_EXPORT + 10 * 64 + opp_king_sq)
    feats.append(bucket * N_PIECE_SQ_EXPORT + 10 * 64 + king_sq_o)
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
        si, so, sv, ni, no, nv, target, _, bkt = make_batch(data[idx])
        opt.zero_grad()
        loss = wdl_loss(model(si, so, sv, ni, no, nv, bkt), target)
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
        outw = np.frombuffer(blob, "<f4", N_OUT_BUCKETS * 2 * ACC, off).reshape(
            N_OUT_BUCKETS, 2 * ACC
        )
        off += N_OUT_BUCKETS * 2 * ACC * 4
        outb = np.frombuffer(blob, "<f4", N_OUT_BUCKETS, off)

        def screlu_np(x):
            v = np.clip(x, 0, 1)
            return v * v

        # Manual numpy forward from the exported (coalesced) arrays vs
        # model(), using the independent _manual_* reimplementation above.
        # This exercises the export-time coalesce+merge path, not just the
        # training-time factorized path -- catches bugs the training loop's
        # own forward pass wouldn't.
        max_diff = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(10):
                sel = data[i : i + 1]
                bits = _manual_bits(sel["bb"][0])
                nstm_bits = _manual_nstm_bits(bits)
                stm_feats = _manual_halfka_v2_hm(bits)
                nstm_feats = _manual_halfka_v2_hm(nstm_bits)
                a = ftw[stm_feats].sum(axis=0) + ftb
                b = ftw[nstm_feats].sum(axis=0) + ftb
                h = np.concatenate([screlu_np(a), screlu_np(b)])
                bkt = min((int(bits.sum()) - 1) // 4, N_OUT_BUCKETS - 1)
                manual = float(h @ outw[bkt] + outb[bkt])
                si, so, sv, ni, no, nv, _, _, bkt_t = make_batch(sel)
                got = float(model(si, so, sv, ni, no, nv, bkt_t)[0])
                assert int(bkt_t[0]) == bkt
                max_diff = max(max_diff, abs(manual - got))
        check(f"numpy forward matches model (max diff {max_diff:.2e} <= 1e-3)",
              max_diff <= 1e-3)
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
