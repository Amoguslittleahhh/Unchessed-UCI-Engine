#!/usr/bin/env python3
"""Train the compact XT-NNUE threat-residual prototype on an A100 80GB.

Input is the existing 104-byte NNUE record format. Threat relations are derived
on GPU from the 12 bitboards, including slider blockers, so no expanded sidecar
dataset is required. Separate --train and --validation shards are mandatory.

The output is a resumable PyTorch research checkpoint. It is intentionally not
loadable by production Rust until quantized holdout gates pass and the
UNCHNNX4 inference format is frozen.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a100_common import (
    FixedRecordShards,
    atomic_torch_save,
    configure_torch,
    learning_rate,
    load_config,
    model_parameter_count,
    numpy_to_device,
)

NNUE_REC_SPEC = [
    ("bb", "<u8", 12),
    ("score", "<i2"),
    ("wdl", "u1"),
    ("pad", "u1", 5),
]
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
    ],
    dtype=np.int64,
)
PLANE_SWAP = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)
PIDX_TO_PLANE = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11], dtype=np.int64)
PLANE_TO_CLASS = np.array([0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], dtype=np.int64)
POSITION_MAIN = 32 * 12 * 64
POSITION_VIRTUAL = 12 * 64
THREAT_DIMENSIONS = 12 * 12 * 225


def screlu(values):
    clipped = values.clamp(0.0, 1.0)
    return clipped * clipped


def crelu(values):
    return values.clamp(0.0, 1.0)


def unpack_planes(bitboards):
    shifts = torch.arange(64, dtype=torch.int64, device=bitboards.device)
    return ((bitboards.unsqueeze(-1) >> shifts) & 1).bool()


def both_perspectives(bits, plane_swap):
    batch = bits.shape[0]
    opposite = bits[:, plane_swap].reshape(batch, 12, 8, 8).flip(2).reshape(batch, 12, 64)
    return bits, opposite


def halfka_indices(bits, king_buckets, pidx_to_plane):
    batch = bits.shape[0]
    king_square = bits[:, 5].long().argmax(1)
    mirror = king_square.remainder(8) < 4
    mirrored = bits.reshape(batch, 12, 8, 8).flip(3).reshape(batch, 12, 64)
    oriented = torch.where(mirror[:, None, None], mirrored, bits)
    oriented_king = torch.where(mirror, king_square ^ 7, king_square)
    bucket = king_buckets[oriented_king]
    pidx_bits = oriented[:, pidx_to_plane].reshape(batch, -1)
    nonzero = pidx_bits.nonzero(as_tuple=False)
    rows, columns = nonzero[:, 0], nonzero[:, 1]
    counts = pidx_bits.sum(1).long()
    offsets = torch.zeros(batch + 1, dtype=torch.long, device=bits.device)
    offsets[1:] = counts.cumsum(0)
    return bucket[rows] * 768 + columns, columns, offsets


def square_attacks(piece, source, target, own):
    sf, sr = source & 7, source >> 3
    tf, tr = target & 7, target >> 3
    df, dr = tf - sf, tr - sr
    adf, adr = abs(df), abs(dr)
    if piece == 0:
        return dr == (1 if own else -1) and adf == 1
    if piece == 1:
        return (adf, adr) in ((1, 2), (2, 1))
    if piece == 2:
        return adf == adr and adf > 0
    if piece == 3:
        return (df == 0) != (dr == 0)
    if piece == 4:
        return (adf == adr and adf > 0) or ((df == 0) != (dr == 0))
    return max(adf, adr) == 1


def between_mask(source, target):
    sf, sr = source & 7, source >> 3
    tf, tr = target & 7, target >> 3
    df, dr = tf - sf, tr - sr
    if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        return 0
    step_f = 0 if df == 0 else (1 if df > 0 else -1)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    file, rank = sf + step_f, sr + step_r
    mask = 0
    while (file, rank) != (tf, tr):
        mask |= 1 << (rank * 8 + file)
        file += step_f
        rank += step_r
    return mask


class ThreatIndexer:
    def __init__(self, device):
        pseudo = np.zeros((12, 64, 64), dtype=np.bool_)
        for plane in range(12):
            piece = plane % 6
            own = plane < 6
            for source in range(64):
                for target in range(64):
                    pseudo[plane, source, target] = square_attacks(
                        piece, source, target, own
                    )
        between = np.zeros((64, 64), dtype=np.uint64)
        relation = np.zeros((64, 64), dtype=np.int64)
        relation_mirror = np.zeros((64, 64), dtype=np.int64)
        for source in range(64):
            for target in range(64):
                between[source, target] = between_mask(source, target)
                for mirrored, table in ((False, relation), (True, relation_mirror)):
                    first = source ^ 7 if mirrored else source
                    second = target ^ 7 if mirrored else target
                    df = (second & 7) - (first & 7)
                    dr = (second >> 3) - (first >> 3)
                    table[source, target] = (dr + 7) * 15 + df + 7
        self.pseudo = torch.from_numpy(pseudo).to(device)
        self.between = torch.from_numpy(between.view(np.int64)).to(device)
        self.relation = torch.from_numpy(relation).to(device)
        self.relation_mirror = torch.from_numpy(relation_mirror).to(device)
        self.plane_to_class = torch.from_numpy(PLANE_TO_CLASS).to(device)
        self.sources = torch.arange(64, device=device).view(1, 64)

    def indices(self, bits, bitboards):
        batch = bits.shape[0]
        occupied = bits.any(1)
        plane = bits.long().argmax(1)
        pseudo = self.pseudo[plane, self.sources]
        occupancy = bitboards[:, 0].clone()
        for index in range(1, 12):
            occupancy |= bitboards[:, index]
        blocked = (occupancy[:, None, None] & self.between[None]).ne(0)
        active = (
            occupied[:, :, None]
            & occupied[:, None, :]
            & pseudo
            & ~blocked
        )
        nonzero = active.nonzero(as_tuple=False)
        rows, source, target = nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]
        classes = self.plane_to_class[plane]
        attacker_class = classes[rows, source]
        target_class = classes[rows, target]
        king_square = bits[:, 5].long().argmax(1)
        mirror = king_square.remainder(8) < 4
        relation = torch.where(
            mirror[rows], self.relation_mirror[source, target], self.relation[source, target]
        )
        indices = (attacker_class * 12 + target_class) * 225 + relation
        counts = active.sum((1, 2)).long()
        offsets = torch.zeros(batch + 1, dtype=torch.long, device=bits.device)
        offsets[1:] = counts.cumsum(0)
        return indices, offsets, counts


class StackHead(nn.Module):
    def __init__(self, input_width, hidden):
        super().__init__()
        self.first = nn.Linear(input_width, 16)
        self.second = nn.Linear(32, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, values):
        first = self.first(values)
        first = torch.cat((screlu(first), crelu(first)), 1)
        return self.output(crelu(self.second(first))).squeeze(1)


class XtNnue(nn.Module):
    def __init__(self, config):
        super().__init__()
        position_width = config["position_width"]
        threat_width = config["threat_width"]
        self.config = dict(config)
        self.position_width = position_width
        self.threat_width = threat_width
        self.position_main = nn.EmbeddingBag(
            POSITION_MAIN, position_width, mode="sum", include_last_offset=True
        )
        self.position_virtual = nn.EmbeddingBag(
            POSITION_VIRTUAL, position_width, mode="sum", include_last_offset=True
        )
        self.position_bias = nn.Parameter(torch.zeros(position_width))
        self.threat = nn.EmbeddingBag(
            THREAT_DIMENSIONS, threat_width, mode="sum", include_last_offset=True
        )
        self.threat_bias = nn.Parameter(torch.zeros(threat_width))
        input_width = 2 * position_width + 2 * threat_width
        self.heads = nn.ModuleList(
            StackHead(input_width, config["head_hidden"])
            for _ in range(config["phase_stacks"])
        )
        nn.init.uniform_(self.position_main.weight, -0.04, 0.04)
        nn.init.zeros_(self.position_virtual.weight)
        nn.init.uniform_(self.threat.weight, -0.01, 0.01)

    def forward(
        self,
        stm_main,
        stm_virtual,
        stm_offsets,
        nstm_main,
        nstm_virtual,
        nstm_offsets,
        stm_threat,
        stm_threat_offsets,
        nstm_threat,
        nstm_threat_offsets,
        phase_stack,
    ):
        stm_position = (
            self.position_main(stm_main, stm_offsets)
            + self.position_virtual(stm_virtual, stm_offsets)
            + self.position_bias
        )
        nstm_position = (
            self.position_main(nstm_main, nstm_offsets)
            + self.position_virtual(nstm_virtual, nstm_offsets)
            + self.position_bias
        )
        stm_threats = self.threat(stm_threat, stm_threat_offsets) + self.threat_bias
        nstm_threats = self.threat(nstm_threat, nstm_threat_offsets) + self.threat_bias
        values = torch.cat(
            (
                screlu(stm_position),
                screlu(nstm_position),
                crelu(stm_threats),
                crelu(nstm_threats),
            ),
            1,
        )
        all_outputs = torch.stack([head(values) for head in self.heads], 1)
        return all_outputs.gather(1, phase_stack[:, None]).squeeze(1)

    @torch.no_grad()
    def clamp_quantizable_weights(self):
        # Conservative QAT bounds. Exact integer calibration is performed by
        # the later UNCHNNX4 exporter, not assumed here.
        self.position_main.weight.clamp_(-1.5, 1.5)
        self.position_virtual.weight.clamp_(-1.5, 1.5)
        self.threat.weight.clamp_(-0.5, 0.5)
        for head in self.heads:
            head.first.weight.clamp_(-2.0, 2.0)
            head.second.weight.clamp_(-2.0, 2.0)
            head.output.weight.clamp_(-2.0, 2.0)


def make_batch(records, device, constants, threat_indexer, phase_stacks=8):
    bitboards = numpy_to_device(
        np.ascontiguousarray(records["bb"]).view(np.int64), device
    )
    bits = unpack_planes(bitboards)
    stm_bits, nstm_bits = both_perspectives(bits, constants["plane_swap"])
    nstm_bitboards = (
        nstm_bits.long() * constants["bit_weights"].view(1, 1, 64)
    ).sum(2)
    stm_main, stm_virtual, stm_offsets = halfka_indices(
        stm_bits, constants["king_buckets"], constants["pidx_to_plane"]
    )
    nstm_main, nstm_virtual, nstm_offsets = halfka_indices(
        nstm_bits, constants["king_buckets"], constants["pidx_to_plane"]
    )
    stm_threat, stm_threat_offsets, stm_counts = threat_indexer.indices(stm_bits, bitboards)
    nstm_threat, nstm_threat_offsets, nstm_counts = threat_indexer.indices(
        nstm_bits, nstm_bitboards
    )
    if int(stm_counts.max()) > 256 or int(nstm_counts.max()) > 256:
        raise RuntimeError("threat relation bound exceeded; do not truncate training data")
    non_pawn_non_king = bits[:, [1, 2, 3, 4, 7, 8, 9, 10]].sum((1, 2)).long()
    # Bucket by non-pawn/non-king piece count into `phase_stacks` heads,
    # scaled so the same /25 denominator behaves consistently regardless of
    # how many heads the model actually has (selfcheck reduces phase_stacks
    # for speed; the real config uses 8).
    phase_stack = (non_pawn_non_king * phase_stacks // 25).clamp(0, phase_stacks - 1)
    score = numpy_to_device(
        np.ascontiguousarray(records["score"]).astype(np.float32), device
    )
    wdl = numpy_to_device(
        np.ascontiguousarray(records["wdl"]).astype(np.float32), device
    )
    target = 0.70 * torch.sigmoid(score / 400.0) + 0.30 * (wdl / 2.0)
    return (
        stm_main,
        stm_virtual,
        stm_offsets,
        nstm_main,
        nstm_virtual,
        nstm_offsets,
        stm_threat,
        stm_threat_offsets,
        nstm_threat,
        nstm_threat_offsets,
        phase_stack,
        target,
        score,
    )


def xt_loss(raw, target):
    difference = torch.sigmoid(raw) - target
    return (difference.square() * (difference.abs() + 1e-8).sqrt()).mean()


def constants(device):
    return {
        "plane_swap": torch.from_numpy(PLANE_SWAP).to(device),
        "pidx_to_plane": torch.from_numpy(PIDX_TO_PLANE).to(device),
        "king_buckets": torch.from_numpy(KING_BUCKETS).to(device),
        "bit_weights": torch.from_numpy(
            (np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))).view(np.int64)
        ).to(device),
    }


@torch.no_grad()
def evaluate(model, shards, device, fixed, indexer, batch_size, maximum=200_000, phase_stacks=8):
    model.eval()
    total = 0
    total_loss = total_cp = 0.0
    for records in shards.sequential_batches(batch_size, maximum):
        batch = make_batch(records, device, fixed, indexer, phase_stacks)
        raw = model(*batch[:-2])
        total_loss += float(xt_loss(raw, batch[-2])) * len(records)
        prediction_cp = torch.logit(torch.sigmoid(raw).clamp(1e-5, 1 - 1e-5)) * 400.0
        total_cp += float((prediction_cp - batch[-1]).abs().sum())
        total += len(records)
    return {
        "records": total,
        "loss": total_loss / max(1, total),
        "mae_cp": total_cp / max(1, total),
    }


def make_optimizer(model, config, device):
    kwargs = dict(
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.99),
    )
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def train(args):
    config, hardware = load_config(args.config, "xt_nnue")
    device = configure_torch(config["seed"], args.deterministic)
    dtype = np.dtype(NNUE_REC_SPEC)
    train_data = FixedRecordShards(args.train, dtype)
    validation_data = FixedRecordShards(args.validation, dtype)
    rng = np.random.default_rng(config["seed"])
    fixed = constants(device)
    indexer = ThreatIndexer(device)
    raw_model = XtNnue(config).to(device)
    optimizer = make_optimizer(raw_model, config, device)
    global_step = 0
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_loss = float(checkpoint.get("metrics", {}).get("loss", best_loss))
    train_model = raw_model
    if hardware.get("compile", True) and device.type == "cuda" and not args.no_compile:
        train_model = torch.compile(raw_model, mode="max-autotune", dynamic=True)
    total_steps = config["epochs"] * config["steps_per_epoch"]
    print(
        f"device={device} parameters={model_parameter_count(raw_model):,} "
        f"train_records={train_data.total:,} validation_records={validation_data.total:,}",
        flush=True,
    )

    for epoch in range(start_epoch, config["epochs"]):
        train_model.train()
        started = time.monotonic()
        epoch_loss = 0.0
        for _ in range(config["steps_per_epoch"]):
            lr = learning_rate(
                global_step,
                total_steps,
                config["learning_rate"],
                config["warmup_steps"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            records = train_data.sample(rng, config["batch_size"])
            batch = make_batch(records, device, fixed, indexer, config["phase_stacks"])
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp:
                raw = train_model(*batch[:-2])
                loss = xt_loss(raw, batch[-2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config["gradient_clip"])
            optimizer.step()
            raw_model.clamp_quantizable_weights()
            epoch_loss += float(loss.detach())
            global_step += 1

        metrics = evaluate(
            raw_model,
            validation_data,
            device,
            fixed,
            indexer,
            min(config["batch_size"], 4096),
            args.validation_records,
            config["phase_stacks"],
        )
        elapsed = time.monotonic() - started
        print(
            f"epoch={epoch + 1} train_loss={epoch_loss / config['steps_per_epoch']:.6f} "
            f"val_loss={metrics['loss']:.6f} val_mae={metrics['mae_cp']:.2f}cp "
            f"samples_per_second={config['batch_size'] * config['steps_per_epoch'] / elapsed:.0f}",
            flush=True,
        )
        payload = {
            "format": "UNCHNNX4_TRAINING_V1",
            "config": config,
            "epoch": epoch + 1,
            "global_step": global_step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "train_manifest": train_data.manifest(),
            "validation_manifest": validation_data.manifest(),
        }
        atomic_torch_save(payload, args.output)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            atomic_torch_save(payload, str(args.output) + ".best")


def synthetic_records(count, seed=7):
    rng = np.random.default_rng(seed)
    records = np.zeros(count, dtype=np.dtype(NNUE_REC_SPEC))
    for index in range(count):
        squares = rng.choice(64, size=18, replace=False)
        planes = np.zeros((12, 64), dtype=np.uint8)
        planes[5, squares[0]] = 1
        planes[11, squares[1]] = 1
        for square in squares[2:]:
            plane = int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))
            planes[plane, square] = 1
        records["bb"][index] = np.packbits(planes, axis=1, bitorder="little").view("<u8").ravel()
    records["score"] = rng.integers(-1000, 1001, count, dtype=np.int16)
    records["wdl"] = rng.integers(0, 3, count, dtype=np.uint8)
    return records


def selfcheck(args):
    config, _ = load_config(args.config, "xt_nnue")
    config = {**config, "position_width": 64, "threat_width": 16, "phase_stacks": 2}
    device = configure_torch(config["seed"], True)
    fixed = constants(device)
    indexer = ThreatIndexer(device)
    model = XtNnue(config).to(device)
    optimizer = make_optimizer(model, config, device)
    records = synthetic_records(32)
    for _ in range(2):
        batch = make_batch(records, device, fixed, indexer, config["phase_stacks"])
        optimizer.zero_grad(set_to_none=True)
        raw = model(*batch[:-2])
        loss = xt_loss(raw, batch[-2])
        loss.backward()
        optimizer.step()
    assert raw.shape == (32,)
    assert torch.isfinite(loss)
    print(
        f"selfcheck PASS device={device} parameters={model_parameter_count(model):,} "
        f"loss={float(loss.detach()):.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "selfcheck"))
    parser.add_argument("--config", default="config/a100_hybrid_training.json")
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--validation", nargs="+", default=[])
    parser.add_argument("--output", type=Path, default=Path("xt-nnue.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-records", type=int, default=200_000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "selfcheck":
        selfcheck(arguments)
    else:
        if not arguments.train or not arguments.validation:
            raise SystemExit("train requires separate --train and --validation shards")
        train(arguments)
