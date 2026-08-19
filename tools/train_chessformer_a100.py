#!/usr/bin/env python3
"""Train a project-native Chessformer policy on an A100 80GB.

The current 104-byte UNCHMAIA v2 records are accepted directly. For production
claims, pass separately mined player-disjoint --train and --validation shards;
this script never manufactures a random split that could leak accounts.

Usage:
  train_chessformer_a100.py selfcheck [--config config/a100_hybrid_training.json]
  train_chessformer_a100.py train --train bucket0.bin ... --validation holdout.bin ... \
      --output checkpoints/chessformer.pt [--config ...]

The checkpoint is a training artifact, not the eventual compact UNCHFORM CPU
inference format.
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


POLICY_REC_SPEC = [
    ("bb", "<u8", 12),
    ("mv", "<u2"),
    ("rating", "<u2"),
    ("castle", "u1"),
    ("ep", "u1"),
    ("flags", "u1"),
    ("pad", "u1"),
]


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, values):
        normalized = values * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return normalized.to(values.dtype) * self.scale


class GeometricAttentionBias(nn.Module):
    """Dynamic global mixture of shared 64x64 geometric templates."""

    def __init__(self, d_model, layers, heads, d1, hidden, templates):
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.templates_count = templates
        self.token_projection = nn.Linear(d_model, d1, bias=False)
        self.compress = nn.Linear(64 * d1, hidden)
        self.norm = RMSNorm(hidden)
        self.coefficients = nn.ModuleList(
            nn.Linear(hidden, heads * templates, bias=False) for _ in range(layers)
        )
        self.templates = nn.Parameter(torch.empty(templates, 64, 64))
        nn.init.normal_(self.templates, std=0.02)

    def context(self, tokens):
        compressed = self.token_projection(tokens).flatten(1)
        return self.norm(F.gelu(self.compress(compressed)))

    def bias(self, context, layer):
        coefficients = self.coefficients[layer](context).view(
            -1, self.heads, self.templates_count
        )
        return torch.einsum("bht,tij->bhij", coefficients, self.templates)


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, ffn, dropout):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.norm_attention = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.project = nn.Linear(width, width)
        self.norm_ffn = RMSNorm(width)
        self.up = nn.Linear(width, 2 * ffn)
        self.down = nn.Linear(ffn, width)

    def forward(self, values, geometric_bias):
        batch, tokens, width = values.shape
        normalized = self.norm_attention(values)
        qkv = self.qkv(normalized).view(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=geometric_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        values = values + self.project(attended)
        hidden, gate = self.up(self.norm_ffn(values)).chunk(2, dim=-1)
        values = values + self.down(F.silu(gate) * hidden)
        return values


class ChessformerPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["d_model"]
        layers = config["layers"]
        heads = config["heads"]
        self.config = dict(config)
        self.piece_embedding = nn.Embedding(13, width)
        self.square_embedding = nn.Embedding(64, width)
        self.castling_embedding = nn.Embedding(16, width)
        self.ep_embedding = nn.Embedding(9, width)
        self.rating_embedding = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width, bias=False)
        )
        self.gab = GeometricAttentionBias(
            width,
            layers,
            heads,
            config["gab_token_projection"],
            config["gab_hidden"],
            config["gab_templates"],
        )
        self.blocks = nn.ModuleList(
            EncoderBlock(width, heads, config["ffn"], config["dropout"])
            for _ in range(layers)
        )
        self.final_norm = RMSNorm(width)
        self.policy_body = nn.Linear(width, width)
        self.policy_source = nn.Linear(width, width, bias=False)
        self.policy_target = nn.Linear(width, width, bias=False)

    def forward(self, pieces, castling, ep_file, rating):
        batch = pieces.shape[0]
        squares = torch.arange(64, device=pieces.device).expand(batch, -1)
        rating = ((rating.float() - 100.0) / 3550.0).clamp(0.0, 1.0).unsqueeze(-1)
        global_state = (
            self.castling_embedding(castling)
            + self.ep_embedding(ep_file)
            + self.rating_embedding(rating)
        ).unsqueeze(1)
        values = self.piece_embedding(pieces) + self.square_embedding(squares) + global_state
        gab_context = self.gab.context(values)
        for layer, block in enumerate(self.blocks):
            values = block(values, self.gab.bias(gab_context, layer))
        values = self.policy_body(self.final_norm(values))
        source = self.policy_source(values)
        target = self.policy_target(values)
        logits = torch.matmul(source, target.transpose(1, 2)) / math.sqrt(source.shape[-1])
        return logits.flatten(1)


def bitboards_to_pieces(bitboards):
    shifts = torch.arange(64, device=bitboards.device, dtype=torch.int64)
    active = ((bitboards.unsqueeze(-1) >> shifts) & 1).bool()
    pieces = torch.zeros((bitboards.shape[0], 64), dtype=torch.long, device=bitboards.device)
    for plane in range(12):
        pieces.masked_fill_(active[:, plane], plane + 1)
    return pieces


def prepare_batch(records, device, augment=False):
    bb = numpy_to_device(
        np.ascontiguousarray(records["bb"]).view(np.int64), device
    )
    pieces = bitboards_to_pieces(bb)
    move = numpy_to_device(
        np.ascontiguousarray(records["mv"]).astype(np.int64), device
    )
    rating = numpy_to_device(
        np.ascontiguousarray(records["rating"]).astype(np.int64), device
    )
    castling = numpy_to_device(
        np.ascontiguousarray(records["castle"]).astype(np.int64), device
    )
    ep = np.ascontiguousarray(records["ep"]).astype(np.int64)
    ep[ep == 0xFF] = 8
    ep = numpy_to_device(ep, device)
    flags = numpy_to_device(
        np.ascontiguousarray(records["flags"]).astype(np.int64), device
    )

    if augment:
        mirror = torch.rand(len(records), device=device) < 0.5
        if mirror.any():
            board = pieces.view(-1, 8, 8)
            pieces = torch.where(mirror[:, None, None], board.flip(2), board).view(-1, 64)
            source = move & 63
            target = (move >> 6) & 63
            source = torch.where(mirror, source ^ 7, source)
            target = torch.where(mirror, target ^ 7, target)
            move = source | (target << 6)
            mirrored_castling = (
                ((castling & 1) << 1)
                | ((castling & 2) >> 1)
                | ((castling & 4) << 1)
                | ((castling & 8) >> 1)
            )
            castling = torch.where(mirror, mirrored_castling, castling)
            mirrored_ep = torch.where(ep < 8, 7 - ep, ep)
            ep = torch.where(mirror, mirrored_ep, ep)
    return pieces, castling, ep, rating, move, flags


def weighted_policy_loss(logits, target, flags):
    per_sample = F.cross_entropy(logits.float(), target, reduction="none", label_smoothing=0.02)
    weights = torch.ones_like(per_sample)
    weights += 0.25 * ((flags & 1) != 0)
    weights += 1.00 * ((flags & 2) != 0)
    weights += 1.00 * ((flags & 4) != 0)
    return (per_sample * weights).sum() / weights.sum()


@torch.no_grad()
def evaluate(model, shards, device, batch_size, maximum=200_000):
    model.eval()
    total = hits = 0
    losses = 0.0
    special_hits = torch.zeros(3, dtype=torch.long, device=device)
    special_total = torch.zeros(3, dtype=torch.long, device=device)
    for records in shards.sequential_batches(batch_size, maximum):
        pieces, castle, ep, rating, target, flags = prepare_batch(records, device)
        logits = model(pieces, castle, ep, rating)
        loss = F.cross_entropy(logits.float(), target, reduction="sum")
        prediction = logits.argmax(1)
        correct = prediction == target
        losses += float(loss)
        hits += int(correct.sum())
        total += len(records)
        for index, flag in enumerate((1, 2, 4)):
            mask = (flags & flag) != 0
            special_hits[index] += (correct & mask).sum()
            special_total[index] += mask.sum()
    rates = (special_hits.float() / special_total.clamp_min(1)).cpu().tolist()
    return {
        "records": total,
        "nll": losses / max(1, total),
        "top1": hits / max(1, total),
        "castle_top1": rates[0],
        "ep_top1": rates[1],
        "promotion_top1": rates[2],
        "special_totals": special_total.cpu().tolist(),
    }


def make_optimizer(model, config, device):
    kwargs = dict(
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.98),
    )
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def train(args):
    config, hardware = load_config(args.config, "chessformer")
    device = configure_torch(config["seed"], args.deterministic)
    dtype = np.dtype(POLICY_REC_SPEC)
    train_data = FixedRecordShards(args.train, dtype)
    validation_data = FixedRecordShards(args.validation, dtype)
    rng = np.random.default_rng(config["seed"])

    raw_model = ChessformerPolicy(config).to(device)
    optimizer = make_optimizer(raw_model, config, device)
    global_step = 0
    start_epoch = 0
    best_nll = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_nll = float(checkpoint.get("metrics", {}).get("nll", best_nll))
    train_model = raw_model
    if hardware.get("compile", True) and device.type == "cuda" and not args.no_compile:
        train_model = torch.compile(raw_model, mode="max-autotune")
    total_steps = config["epochs"] * config["steps_per_epoch"]
    print(
        f"device={device} model_parameters={model_parameter_count(raw_model):,} "
        f"train_records={train_data.total:,} validation_records={validation_data.total:,}",
        flush=True,
    )

    for epoch in range(start_epoch, config["epochs"]):
        train_model.train()
        started = time.monotonic()
        accumulated_loss = 0.0
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
            pieces, castle, ep, rating, target, flags = prepare_batch(
                records, device, augment=True
            )
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp:
                logits = train_model(pieces, castle, ep, rating)
                loss = weighted_policy_loss(logits, target, flags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config["gradient_clip"])
            optimizer.step()
            accumulated_loss += float(loss.detach())
            global_step += 1

        metrics = evaluate(
            raw_model,
            validation_data,
            device,
            config["batch_size"],
            args.validation_records,
        )
        elapsed = time.monotonic() - started
        print(
            f"epoch={epoch + 1} train_loss={accumulated_loss / config['steps_per_epoch']:.5f} "
            f"val_nll={metrics['nll']:.5f} val_top1={metrics['top1']:.4f} "
            f"samples_per_second={config['batch_size'] * config['steps_per_epoch'] / elapsed:.0f}",
            flush=True,
        )
        payload = {
            "format": "UNCHFORM_TRAINING_V1",
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
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            atomic_torch_save(payload, str(args.output) + ".best")


def selfcheck(args):
    config, _ = load_config(args.config, "chessformer")
    config = {**config, "d_model": 64, "layers": 2, "heads": 4, "ffn": 128}
    device = configure_torch(config["seed"], True)
    model = ChessformerPolicy(config).to(device)
    optimizer = make_optimizer(model, config, device)
    rng = np.random.default_rng(7)
    records = np.zeros(32, dtype=np.dtype(POLICY_REC_SPEC))
    for index in range(len(records)):
        occupied = rng.choice(64, size=18, replace=False)
        planes = np.zeros((12, 64), dtype=np.uint8)
        planes[5, occupied[0]] = 1
        planes[11, occupied[1]] = 1
        for square in occupied[2:]:
            plane = int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))
            planes[plane, square] = 1
        records["bb"][index] = np.packbits(planes, axis=1, bitorder="little").view("<u8").ravel()
    records["mv"] = rng.integers(0, 4096, len(records), dtype=np.uint16)
    records["rating"] = rng.integers(600, 2600, len(records), dtype=np.uint16)
    records["ep"] = 0xFF
    for _ in range(2):
        pieces, castle, ep, rating, target, flags = prepare_batch(records, device, True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pieces, castle, ep, rating)
        loss = weighted_policy_loss(logits, target, flags)
        loss.backward()
        optimizer.step()
    assert logits.shape == (32, 4096)
    assert torch.isfinite(loss)
    print(
        f"selfcheck PASS device={device} parameters={model_parameter_count(model):,} "
        f"loss={float(loss):.5f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "selfcheck"))
    parser.add_argument("--config", default="config/a100_hybrid_training.json")
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--validation", nargs="+", default=[])
    parser.add_argument("--output", type=Path, default=Path("chessformer.pt"))
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
