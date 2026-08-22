#!/usr/bin/env python3
"""Train Hydra Aegis v3's three-stage XT-NNUE on one A100 80GB.

This is the first trainer in the repository that materializes all three Aegis
relation groups: direct occupied-target relations, slider/blocker/behind-target
x-ray triples, and hashed pawn/king topology.  It trains position-only,
direct, and full heads together.  Two independently initialized fast heads
predict mean and log variance so runtime can combine heteroscedastic scale with
ensemble disagreement before selecting a compute tier.

The input remains the compact 104-byte NNUE value record.  The richer 160-byte
UNCHD3R0 ABI is needed by the history-conditioned policy trainer, not by this
board-state-only value branch.  Separate train, calibration, and final
validation shards are mandatory; this script never invents a random split.
"""

from __future__ import annotations

import argparse
import contextlib
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
)
from train_nnue_xt_a100 import (
    NNUE_REC_SPEC,
    POSITION_MAIN,
    POSITION_VIRTUAL,
    PLANE_TO_CLASS,
    ThreatIndexer,
    both_perspectives,
    constants,
    crelu,
    halfka_indices,
    numpy_to_device,
    screlu,
    synthetic_records,
    unpack_planes,
)

XRAY_DIMENSIONS = 12 * 12 * 12 * 8
PAWN_TOPOLOGY_DIMENSIONS = 4096
MAX_XRAYS = 256
MAX_TOPOLOGIES = 34
RAY_DELTAS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
DIRECTION_MIRROR = np.array([0, 7, 6, 5, 4, 3, 2, 1], dtype=np.int64)


class XrayIndexer:
    """Vectorized first/second-occupied-square ray extractor."""

    def __init__(self, device):
        squares = np.zeros((64, 8, 7), dtype=np.int64)
        valid = np.zeros((64, 8, 7), dtype=np.bool_)
        for source in range(64):
            source_file, source_rank = source & 7, source >> 3
            for direction, (df, dr) in enumerate(RAY_DELTAS):
                file, rank = source_file + df, source_rank + dr
                for distance in range(7):
                    if 0 <= file < 8 and 0 <= rank < 8:
                        squares[source, direction, distance] = rank * 8 + file
                        valid[source, direction, distance] = True
                    file += df
                    rank += dr
        slider_direction = np.zeros((12, 8), dtype=np.bool_)
        for plane in range(12):
            piece = plane % 6
            for direction in range(8):
                slider_direction[plane, direction] = (
                    piece == 4 or (piece == 2 and direction % 2 == 1) or (piece == 3 and direction % 2 == 0)
                )
        self.squares = torch.from_numpy(squares).to(device)
        self.valid = torch.from_numpy(valid).to(device)
        self.slider_direction = torch.from_numpy(slider_direction).to(device)
        self.plane_to_class = torch.from_numpy(PLANE_TO_CLASS).to(device)
        self.direction_mirror = torch.from_numpy(DIRECTION_MIRROR).to(device)
        self.square_values = self.squares.view(1, 64, 8, 7)

    def indices(self, bits):
        batch = bits.shape[0]
        occupied = bits.any(1)
        plane = bits.long().argmax(1)
        ray_occupied = occupied[:, self.squares] & self.valid.unsqueeze(0)
        ordinal = ray_occupied.long().cumsum(-1)
        first_mask = ray_occupied & (ordinal == 1)
        second_mask = ray_occupied & (ordinal == 2)
        first_square = (first_mask.long() * self.square_values).sum(-1)
        second_square = (second_mask.long() * self.square_values).sum(-1)
        has_second = second_mask.any(-1)
        directions = self.slider_direction[plane]
        active = occupied.unsqueeze(-1) & directions & has_second
        nonzero = active.nonzero(as_tuple=False)
        rows, source, direction = nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]
        blocker = first_square[rows, source, direction]
        target = second_square[rows, source, direction]
        classes = self.plane_to_class[plane]
        attacker_class = classes[rows, source]
        blocker_class = classes[rows, blocker]
        target_class = classes[rows, target]
        own_king = bits[:, 5].long().argmax(1)
        mirror = own_king.remainder(8) < 4
        canonical_direction = torch.where(
            mirror[rows], self.direction_mirror[direction], direction
        )
        indices = (
            ((attacker_class * 12 + blocker_class) * 12 + target_class) * 8
            + canonical_direction
        )
        counts = active.sum((1, 2)).long()
        if int(counts.max()) > MAX_XRAYS:
            raise RuntimeError("x-ray relation bound exceeded; do not truncate training data")
        offsets = torch.zeros(batch + 1, dtype=torch.long, device=bits.device)
        offsets[1:] = counts.cumsum(0)
        return indices, offsets, counts


class PawnTopologyIndexer:
    """GPU implementation of the Rust 3-file x 4-rank topology hash."""

    def __init__(self, device):
        window = np.zeros((64, 12), dtype=np.int64)
        valid = np.zeros((64, 12), dtype=np.bool_)
        passed = np.zeros((2, 64, 64), dtype=np.bool_)
        adjacent = np.zeros((64, 64), dtype=np.bool_)
        connected = np.zeros((64, 64), dtype=np.bool_)
        doubled = np.zeros((64, 64), dtype=np.bool_)
        lever = np.zeros((2, 64, 64), dtype=np.bool_)
        blocked = np.zeros((2, 64), dtype=np.int64)
        blocked_valid = np.zeros((2, 64), dtype=np.bool_)
        for source in range(64):
            sf, sr = source & 7, source >> 3
            slot = 0
            for dr in (-1, 0, 1, 2):
                for df in (-1, 0, 1):
                    file, rank = sf + df, sr + dr
                    if 0 <= file < 8 and 0 <= rank < 8:
                        window[source, slot] = rank * 8 + file
                        valid[source, slot] = True
                    slot += 1
            for owner, forward in enumerate((1, -1)):
                next_rank = sr + forward
                if 0 <= next_rank < 8:
                    blocked[owner, source] = next_rank * 8 + sf
                    blocked_valid[owner, source] = True
                    for file in (sf - 1, sf + 1):
                        if 0 <= file < 8:
                            lever[owner, source, next_rank * 8 + file] = True
                for other in range(64):
                    of, other_rank = other & 7, other >> 3
                    passed[owner, source, other] = (
                        abs(of - sf) <= 1 and (other_rank - sr) * forward > 0
                    )
            for other in range(64):
                of, other_rank = other & 7, other >> 3
                adjacent[source, other] = abs(of - sf) == 1
                connected[source, other] = abs(of - sf) == 1 and abs(other_rank - sr) <= 1
                doubled[source, other] = of == sf and other != source
        self.window = torch.from_numpy(window).to(device)
        self.valid = torch.from_numpy(valid).to(device)
        self.window_weights = (1 << torch.arange(12, dtype=torch.int64, device=device)).view(1, 1, 12)
        self.passed = torch.from_numpy(passed).to(device)
        self.adjacent = torch.from_numpy(adjacent).to(device)
        self.connected = torch.from_numpy(connected).to(device)
        self.doubled = torch.from_numpy(doubled).to(device)
        self.lever = torch.from_numpy(lever).to(device)
        self.blocked = torch.from_numpy(blocked).to(device)
        self.blocked_valid = torch.from_numpy(blocked_valid).to(device)

    @staticmethod
    def _any_relation(pawns, relation):
        return (pawns[:, None, :] & relation[None, :, :]).any(-1)

    def _pawn_flags(self, friendly, enemy, occupied, owner):
        enemy_ahead = self._any_relation(enemy, self.passed[owner])
        adjacent = self._any_relation(friendly, self.adjacent)
        connected = self._any_relation(friendly, self.connected)
        lever = self._any_relation(enemy, self.lever[owner])
        doubled = self._any_relation(friendly, self.doubled)
        blocked_square = self.blocked[owner]
        is_blocked = occupied[:, blocked_square] & self.blocked_valid[owner].unsqueeze(0)
        return (
            (~enemy_ahead).long()
            | ((~adjacent).long() << 1)
            | (connected.long() << 2)
            | (lever.long() << 3)
            | (doubled.long() << 4)
            | (is_blocked.long() << 5)
        )

    @staticmethod
    def _hash(key):
        key = key ^ (key >> 16)
        key = (key * 0x045D9F3B) & 0xFFFFFFFF
        key = key ^ (key >> 16)
        key = (key * 0x045D9F3B) & 0xFFFFFFFF
        key = key ^ (key >> 16)
        return key & 4095

    def indices(self, bits):
        batch = bits.shape[0]
        own_king = bits[:, 5].long().argmax(1)
        mirror = own_king.remainder(8) < 4
        mirrored = bits.reshape(batch, 12, 8, 8).flip(3).reshape(batch, 12, 64)
        canonical = torch.where(mirror[:, None, None], mirrored, bits)
        own_pawns = canonical[:, 0]
        enemy_pawns = canonical[:, 6]
        occupied = canonical.any(1)
        anchor_class = torch.full((batch, 64), -1, dtype=torch.long, device=bits.device)
        anchor_class.masked_fill_(canonical[:, 0], 0)
        anchor_class.masked_fill_(canonical[:, 6], 1)
        anchor_class.masked_fill_(canonical[:, 5], 2)
        anchor_class.masked_fill_(canonical[:, 11], 3)
        active = anchor_class >= 0

        own_window = (
            (own_pawns[:, self.window] & self.valid.unsqueeze(0)).long()
            * self.window_weights
        ).sum(-1)
        enemy_window = (
            (enemy_pawns[:, self.window] & self.valid.unsqueeze(0)).long()
            * self.window_weights
        ).sum(-1)
        own_flags = self._pawn_flags(own_pawns, enemy_pawns, occupied, 0)
        enemy_flags = self._pawn_flags(enemy_pawns, own_pawns, occupied, 1)
        flags = torch.where(anchor_class == 0, own_flags, torch.zeros_like(own_flags))
        flags = torch.where(anchor_class == 1, enemy_flags, flags)
        key = (
            own_window
            | (enemy_window << 12)
            | (anchor_class.clamp_min(0) << 24)
            | (flags << 26)
        )
        hashed = self._hash(key)
        nonzero = active.nonzero(as_tuple=False)
        rows, squares = nonzero[:, 0], nonzero[:, 1]
        indices = hashed[rows, squares]
        counts = active.sum(1).long()
        if int(counts.max()) > MAX_TOPOLOGIES:
            raise RuntimeError("pawn topology anchor bound exceeded; do not truncate training data")
        offsets = torch.zeros(batch + 1, dtype=torch.long, device=bits.device)
        offsets[1:] = counts.cumsum(0)
        return indices, offsets, counts


class StackHead(nn.Module):
    def __init__(self, input_width, l1, hidden, outputs):
        super().__init__()
        self.first = nn.Linear(input_width, l1)
        self.second = nn.Linear(2 * l1, hidden)
        self.output = nn.Linear(hidden, outputs)

    def forward(self, values):
        first = self.first(values)
        first = torch.cat((screlu(first), crelu(first)), 1)
        return self.output(crelu(self.second(first)))


class PhaseHeads(nn.Module):
    def __init__(self, fast_input, direct_input, full_input, config):
        super().__init__()
        l1 = config["head_l1"]
        hidden = config["head_hidden"]
        self.fast = nn.ModuleList(
            StackHead(fast_input, l1, hidden, 2)
            for _ in range(config["fast_ensemble_members"])
        )
        self.direct = StackHead(direct_input, l1, hidden, 2)
        self.full = StackHead(full_input, l1, hidden, 1)

    def forward(self, fast_values, direct_values, full_values):
        fast = torch.stack([head(fast_values) for head in self.fast], 1)
        direct = self.direct(direct_values)
        full = self.full(full_values).squeeze(1)
        return fast, direct, full


class XtNnueV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        position_width = config["position_width"]
        direct_width = config["direct_threat_width"]
        xray_width = config["xray_width"]
        pawn_width = config["pawn_topology_width"]
        self.position_main = nn.EmbeddingBag(POSITION_MAIN, position_width, mode="sum", include_last_offset=True)
        self.position_virtual = nn.EmbeddingBag(POSITION_VIRTUAL, position_width, mode="sum", include_last_offset=True)
        self.position_bias = nn.Parameter(torch.zeros(position_width))
        self.direct = nn.EmbeddingBag(config["direct_threat_dimensions"], direct_width, mode="sum", include_last_offset=True)
        self.direct_bias = nn.Parameter(torch.zeros(direct_width))
        self.xray = nn.EmbeddingBag(config["xray_dimensions"], xray_width, mode="sum", include_last_offset=True)
        self.xray_bias = nn.Parameter(torch.zeros(xray_width))
        self.pawn = nn.EmbeddingBag(config["pawn_topology_dimensions"], pawn_width, mode="sum", include_last_offset=True)
        self.pawn_bias = nn.Parameter(torch.zeros(pawn_width))
        fast_input = 4 * position_width
        direct_input = ((fast_input + 2 * direct_width + 6 + 31) // 32) * 32
        full_input = ((fast_input + 2 * (direct_width + xray_width + pawn_width) + 6 + 31) // 32) * 32
        self.fast_input = fast_input
        self.direct_input = direct_input
        self.full_input = full_input
        self.heads = nn.ModuleList(
            PhaseHeads(fast_input, direct_input, full_input, config)
            for _ in range(config["phase_stacks"])
        )
        nn.init.uniform_(self.position_main.weight, -0.04, 0.04)
        nn.init.zeros_(self.position_virtual.weight)
        for table in (self.direct, self.xray, self.pawn):
            nn.init.uniform_(table.weight, -0.01, 0.01)

    @staticmethod
    def _pad(values, width):
        return F.pad(values, (0, width - values.shape[1]))

    def forward(self, features):
        (
            stm_main, stm_virtual, stm_offsets, nstm_main, nstm_virtual, nstm_offsets,
            stm_direct, stm_direct_offsets, nstm_direct, nstm_direct_offsets,
            stm_xray, stm_xray_offsets, nstm_xray, nstm_xray_offsets,
            stm_pawn, stm_pawn_offsets, nstm_pawn, nstm_pawn_offsets,
            phase_stack, global_state,
        ) = features
        stm_position = self.position_main(stm_main, stm_offsets) + self.position_virtual(stm_virtual, stm_offsets) + self.position_bias
        nstm_position = self.position_main(nstm_main, nstm_offsets) + self.position_virtual(nstm_virtual, nstm_offsets) + self.position_bias
        stm_direct_value = self.direct(stm_direct, stm_direct_offsets) + self.direct_bias
        nstm_direct_value = self.direct(nstm_direct, nstm_direct_offsets) + self.direct_bias
        stm_xray_value = self.xray(stm_xray, stm_xray_offsets) + self.xray_bias
        nstm_xray_value = self.xray(nstm_xray, nstm_xray_offsets) + self.xray_bias
        stm_pawn_value = self.pawn(stm_pawn, stm_pawn_offsets) + self.pawn_bias
        nstm_pawn_value = self.pawn(nstm_pawn, nstm_pawn_offsets) + self.pawn_bias
        fast_values = torch.cat((screlu(stm_position), crelu(stm_position), screlu(nstm_position), crelu(nstm_position)), 1)
        direct_values = self._pad(torch.cat((fast_values, crelu(stm_direct_value), crelu(nstm_direct_value), global_state), 1), self.direct_input)
        full_values = self._pad(torch.cat((direct_values[:, : self.fast_input + 2 * self.config["direct_threat_width"]], crelu(stm_xray_value), crelu(nstm_xray_value), crelu(stm_pawn_value), crelu(nstm_pawn_value), global_state), 1), self.full_input)
        all_fast, all_direct, all_full = zip(*(head(fast_values, direct_values, full_values) for head in self.heads))
        batch_indices = torch.arange(phase_stack.shape[0], device=phase_stack.device)
        fast = torch.stack(all_fast, 1)[batch_indices, phase_stack]
        direct = torch.stack(all_direct, 1)[batch_indices, phase_stack]
        full = torch.stack(all_full, 1)[batch_indices, phase_stack]
        return fast, direct, full

    @torch.no_grad()
    def clamp_quantizable_weights(self):
        self.position_main.weight.clamp_(-1.5, 1.5)
        self.position_virtual.weight.clamp_(-1.5, 1.5)
        for table in (self.direct, self.xray, self.pawn):
            table.weight.clamp_(-0.5, 0.5)
        for parameter in self.heads.parameters():
            parameter.clamp_(-2.0, 2.0)


def make_batch(records, device, fixed, direct_indexer, xray_indexer, pawn_indexer):
    bitboards = numpy_to_device(np.ascontiguousarray(records["bb"]).view(np.int64), device)
    bits = unpack_planes(bitboards)
    stm_bits, nstm_bits = both_perspectives(bits, fixed["plane_swap"])
    nstm_bitboards = (nstm_bits.long() * fixed["bit_weights"].view(1, 1, 64)).sum(2)
    stm_main, stm_virtual, stm_offsets = halfka_indices(stm_bits, fixed["king_buckets"], fixed["pidx_to_plane"])
    nstm_main, nstm_virtual, nstm_offsets = halfka_indices(nstm_bits, fixed["king_buckets"], fixed["pidx_to_plane"])
    stm_direct, stm_direct_offsets, stm_direct_counts = direct_indexer.indices(stm_bits, bitboards)
    nstm_direct, nstm_direct_offsets, nstm_direct_counts = direct_indexer.indices(nstm_bits, nstm_bitboards)
    if int(stm_direct_counts.max()) > 256 or int(nstm_direct_counts.max()) > 256:
        raise RuntimeError("direct relation bound exceeded; do not truncate training data")
    stm_xray, stm_xray_offsets, _ = xray_indexer.indices(stm_bits)
    nstm_xray, nstm_xray_offsets, _ = xray_indexer.indices(nstm_bits)
    stm_pawn, stm_pawn_offsets, _ = pawn_indexer.indices(stm_bits)
    nstm_pawn, nstm_pawn_offsets, _ = pawn_indexer.indices(nstm_bits)
    non_pawn_non_king = bits[:, [1, 2, 3, 4, 7, 8, 9, 10]].sum((1, 2)).long()
    phase_stack = (non_pawn_non_king * 8 // 25).clamp(0, 7)
    piece_counts = bits.sum(2).float()
    material_weights = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0], device=device)
    own_material = (piece_counts[:, :5] * material_weights).sum(1)
    enemy_material = (piece_counts[:, 6:11] * material_weights).sum(1)
    global_state = torch.stack((
        piece_counts[:, [0, 6]].sum(1) / 16.0,
        piece_counts[:, [1, 2, 7, 8]].sum(1) / 8.0,
        piece_counts[:, [3, 9]].sum(1) / 4.0,
        piece_counts[:, [4, 10]].sum(1) / 2.0,
        non_pawn_non_king.float() / 24.0,
        (own_material - enemy_material) / 39.0,
    ), 1)
    score = numpy_to_device(np.ascontiguousarray(records["score"]).astype(np.float32), device)
    wdl = numpy_to_device(np.ascontiguousarray(records["wdl"]).astype(np.float32), device)
    target = 0.70 * torch.sigmoid(score / 400.0) + 0.30 * (wdl / 2.0)
    features = (
        stm_main, stm_virtual, stm_offsets, nstm_main, nstm_virtual, nstm_offsets,
        stm_direct, stm_direct_offsets, nstm_direct, nstm_direct_offsets,
        stm_xray, stm_xray_offsets, nstm_xray, nstm_xray_offsets,
        stm_pawn, stm_pawn_offsets, nstm_pawn, nstm_pawn_offsets,
        phase_stack, global_state,
    )
    return features, target, score


def powered_loss(raw, target):
    difference = torch.sigmoid(raw) - target
    return (difference.square() * (difference.abs() + 1e-8).sqrt()).mean()


def multi_stage_loss(outputs, target, weights, bootstrap_fast=False):
    fast, direct, full = outputs
    full_loss = powered_loss(full, target)
    direct_mean, direct_logvar = direct[:, 0], direct[:, 1].clamp(-8.0, 4.0)
    direct_difference = full.detach() - direct_mean
    direct_nll = 0.5 * (
        direct_difference.square() * torch.exp(-direct_logvar) + direct_logvar
    ).mean()
    direct_loss = 0.5 * powered_loss(direct_mean, target) + 0.5 * direct_nll
    fast_losses = []
    fast_means = []
    for member in range(fast.shape[1]):
        mean = fast[:, member, 0]
        logvar = fast[:, member, 1].clamp(-8.0, 4.0)
        difference = full.detach() - mean
        per_sample = 0.5 * (difference.square() * torch.exp(-logvar) + logvar)
        if bootstrap_fast:
            # Independent online sub-bagging masks prevent the two members
            # from seeing exactly the same examples while retaining fixed
            # tensor shapes for A100 compilation.
            mask = (torch.rand_like(per_sample) < 0.8).to(per_sample.dtype)
            fast_losses.append((per_sample * mask).sum() / mask.sum().clamp_min(1.0))
        else:
            fast_losses.append(per_sample.mean())
        fast_means.append(mean)
    fast_loss = torch.stack(fast_losses).mean()
    disagreement = (fast_means[0] - fast_means[1]).abs()
    diversity = F.relu(0.005 - disagreement).mean()
    total = (
        weights["full"] * full_loss
        + weights["direct"] * direct_loss
        + weights["fast_heteroscedastic"] * fast_loss
        + weights["ensemble_diversity"] * diversity
    )
    metrics = {
        "full": full_loss.detach(),
        "direct": direct_loss.detach(),
        "fast": fast_loss.detach(),
        "direct_sigma": torch.exp(0.5 * direct_logvar).mean().detach(),
    }
    return total, metrics


def make_optimizer(model, config, device):
    kwargs = dict(lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=(0.9, 0.99))
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


@torch.no_grad()
def evaluate(model, shards, device, fixed, indexers, config, maximum=200_000):
    model.eval()
    total = 0
    sums = {key: 0.0 for key in ("loss", "fast_mae_cp", "direct_mae_cp", "full_mae_cp", "fast_rate", "direct_rate")}
    corr_n = corr_x = corr_y = corr_x2 = corr_y2 = corr_xy = 0.0
    for records in shards.sequential_batches(config["validation_batch_size"], maximum):
        features, target, score = make_batch(records, device, fixed, *indexers)
        outputs = model(features)
        loss, _ = multi_stage_loss(outputs, target, config["loss_weights"])
        fast, direct, full = outputs
        fast_mean = fast[:, :, 0].mean(1)
        aleatoric = torch.exp(fast[:, :, 1].clamp(-8.0, 4.0)).mean(1)
        epistemic = fast[:, :, 0].var(1, unbiased=False)
        fast_sigma = torch.sqrt(aleatoric + epistemic).clamp_min(1e-6)
        direct_mean = direct[:, 0]
        direct_sigma = torch.exp(0.5 * direct[:, 1].clamp(-8.0, 4.0))
        fast_mask = fast_sigma <= config["fast_to_direct_uncertainty"]
        direct_mask = (~fast_mask) & (direct_sigma <= config["direct_to_full_uncertainty"])
        error = (full - fast_mean).abs()
        count = len(records)
        sums["loss"] += float(loss) * count
        sums["fast_mae_cp"] += float((fast_mean * 400.0 - score).abs().sum())
        sums["direct_mae_cp"] += float((direct_mean * 400.0 - score).abs().sum())
        sums["full_mae_cp"] += float((full * 400.0 - score).abs().sum())
        sums["fast_rate"] += int(fast_mask.sum())
        sums["direct_rate"] += int(direct_mask.sum())
        corr_n += count
        corr_x += float(fast_sigma.sum())
        corr_y += float(error.sum())
        corr_x2 += float(fast_sigma.square().sum())
        corr_y2 += float(error.square().sum())
        corr_xy += float((fast_sigma * error).sum())
        total += count
    denominator = math.sqrt(max(0.0, corr_n * corr_x2 - corr_x * corr_x) * max(0.0, corr_n * corr_y2 - corr_y * corr_y))
    correlation = (corr_n * corr_xy - corr_x * corr_y) / denominator if denominator else 0.0
    return {
        "records": total,
        **{key: value / max(1, total) for key, value in sums.items()},
        "full_rate": 1.0 - (sums["fast_rate"] + sums["direct_rate"]) / max(1, total),
        "uncertainty_error_correlation": correlation,
    }


def train(args):
    config, hardware = load_config(args.config, "xt_nnue")
    if abs(sum(config["loss_weights"].values()) - 1.0) > 1e-12:
        raise ValueError("XT v3 loss weights must sum to one")
    device = configure_torch(config["seed"], args.deterministic)
    dtype = np.dtype(NNUE_REC_SPEC)
    train_data = FixedRecordShards(args.train, dtype)
    calibration_data = FixedRecordShards(args.calibration, dtype)
    validation_data = FixedRecordShards(args.validation, dtype)
    rng = np.random.default_rng(config["seed"])
    fixed = constants(device)
    indexers = (ThreatIndexer(device), XrayIndexer(device), PawnTopologyIndexer(device))
    raw_model = XtNnueV3(config).to(device)
    optimizer = make_optimizer(raw_model, config, device)
    global_step = start_epoch = 0
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
    print(f"device={device} parameters={model_parameter_count(raw_model):,} train_records={train_data.total:,} calibration_records={calibration_data.total:,} validation_records={validation_data.total:,}", flush=True)
    for epoch in range(start_epoch, config["epochs"]):
        train_model.train()
        started = time.monotonic()
        epoch_loss = 0.0
        for _ in range(config["steps_per_epoch"]):
            lr = learning_rate(global_step, total_steps, config["learning_rate"], config["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            records = train_data.sample(rng, config["batch_size"])
            features, target, _ = make_batch(records, device, fixed, *indexers)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                outputs = train_model(features)
                loss, _ = multi_stage_loss(
                    outputs,
                    target,
                    config["loss_weights"],
                    bootstrap_fast=config.get("bootstrap_fast_ensemble", True),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config["gradient_clip"])
            optimizer.step()
            raw_model.clamp_quantizable_weights()
            epoch_loss += float(loss.detach())
            global_step += 1
        # Calibration is reported separately and must not be the final holdout.
        calibration_metrics = evaluate(raw_model, calibration_data, device, fixed, indexers, config, args.calibration_records)
        metrics = evaluate(raw_model, validation_data, device, fixed, indexers, config, args.validation_records)
        elapsed = time.monotonic() - started
        print(f"epoch={epoch + 1} train_loss={epoch_loss / config['steps_per_epoch']:.6f} val_loss={metrics['loss']:.6f} full_mae={metrics['full_mae_cp']:.2f}cp tiers={metrics['fast_rate']:.3f}/{metrics['direct_rate']:.3f}/{metrics['full_rate']:.3f} uncertainty_r={metrics['uncertainty_error_correlation']:.3f} samples_per_second={config['batch_size'] * config['steps_per_epoch'] / elapsed:.0f}", flush=True)
        payload = {
            "format": "UNCHAEG3_XT_TRAINING_V1",
            "config": config,
            "epoch": epoch + 1,
            "global_step": global_step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "calibration_metrics": calibration_metrics,
            "metrics": metrics,
            "train_manifest": train_data.manifest(),
            "calibration_manifest": calibration_data.manifest(),
            "validation_manifest": validation_data.manifest(),
        }
        atomic_torch_save(payload, args.output)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            atomic_torch_save(payload, str(args.output) + ".best")


def selfcheck(args):
    config, _ = load_config(args.config, "xt_nnue")
    config = {**config, "position_width": 64, "direct_threat_width": 16, "xray_width": 8, "pawn_topology_width": 8, "phase_stacks": 2, "batch_size": 16, "validation_batch_size": 16}
    device = configure_torch(config["seed"], True)
    fixed = constants(device)
    indexers = (ThreatIndexer(device), XrayIndexer(device), PawnTopologyIndexer(device))
    model = XtNnueV3(config).to(device)
    optimizer = make_optimizer(model, config, device)
    records = synthetic_records(16)
    for _ in range(2):
        features, target, _ = make_batch(records, device, fixed, *indexers)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss, _ = multi_stage_loss(outputs, target, config["loss_weights"])
        loss.backward()
        optimizer.step()
    fast, direct, full = outputs
    assert fast.shape == (16, 2, 2)
    assert direct.shape == (16, 2)
    assert full.shape == (16,)
    assert torch.isfinite(loss)
    print(f"selfcheck PASS device={device} parameters={model_parameter_count(model):,} loss={float(loss):.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "selfcheck"))
    parser.add_argument("--config", default="config/a100_hydra_v3_training.json")
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--calibration", nargs="+", default=[])
    parser.add_argument("--validation", nargs="+", default=[])
    parser.add_argument("--output", type=Path, default=Path("xt-nnue-v3.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--calibration-records", type=int, default=200_000)
    parser.add_argument("--validation-records", type=int, default=200_000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "selfcheck":
        selfcheck(arguments)
    else:
        if not arguments.train or not arguments.calibration or not arguments.validation:
            raise SystemExit("train requires separate --train, --calibration, and --validation shards")
        train(arguments)
