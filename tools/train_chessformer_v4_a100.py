#!/usr/bin/env python3
"""Train Aegis v4's elastic, legal-only Chessformer on an A100 80GB.

Implemented paths:
- nested 2/128, 4/192, and 8/256 width/depth exits;
- legal promotion-aware policy logits over at most 218 supplied actions;
- private rank-16 human/guide policy adapters;
- policy-only Elo/time/eight-ply history conditioning;
- evidential Dirichlet WDL at every exit;
- per-legal-move regret mean and log-scale;
- full-to-shallow policy/value/representation distillation; and
- separate train, calibration, and final holdout shards.

Input is the fixed 64-byte-header/1088-byte `UNCHD4R0` format. Checkpoints are
research artifacts, not production `UNCHAEG4` packages.
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
    atomic_torch_save,
    configure_torch,
    learning_rate,
    load_config,
    model_parameter_count,
    numpy_to_device,
    sha256_file,
)
from aegis_v4_data import (
    ACTION_SENTINEL,
    HEADER_BYTES,
    LEGAL_FLAG_REGRETS,
    MAGIC,
    MAX_LEGAL_ACTIONS,
    POLICY_GUIDE,
    POLICY_HUMAN,
    RECORD_BYTES,
    REGRET_SENTINEL,
    numpy_dtype,
    parse_header,
)


class V4RecordShards:
    """Header-validated random access over fixed Aegis v4 records."""

    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]
        self.dtype = numpy_dtype()
        self.shards = []
        self.active_paths = []
        self.counts = []
        for path in self.paths:
            size = path.stat().st_size
            if size < HEADER_BYTES or (size - HEADER_BYTES) % RECORD_BYTES:
                raise ValueError(f"{path}: not header + N*{RECORD_BYTES} bytes")
            with path.open("rb") as handle:
                count = parse_header(handle.read(HEADER_BYTES))
            physical = (size - HEADER_BYTES) // RECORD_BYTES
            if count != physical:
                raise ValueError(f"{path}: header count {count}, physical count {physical}")
            if count:
                self.shards.append(
                    np.memmap(path, dtype=self.dtype, mode="r", offset=HEADER_BYTES, shape=(count,))
                )
                self.active_paths.append(path)
                self.counts.append(count)
        if not self.shards:
            raise ValueError("no non-empty Aegis v4 shards")
        self.cumulative = np.cumsum(self.counts)
        self.total = int(self.cumulative[-1])

    def gather(self, global_indices):
        """Gather exact global record indices across memory-mapped shards."""
        global_indices = np.asarray(global_indices, dtype=np.int64)
        if global_indices.ndim != 1:
            raise ValueError("global record indices must be one-dimensional")
        if global_indices.size and (
            int(global_indices.min()) < 0 or int(global_indices.max()) >= self.total
        ):
            raise IndexError("global record index outside shard set")
        shard_ids = np.searchsorted(self.cumulative, global_indices, side="right")
        previous = np.where(shard_ids == 0, 0, self.cumulative[shard_ids - 1])
        local_indices = global_indices - previous
        output = np.empty(len(global_indices), dtype=self.dtype)
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            output[mask] = self.shards[int(shard_id)][local_indices[mask]]
        return output

    def sample(self, rng, count):
        global_indices = rng.integers(0, self.total, count, endpoint=False)
        return self.gather(global_indices)

    def sequential_batches(self, batch_size, maximum=None):
        emitted = 0
        for shard in self.shards:
            for start in range(0, len(shard), batch_size):
                if maximum is not None and emitted >= maximum:
                    return
                end = min(start + batch_size, len(shard))
                if maximum is not None:
                    end = min(end, start + maximum - emitted)
                if end > start:
                    batch = shard[start:end]
                    emitted += len(batch)
                    yield batch

    def manifest(self):
        return [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "records": count,
                "sha256": sha256_file(path),
                "magic": MAGIC.decode("ascii"),
            }
            for path, count in zip(self.active_paths, self.counts)
        ]


class ElasticRMSNorm(nn.Module):
    def __init__(self, maximum_width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(maximum_width))

    def forward(self, values, width):
        normalized = values * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return normalized.to(values.dtype) * self.scale[:width]


class ElasticBlock(nn.Module):
    def __init__(self, maximum_width, heads, dropout):
        super().__init__()
        self.maximum_width = maximum_width
        self.heads = heads
        self.dropout = dropout
        self.norm_attention = ElasticRMSNorm(maximum_width)
        self.qkv = nn.Parameter(torch.empty(3, maximum_width, maximum_width))
        self.project = nn.Parameter(torch.empty(maximum_width, maximum_width))
        self.project_bias = nn.Parameter(torch.zeros(maximum_width))
        self.norm_ffn = ElasticRMSNorm(maximum_width)
        self.up = nn.Parameter(torch.empty(2, maximum_width, maximum_width))
        self.up_bias = nn.Parameter(torch.zeros(2, maximum_width))
        self.down = nn.Parameter(torch.empty(maximum_width, maximum_width))
        self.down_bias = nn.Parameter(torch.zeros(maximum_width))
        for parameter in (self.qkv, self.project, self.up, self.down):
            nn.init.xavier_uniform_(parameter.view(-1, maximum_width))

    def forward(self, values, geometric_bias, width):
        batch, tokens, _ = values.shape
        if width % self.heads:
            raise ValueError("elastic width must divide evenly across attention heads")
        normalized = self.norm_attention(values, width)
        qkv_weight = self.qkv[:, :width, :width].reshape(3 * width, width)
        qkv = F.linear(normalized, qkv_weight).view(
            batch, tokens, 3, self.heads, width // self.heads
        )
        query, key, value = qkv.unbind(2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=geometric_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        values = values + F.linear(
            attended, self.project[:width, :width], self.project_bias[:width]
        )
        normalized = self.norm_ffn(values, width)
        up_weight = self.up[:, :width, :width].reshape(2 * width, width)
        hidden, gate = F.linear(
            normalized, up_weight, self.up_bias[:, :width].reshape(2 * width)
        ).chunk(2, dim=-1)
        values = values + F.linear(
            F.silu(gate) * hidden,
            self.down[:width, :width],
            self.down_bias[:width],
        )
        return values


class ElasticGeometricBias(nn.Module):
    def __init__(self, maximum_width, layers, heads, d1, hidden, templates):
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.templates_count = templates
        self.token_projection = nn.Parameter(torch.empty(d1, maximum_width))
        self.compress = nn.Linear(64 * d1, hidden)
        self.norm = nn.Parameter(torch.ones(hidden))
        self.coefficients = nn.ModuleList(
            nn.Linear(hidden, heads * templates, bias=False) for _ in range(layers)
        )
        self.templates = nn.Parameter(torch.empty(templates, 64, 64))
        nn.init.xavier_uniform_(self.token_projection)
        nn.init.normal_(self.templates, std=0.02)

    def context(self, tokens, width):
        projected = F.linear(tokens, self.token_projection[:, :width]).flatten(1)
        hidden = F.gelu(self.compress(projected))
        hidden = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return hidden.to(tokens.dtype) * self.norm

    def bias(self, context, layer):
        coefficients = self.coefficients[layer](context).view(
            -1, self.heads, self.templates_count
        )
        return torch.einsum("bht,tij->bhij", coefficients, self.templates)


class PersonaProjection(nn.Module):
    def __init__(self, maximum_width, rank, bias):
        super().__init__()
        self.rank = rank
        self.weight = nn.Parameter(torch.empty(maximum_width, maximum_width))
        self.bias = nn.Parameter(torch.zeros(maximum_width)) if bias else None
        self.adapter_a = nn.Parameter(torch.empty(2, rank, maximum_width))
        self.adapter_b = nn.Parameter(torch.zeros(2, maximum_width, rank))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.adapter_a, std=0.02)

    def forward(self, values, width, policy_kind, history):
        bias = self.bias[:width] if self.bias is not None else None
        base = F.linear(values, self.weight[:width, :width], bias)
        adapter_input = values + history[:, None, :width]
        a = self.adapter_a[policy_kind, :, :width]
        b = self.adapter_b[policy_kind, :width, :]
        low = torch.einsum("bti,bri->btr", adapter_input, a)
        return base + torch.einsum("btr,bwr->btw", low, b) / self.rank


class AegisV4Chessformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["d_model"]
        layers = config["layers"]
        self.config = dict(config)
        self.piece_embedding = nn.Embedding(13, d)
        self.square_embedding = nn.Embedding(64, d)
        self.castling_embedding = nn.Embedding(16, d)
        self.ep_embedding = nn.Embedding(9, d)
        self.halfmove_embedding = nn.Embedding(16, d)
        self.gab = ElasticGeometricBias(
            d,
            layers,
            config["heads"],
            config["gab_token_projection"],
            config["gab_hidden"],
            config["gab_templates"],
        )
        self.blocks = nn.ModuleList(
            ElasticBlock(d, config["heads"], config["dropout"]) for _ in range(layers)
        )
        self.final_norm = ElasticRMSNorm(d)

        rank = config["policy_adapter_rank"]
        self.policy_body = PersonaProjection(d, rank, True)
        self.policy_source = PersonaProjection(d, rank, False)
        self.policy_target = PersonaProjection(d, rank, False)
        self.promotion_bias = nn.Embedding(5, 1)
        nn.init.zeros_(self.promotion_bias.weight)

        history_width = config["history_width"]
        self.history_from = nn.Embedding(64, history_width)
        self.history_to = nn.Embedding(64, history_width)
        self.history_promotion = nn.Embedding(5, history_width)
        self.history_position = nn.Embedding(config["history_plies"], history_width)
        self.time_embedding = nn.Embedding(config["time_classes"], history_width)
        self.rating_weight = nn.Parameter(torch.empty(history_width))
        self.rating_bias = nn.Parameter(torch.zeros(history_width))
        self.history_project = nn.Linear(history_width, d)
        nn.init.normal_(self.rating_weight, std=0.02)

        self.value_weight = nn.Parameter(torch.empty(3, d))
        self.value_bias = nn.Parameter(torch.zeros(3))
        nn.init.xavier_uniform_(self.value_weight)

        regret_width = config["regret_width"]
        self.regret_from = nn.Parameter(torch.empty(regret_width, d))
        self.regret_to = nn.Parameter(torch.empty(regret_width, d))
        self.regret_promotion = nn.Embedding(5, regret_width)
        self.regret_output = nn.Linear(regret_width, 2)
        nn.init.xavier_uniform_(self.regret_from)
        nn.init.xavier_uniform_(self.regret_to)

        # Shared semantic bank is part of the package even in branch-only
        # training; entropy and exit-representation losses keep it non-degenerate.
        self.concept_bank = nn.Parameter(torch.empty(64, 32))
        self.concept_board = nn.Parameter(torch.empty(32, d))
        self.concept_policy = nn.Parameter(torch.empty(32, d))
        nn.init.normal_(self.concept_bank, std=0.02)
        nn.init.xavier_uniform_(self.concept_board)
        nn.init.xavier_uniform_(self.concept_policy)

    def history_context(self, history, history_len, time_class, rating):
        source = history & 63
        target = (history >> 6) & 63
        kind = history >> 14
        promotion = torch.where(kind == 1, ((history >> 12) & 3) + 1, 0)
        positions = torch.arange(history.shape[1], device=history.device).view(1, -1)
        values = (
            self.history_from(source)
            + self.history_to(target)
            + self.history_promotion(promotion)
            + self.history_position(positions)
        )
        mask = positions < history_len[:, None]
        values = (values * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        normalized_rating = ((rating.float() - 100.0) / 3550.0).clamp(0.0, 1.0)
        values = values + self.time_embedding(time_class)
        values = values + normalized_rating[:, None] * self.rating_weight + self.rating_bias
        return self.history_project(values)

    @staticmethod
    def gather_squares(values, squares):
        return torch.gather(values, 1, squares[:, :, None].expand(-1, -1, values.shape[-1]))

    def forward_path(self, batch, layers, width):
        pieces = batch["pieces"]
        batch_size = pieces.shape[0]
        squares = torch.arange(64, device=pieces.device).expand(batch_size, -1)
        global_state = (
            self.castling_embedding(batch["castling"])
            + self.ep_embedding(batch["ep_file"])
            + self.halfmove_embedding(batch["halfmove_bucket"])
        )[:, None, :width]
        values = (
            self.piece_embedding(pieces)[:, :, :width]
            + self.square_embedding(squares)[:, :, :width]
            + global_state
        )
        context = self.gab.context(values, width)
        for layer in range(layers):
            values = self.blocks[layer](values, self.gab.bias(context, layer), width)
        normalized = self.final_norm(values, width)
        pooled = normalized.mean(1)
        evidence = F.softplus(F.linear(pooled, self.value_weight[:, :width], self.value_bias))

        history = self.history_context(
            batch["history"],
            batch["history_len"],
            batch["time_class"],
            batch["rating"],
        )
        body = F.gelu(
            self.policy_body(normalized, width, batch["policy_kind"], history)
        )
        source_values = self.policy_source(body, width, batch["policy_kind"], history)
        target_values = self.policy_target(body, width, batch["policy_kind"], history)
        actions = batch["safe_actions"]
        source = actions & 63
        target = (actions >> 6) & 63
        promotion = (actions >> 12).clamp(0, 4)
        source_vectors = self.gather_squares(source_values, source)
        target_vectors = self.gather_squares(target_values, target)
        logits = (source_vectors * target_vectors).sum(-1) / math.sqrt(width)
        logits = logits + self.promotion_bias(promotion).squeeze(-1)
        logits = logits.masked_fill(~batch["legal_mask"], -1e4)

        regret_source_all = F.linear(normalized, self.regret_from[:, :width])
        regret_target_all = F.linear(normalized, self.regret_to[:, :width])
        regret_hidden = torch.tanh(
            self.gather_squares(regret_source_all, source)
            + self.gather_squares(regret_target_all, target)
            + self.regret_promotion(promotion)
        )
        regret_raw = self.regret_output(regret_hidden)
        regret_mean = F.softplus(regret_raw[:, :, 0])
        regret_log_scale = regret_raw[:, :, 1].clamp(-8.0, 4.0)

        board_semantic = F.linear(pooled, self.concept_board[:, :width])
        policy_semantic = F.linear(body.mean(1), self.concept_policy[:, :width])
        board_concepts = F.softmax(F.linear(board_semantic, self.concept_bank), -1)
        policy_concepts = F.softmax(F.linear(policy_semantic, self.concept_bank), -1)
        return {
            "logits": logits,
            "evidence": evidence,
            "regret_mean": regret_mean,
            "regret_log_scale": regret_log_scale,
            "representation": pooled,
            "board_concepts": board_concepts,
            "policy_concepts": policy_concepts,
            "width": width,
            "layers": layers,
        }

    def forward(self, batch):
        return [
            self.forward_path(batch, layers, width)
            for layers, width in zip(
                self.config["exit_layers"], self.config["matryoshka_widths"]
            )
        ]


def bitboards_to_pieces(bitboards):
    shifts = torch.arange(64, device=bitboards.device, dtype=torch.int64)
    active = ((bitboards.unsqueeze(-1) >> shifts) & 1).bool()
    pieces = torch.zeros((bitboards.shape[0], 64), dtype=torch.long, device=bitboards.device)
    for plane in range(12):
        pieces.masked_fill_(active[:, plane], plane + 1)
    return pieces


def mirror_encoded_moves(values, mirror):
    source = (values & 63) ^ 7
    target = ((values >> 6) & 63) ^ 7
    transformed = source | (target << 6) | (values & 0xF000)
    return torch.where(mirror[:, None] if values.ndim == 2 else mirror, transformed, values)


def prepare_batch(records, device, augment=False):
    bitboards = numpy_to_device(np.ascontiguousarray(records["bb"]).view(np.int64), device)
    pieces = bitboards_to_pieces(bitboards)
    castling = numpy_to_device(np.ascontiguousarray(records["castling"]).astype(np.int64), device)
    ep_file = np.ascontiguousarray(records["ep_file"]).astype(np.int64)
    ep_file[ep_file == 0xFF] = 8
    ep_file = numpy_to_device(ep_file, device)
    halfmove = numpy_to_device(np.ascontiguousarray(records["halfmove"]).astype(np.int64), device)
    rating = numpy_to_device(np.ascontiguousarray(records["rating"]).astype(np.int64), device)
    time_class = numpy_to_device(np.ascontiguousarray(records["time_class"]).astype(np.int64), device)
    history = numpy_to_device(np.ascontiguousarray(records["history"]).astype(np.int64), device)
    history_len = numpy_to_device(np.ascontiguousarray(records["history_len"]).astype(np.int64), device)
    actions = numpy_to_device(np.ascontiguousarray(records["legal_actions"]).astype(np.int64), device)
    regrets = numpy_to_device(np.ascontiguousarray(records["legal_regrets"]).astype(np.float32), device)
    legal_count = numpy_to_device(np.ascontiguousarray(records["legal_count"]).astype(np.int64), device)
    target_action = numpy_to_device(np.ascontiguousarray(records["target_action"]).astype(np.int64), device)
    teacher_best_action = numpy_to_device(
        np.ascontiguousarray(records["teacher_best_action"]).astype(np.int64), device
    )
    policy_kind = numpy_to_device(np.ascontiguousarray(records["policy_kind"]).astype(np.int64), device)
    legal_flags = numpy_to_device(np.ascontiguousarray(records["legal_flags"]).astype(np.int64), device)
    wdl = numpy_to_device(np.ascontiguousarray(records["wdl"]).astype(np.int64), device)

    slots = torch.arange(MAX_LEGAL_ACTIONS, device=device).view(1, -1)
    legal_mask = slots < legal_count[:, None]
    if augment:
        mirror = torch.rand(len(records), device=device) < 0.5
        board = pieces.view(-1, 8, 8)
        pieces = torch.where(mirror[:, None, None], board.flip(2), board).view(-1, 64)
        mirrored_castling = (
            ((castling & 1) << 1)
            | ((castling & 2) >> 1)
            | ((castling & 4) << 1)
            | ((castling & 8) >> 1)
        )
        castling = torch.where(mirror, mirrored_castling, castling)
        ep_file = torch.where(mirror & (ep_file < 8), 7 - ep_file, ep_file)
        actions = mirror_encoded_moves(actions, mirror)
        target_action = mirror_encoded_moves(target_action, mirror)
        # Preserve the 0xffff no-teacher sentinel while mirroring labelled rows.
        mirrored_teacher = mirror_encoded_moves(teacher_best_action, mirror)
        teacher_best_action = torch.where(
            teacher_best_action == ACTION_SENTINEL,
            teacher_best_action,
            mirrored_teacher,
        )
        history = mirror_encoded_moves(history, mirror)
    safe_actions = torch.where(legal_mask, actions, torch.zeros_like(actions))
    target_matches = (safe_actions == target_action[:, None]) & legal_mask
    if not bool(target_matches.any(1).all()):
        raise RuntimeError("v4 record target action is absent from legal action set")
    target_index = target_matches.long().argmax(1)
    teacher_mask = ((legal_flags & LEGAL_FLAG_REGRETS) != 0)[:, None] & legal_mask
    teacher_matches = (safe_actions == teacher_best_action[:, None]) & teacher_mask
    labelled_rows = teacher_mask.any(1)
    if labelled_rows.any() and not bool(teacher_matches[labelled_rows].any(1).all()):
        raise RuntimeError("v4 teacher-best action is absent from labelled legal set")
    teacher_best_index = teacher_matches.long().argmax(1)
    return {
        "pieces": pieces,
        "castling": castling,
        "ep_file": ep_file,
        "halfmove_bucket": (halfmove // 8).clamp(0, 15),
        "rating": rating,
        "time_class": time_class,
        "history": history,
        "history_len": history_len,
        "actions": actions,
        "safe_actions": safe_actions,
        "legal_mask": legal_mask,
        "legal_count": legal_count,
        "target_index": target_index,
        "teacher_best_index": teacher_best_index,
        "policy_kind": policy_kind.clamp(POLICY_HUMAN, POLICY_GUIDE),
        "teacher_mask": teacher_mask,
        "regret_target": regrets / 400.0,
        "wdl": wdl,
    }


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def evidential_loss(evidence, wdl, anneal):
    alpha = evidence.float() + 1.0
    strength = alpha.sum(1, keepdim=True)
    probability = alpha / strength
    target = F.one_hot(wdl, 3).float()
    fit = ((target - probability).square() + alpha * (strength - alpha) / (strength.square() * (strength + 1.0))).sum(1)
    adjusted = target + (1.0 - target) * alpha
    adjusted_strength = adjusted.sum(1, keepdim=True)
    kl = (
        torch.lgamma(adjusted_strength)
        - torch.lgamma(adjusted).sum(1, keepdim=True)
        - torch.lgamma(torch.tensor(3.0, device=alpha.device))
        + ((adjusted - 1.0) * (torch.digamma(adjusted) - torch.digamma(adjusted_strength))).sum(1, keepdim=True)
    ).squeeze(1)
    return (fit + anneal * kl).mean()


def training_loss(outputs, batch, config, global_step):
    full = outputs[-1]
    human_rows = batch["policy_kind"] == POLICY_HUMAN
    guide_rows = (batch["policy_kind"] == POLICY_GUIDE) & batch["teacher_mask"].any(1)
    hard_ce = F.cross_entropy(full["logits"].float(), batch["target_index"], reduction="none", label_smoothing=0.02)
    human_policy = masked_mean(hard_ce, human_rows)

    target_regret = batch["regret_target"]
    teacher_distribution = F.softmax(
        (-target_regret / (config["teacher_temperature_cp"] / 400.0)).masked_fill(
            ~batch["teacher_mask"], -1e4
        ),
        1,
    )
    guide_ce = -(teacher_distribution * F.log_softmax(full["logits"].float(), 1)).sum(1)
    guide_policy = masked_mean(guide_ce, guide_rows)

    regret_difference = target_regret - full["regret_mean"]
    regret_nll = 0.5 * (
        regret_difference.square() * torch.exp(-2.0 * full["regret_log_scale"])
        + 2.0 * full["regret_log_scale"]
    )
    legal_regret = masked_mean(regret_nll, batch["teacher_mask"])

    anneal = min(1.0, global_step / max(1, config["evidence_kl_anneal_steps"]))
    evidential = torch.stack(
        [evidential_loss(output["evidence"], batch["wdl"], anneal) for output in outputs]
    ).mean()

    full_policy = F.softmax(full["logits"].detach().float(), 1)
    full_wdl = (full["evidence"].detach().float() + 1.0)
    full_wdl = full_wdl / full_wdl.sum(1, keepdim=True)
    exit_losses = []
    representation_losses = []
    for output in outputs[:-1]:
        policy_distill = -(full_policy * F.log_softmax(output["logits"].float(), 1)).sum(1).mean()
        wdl = output["evidence"].float() + 1.0
        wdl = wdl / wdl.sum(1, keepdim=True)
        value_distill = F.kl_div(wdl.log(), full_wdl, reduction="batchmean")
        exit_losses.append(policy_distill + value_distill)
        width = output["width"]
        representation_losses.append(
            (1.0 - F.cosine_similarity(
                output["representation"].float(),
                full["representation"][:, :width].detach().float(),
                dim=1,
            )).mean()
        )
    exit_distillation = torch.stack(exit_losses).mean()
    representation = torch.stack(representation_losses).mean()
    concept_entropy = -(full["board_concepts"] * full["board_concepts"].clamp_min(1e-8).log()).sum(1).mean()
    concept_alignment = F.kl_div(
        full["policy_concepts"].clamp_min(1e-8).log(),
        full["board_concepts"].detach(),
        reduction="batchmean",
    )
    representation = representation + 0.05 * concept_alignment - 0.001 * concept_entropy

    weights = config["loss_weights"]
    total = (
        weights["human_policy"] * human_policy
        + weights["guide_policy"] * guide_policy
        + weights["evidential_wdl"] * evidential
        + weights["legal_regret"] * legal_regret
        + weights["exit_distillation"] * exit_distillation
        + weights["representation"] * representation
    )
    return total, {
        "human_policy": human_policy.detach(),
        "guide_policy": guide_policy.detach(),
        "evidential": evidential.detach(),
        "legal_regret": legal_regret.detach(),
        "exit_distillation": exit_distillation.detach(),
        "representation": representation.detach(),
    }


@torch.no_grad()
def fit_regret_calibration(model, shards, device, config, maximum):
    model.eval()
    scores = []
    for records in shards.sequential_batches(config["validation_batch_size"], maximum):
        batch = prepare_batch(records, device)
        full = model(batch)[-1]
        scale = torch.exp(full["regret_log_scale"]).clamp_min(1e-4)
        nonconformity = (batch["regret_target"] - full["regret_mean"]) / scale
        active = batch["teacher_mask"]
        if active.any():
            scores.append(nonconformity[active].float().cpu())
    if not scores:
        raise RuntimeError("calibration split has no per-action teacher regrets")
    values = torch.cat(scores)
    quantile = max(0.0, float(torch.quantile(values, 0.995, interpolation="higher")))
    coverage = float((values <= quantile).float().mean())
    return {
        "records_limit": maximum,
        "labelled_actions": values.numel(),
        "upper_regret_quantile": quantile,
        "empirical_coverage": coverage,
        "target_coverage": 0.995,
    }


@torch.no_grad()
def evaluate(model, shards, device, config, calibration, maximum):
    model.eval()
    exit_hits = [0 for _ in config["exit_layers"]]
    human_hits = [0 for _ in config["exit_layers"]]
    guide_hits = [0 for _ in config["exit_layers"]]
    total = human_total = guide_total = wdl_hits = 0
    nll = regret_error = labelled_actions = covered_actions = 0.0
    q = calibration["upper_regret_quantile"]
    for records in shards.sequential_batches(config["validation_batch_size"], maximum):
        batch = prepare_batch(records, device)
        outputs = model(batch)
        human_rows = batch["policy_kind"] == POLICY_HUMAN
        guide_rows = (batch["policy_kind"] == POLICY_GUIDE) & batch["teacher_mask"].any(1)
        human_total += int(human_rows.sum())
        guide_total += int(guide_rows.sum())
        for index, output in enumerate(outputs):
            prediction = output["logits"].argmax(1)
            exit_hits[index] += int((prediction == batch["target_index"]).sum())
            human_hits[index] += int(((prediction == batch["target_index"]) & human_rows).sum())
            guide_hits[index] += int(
                ((prediction == batch["teacher_best_index"]) & guide_rows).sum()
            )
        full = outputs[-1]
        nll += float(F.cross_entropy(full["logits"].float(), batch["target_index"], reduction="sum"))
        alpha = full["evidence"] + 1.0
        wdl_hits += int((alpha.argmax(1) == batch["wdl"]).sum())
        mask = batch["teacher_mask"]
        if mask.any():
            target = batch["regret_target"]
            regret_error += float((full["regret_mean"] - target).abs()[mask].sum()) * 400.0
            upper = full["regret_mean"] + q * torch.exp(full["regret_log_scale"])
            covered_actions += int((target[mask] <= upper[mask]).sum())
            labelled_actions += int(mask.sum())
        total += len(records)
    return {
        "records": total,
        "nll": nll / max(1, total),
        "wdl_top1": wdl_hits / max(1, total),
        "exit_played_top1": [hits / max(1, total) for hits in exit_hits],
        "exit_human_top1": [hits / max(1, human_total) for hits in human_hits],
        "exit_guide_best_top1": [hits / max(1, guide_total) for hits in guide_hits],
        "human_records": human_total,
        "guide_records": guide_total,
        "regret_mae_cp": regret_error / max(1, labelled_actions),
        "regret_upper_coverage": covered_actions / max(1, labelled_actions),
        "labelled_actions": int(labelled_actions),
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
    if abs(sum(config["loss_weights"].values()) - 1.0) > 1e-12:
        raise ValueError("v4 Chessformer loss weights must sum to one")
    device = configure_torch(config["seed"], args.deterministic)
    train_data = V4RecordShards(args.train)
    calibration_data = V4RecordShards(args.calibration)
    validation_data = V4RecordShards(args.validation)
    rng = np.random.default_rng(config["seed"])
    raw_model = AegisV4Chessformer(config).to(device)
    optimizer = make_optimizer(raw_model, config, device)
    global_step = start_epoch = 0
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
        f"device={device} parameters={model_parameter_count(raw_model):,} "
        f"train={train_data.total:,} calibration={calibration_data.total:,} "
        f"validation={validation_data.total:,}",
        flush=True,
    )
    for epoch in range(start_epoch, config["epochs"]):
        train_model.train()
        started = time.monotonic()
        epoch_loss = 0.0
        for _ in range(config["steps_per_epoch"]):
            lr = learning_rate(global_step, total_steps, config["learning_rate"], config["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            records = train_data.sample(rng, config["batch_size"])
            batch = prepare_batch(records, device, augment=True)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                outputs = train_model(batch)
                loss, _ = training_loss(outputs, batch, config, global_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config["gradient_clip"])
            optimizer.step()
            epoch_loss += float(loss.detach())
            global_step += 1
        calibration = fit_regret_calibration(
            raw_model, calibration_data, device, config, args.calibration_records
        )
        metrics = evaluate(
            raw_model,
            validation_data,
            device,
            config,
            calibration,
            args.validation_records,
        )
        elapsed = time.monotonic() - started
        human_exits = "/".join(f"{value:.4f}" for value in metrics["exit_human_top1"])
        guide_exits = "/".join(
            f"{value:.4f}" for value in metrics["exit_guide_best_top1"]
        )
        print(
            f"epoch={epoch + 1} train_loss={epoch_loss / config['steps_per_epoch']:.5f} "
            f"val_nll={metrics['nll']:.5f} human_exits={human_exits} "
            f"guide_exits={guide_exits} wdl={metrics['wdl_top1']:.4f} "
            f"regret_mae={metrics['regret_mae_cp']:.1f}cp "
            f"coverage={metrics['regret_upper_coverage']:.4f} "
            f"samples_per_second={config['batch_size'] * config['steps_per_epoch'] / elapsed:.0f}",
            flush=True,
        )
        payload = {
            "format": "UNCHAEG4_CHESSFORMER_TRAINING_V1",
            "config": config,
            "epoch": epoch + 1,
            "global_step": global_step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "calibration": calibration,
            "metrics": metrics,
            "train_manifest": train_data.manifest(),
            "calibration_manifest": calibration_data.manifest(),
            "validation_manifest": validation_data.manifest(),
        }
        atomic_torch_save(payload, args.output)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            atomic_torch_save(payload, str(args.output) + ".best")


def synthetic_records(count, seed=7):
    rng = np.random.default_rng(seed)
    records = np.zeros(count, dtype=numpy_dtype())
    for index in range(count):
        occupied = rng.choice(64, size=18, replace=False)
        planes = np.zeros((12, 64), dtype=np.uint8)
        planes[5, occupied[0]] = 1
        planes[11, occupied[1]] = 1
        for square in occupied[2:]:
            plane = int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))
            planes[plane, square] = 1
        records["bb"][index] = np.packbits(planes, axis=1, bitorder="little").view("<u8").ravel()
        legal_count = 20
        actions = np.arange(legal_count, dtype=np.uint16)
        records["legal_count"][index] = legal_count
        records["legal_actions"][index].fill(ACTION_SENTINEL)
        records["legal_actions"][index, :legal_count] = actions
        records["legal_regrets"][index].fill(REGRET_SENTINEL)
        records["target_action"][index] = actions[index % legal_count]
        records["move"][index] = records["target_action"][index] & 0x0FFF
        records["rating"][index] = 1200 + index * 20
        records["ep_file"][index] = 0xFF
        records["time_class"][index] = index % 5
        records["wdl"][index] = index % 3
        records["policy_kind"][index] = index % 2
        if index % 2:
            records["legal_flags"][index] = LEGAL_FLAG_REGRETS
            records["flags"][index] |= 1 << 3
            records["legal_regrets"][index, :legal_count] = np.arange(legal_count, dtype=np.int16) * 10
            records["teacher_best_action"][index] = 0
            records["best_move"][index] = 0
        else:
            records["teacher_best_action"][index] = ACTION_SENTINEL
    return records


def selfcheck(args):
    config, _ = load_config(args.config, "chessformer")
    config = {
        **config,
        "d_model": 64,
        "layers": 3,
        "heads": 4,
        "ffn": 64,
        "exit_layers": [1, 2, 3],
        "matryoshka_widths": [32, 48, 64],
        "batch_size": 16,
        "validation_batch_size": 16,
    }
    device = configure_torch(config["seed"], True)
    model = AegisV4Chessformer(config).to(device)
    optimizer = make_optimizer(model, config, device)
    records = synthetic_records(16)
    for step in range(2):
        batch = prepare_batch(records, device, augment=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, _ = training_loss(outputs, batch, config, step)
        loss.backward()
        optimizer.step()
    assert len(outputs) == 3
    assert outputs[0]["logits"].shape == (16, MAX_LEGAL_ACTIONS)
    assert outputs[-1]["evidence"].shape == (16, 3)
    assert torch.isfinite(loss)
    print(
        f"selfcheck PASS device={device} parameters={model_parameter_count(model):,} "
        f"loss={float(loss):.5f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "selfcheck"))
    parser.add_argument("--config", default="config/a100_hydra_v4_training.json")
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--calibration", nargs="+", default=[])
    parser.add_argument("--validation", nargs="+", default=[])
    parser.add_argument("--output", type=Path, default=Path("chessformer-v4.pt"))
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
            raise SystemExit("train requires separate --train, --calibration, and --validation v4 shards")
        train(arguments)
