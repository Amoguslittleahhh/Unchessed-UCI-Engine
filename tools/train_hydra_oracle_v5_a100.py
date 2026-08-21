#!/usr/bin/env python3
"""Train canonical Unarchitectured v1's 58M offline oracle and distill the runtime student.

The oracle is never loaded by the chess engine. It spends A100 compute on a
16-layer board trunk and four legal-action decoder layers, then teaches the
compact Aegis v4/v5 runtime student. `--auto-batch` probes real forward,
backward, and fused-AdamW allocation and selects the largest microbatch below
the configured 92% VRAM ceiling, retaining an 8% fragmentation/safety reserve.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from a100_common import (
    atomic_torch_save,
    configure_torch,
    learning_rate,
    load_config,
    model_parameter_count,
)
from aegis_v4_data import LEGAL_FLAG_REGRETS, POLICY_GUIDE, POLICY_HUMAN
from unarchitectured_v1_safety import TrainingSafetyController, atomic_json, write_heartbeat
from train_chessformer_v4_a100 import (
    AegisV4Chessformer,
    V4RecordShards,
    evidential_loss,
    masked_mean,
    prepare_batch,
)


class DistributedContext:
    def __init__(self, seed, deterministic=False):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        configure_torch(seed + self.rank, deterministic)
        if self.world_size > 1:
            if not torch.cuda.is_available():
                raise RuntimeError("multi-GPU Apex training requires CUDA/NCCL")
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(hours=6),
            )
            self.device = torch.device("cuda", self.local_rank)
        else:
            requested = os.environ.get("DEVICE")
            self.device = torch.device(
                requested or ("cuda:0" if torch.cuda.is_available() else "cpu")
            )

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.rank == 0

    def minimum_int(self, value):
        if not self.distributed:
            return value
        tensor = torch.tensor(value, dtype=torch.int64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    def sum_floats(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def any_bool(self, value):
        if not self.distributed:
            return bool(value)
        tensor = torch.tensor(int(bool(value)), dtype=torch.int32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def resolve_precision(device, hardware):
    if device.type != "cuda":
        return {"name": "float32", "dtype": None, "uses_scaler": False}
    requested = hardware.get("precision", "auto").lower()
    if requested in ("auto", "bfloat16") and torch.cuda.is_bf16_supported():
        return {"name": "bfloat16", "dtype": torch.bfloat16, "uses_scaler": False}
    if requested == "bfloat16":
        raise RuntimeError("resolved profile requires BF16 but this GPU/PyTorch cannot provide it")
    if requested not in ("auto", "float16"):
        raise ValueError(f"unsupported training precision {requested!r}")
    return {"name": "float16", "dtype": torch.float16, "uses_scaler": True}


def autocast_context(device, precision):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=precision["dtype"])


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def configure_compiler_safety(hardware):
    """Disable Inductor CUDA graphs unless a future isolated gate enables them."""
    if not hardware.get("disable_cudagraphs", True):
        return
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    try:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False
    except (ImportError, AttributeError):
        # The environment flag is retained for PyTorch versions without this
        # public-ish config path.
        pass


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, values):
        return (
            values
            * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ).to(values.dtype) * self.scale


class GeometricAttentionBias(nn.Module):
    def __init__(self, width, layers, heads, d1, hidden, templates):
        super().__init__()
        self.heads = heads
        self.templates_count = templates
        self.token_projection = nn.Linear(width, d1, bias=False)
        self.compress = nn.Linear(64 * d1, hidden)
        self.norm = RMSNorm(hidden)
        self.coefficients = nn.ModuleList(
            nn.Linear(hidden, heads * templates, bias=False) for _ in range(layers)
        )
        self.templates = nn.Parameter(torch.empty(templates, 64, 64))
        nn.init.normal_(self.templates, std=0.02)

    def context(self, values):
        return self.norm(F.gelu(self.compress(self.token_projection(values).flatten(1))))

    def bias(self, context, layer):
        coefficients = self.coefficients[layer](context).view(
            -1, self.heads, self.templates_count
        )
        return torch.einsum("bht,tij->bhij", coefficients, self.templates)


class BoardBlock(nn.Module):
    def __init__(self, width, heads, ffn, dropout):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.norm_attention = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.project = nn.Linear(width, width)
        self.norm_ffn = RMSNorm(width)
        self.up = nn.Linear(width, 2 * ffn)
        self.down = nn.Linear(ffn, width)

    def forward(self, values, bias):
        batch, tokens, width = values.shape
        qkv = self.qkv(self.norm_attention(values)).view(
            batch, tokens, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        values = values + self.project(attended.transpose(1, 2).reshape(batch, tokens, width))
        hidden, gate = self.up(self.norm_ffn(values)).chunk(2, dim=-1)
        return values + self.down(F.silu(gate) * hidden)


class LegalDecoderBlock(nn.Module):
    def __init__(self, width, heads, ffn, dropout):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.norm_self = RMSNorm(width)
        self.self_qkv = nn.Linear(width, 3 * width, bias=False)
        self.self_project = nn.Linear(width, width)
        self.norm_cross = RMSNorm(width)
        self.cross_query = nn.Linear(width, width, bias=False)
        self.cross_kv = nn.Linear(width, 2 * width, bias=False)
        self.cross_project = nn.Linear(width, width)
        self.norm_ffn = RMSNorm(width)
        self.up = nn.Linear(width, 2 * ffn)
        self.down = nn.Linear(ffn, width)

    def forward(self, actions, board, legal_mask):
        batch, count, width = actions.shape
        normalized = self.norm_self(actions)
        qkv = self.self_qkv(normalized).view(
            batch, count, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(2)
        allowed_keys = legal_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=allowed_keys,
            dropout_p=self.dropout if self.training else 0.0,
        )
        actions = actions + self.self_project(
            attended.transpose(1, 2).reshape(batch, count, width)
        )

        query = self.cross_query(self.norm_cross(actions)).view(
            batch, count, self.heads, self.head_width
        )
        key, value = self.cross_kv(board).view(
            batch, 64, 2, self.heads, self.head_width
        ).unbind(2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
        )
        actions = actions + self.cross_project(
            attended.transpose(1, 2).reshape(batch, count, width)
        )
        hidden, gate = self.up(self.norm_ffn(actions)).chunk(2, dim=-1)
        actions = actions + self.down(F.silu(gate) * hidden)
        return actions * legal_mask[:, :, None]


class HydraOracleV5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        width = config["d_model"]
        self.piece_embedding = nn.Embedding(13, width)
        self.square_embedding = nn.Embedding(64, width)
        self.castling_embedding = nn.Embedding(16, width)
        self.ep_embedding = nn.Embedding(9, width)
        self.halfmove_embedding = nn.Embedding(16, width)
        self.gab = GeometricAttentionBias(
            width,
            config["board_layers"],
            config["board_heads"],
            config["gab_token_projection"],
            config["gab_hidden"],
            config["gab_templates"],
        )
        self.board_blocks = nn.ModuleList(
            BoardBlock(width, config["board_heads"], config["board_ffn"], config["dropout"])
            for _ in range(config["board_layers"])
        )
        self.board_norm = RMSNorm(width)

        self.action_from = nn.Embedding(64, width)
        self.action_to = nn.Embedding(64, width)
        self.action_promotion = nn.Embedding(5, width)
        self.decoder_blocks = nn.ModuleList(
            LegalDecoderBlock(
                width,
                config["decoder_heads"],
                config["decoder_ffn"],
                config["dropout"],
            )
            for _ in range(config["decoder_layers"])
        )

        history_width = config["history_width"]
        self.history_from = nn.Embedding(64, history_width)
        self.history_to = nn.Embedding(64, history_width)
        self.history_promotion = nn.Embedding(5, history_width)
        self.history_position = nn.Embedding(config["history_plies"], history_width)
        self.time_embedding = nn.Embedding(config["time_classes"], history_width)
        self.rating_weight = nn.Parameter(torch.empty(history_width))
        self.rating_bias = nn.Parameter(torch.zeros(history_width))
        self.history_project = nn.Linear(history_width, width)
        nn.init.normal_(self.rating_weight, std=0.02)

        rank = config["policy_adapter_rank"]
        self.adapter_a = nn.Parameter(torch.empty(2, rank, width))
        self.adapter_b = nn.Parameter(torch.zeros(2, width, rank))
        nn.init.normal_(self.adapter_a, std=0.02)
        self.adapter_rank = rank

        self.policy = nn.Linear(width, 1)
        self.regret = nn.Linear(width, 2)
        self.value = nn.Linear(width, 3)
        self.quantiles = nn.Linear(width, config["score_quantiles"])
        self.concept_bank = nn.Parameter(
            torch.empty(config["concept_count"], config["concept_width"])
        )
        self.concept_board = nn.Parameter(torch.empty(config["concept_width"], width))
        self.concept_policy = nn.Parameter(torch.empty(config["concept_width"], width))
        nn.init.normal_(self.concept_bank, std=0.02)
        nn.init.xavier_uniform_(self.concept_board)
        nn.init.xavier_uniform_(self.concept_policy)

    def history_context(self, batch):
        history = batch["history"]
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
        mask = positions < batch["history_len"][:, None]
        values = (values * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        rating = ((batch["rating"].float() - 100.0) / 3550.0).clamp(0.0, 1.0)
        values = values + self.time_embedding(batch["time_class"])
        values = values + rating[:, None] * self.rating_weight + self.rating_bias
        return self.history_project(values)

    @staticmethod
    def gather(values, squares):
        return torch.gather(values, 1, squares[:, :, None].expand(-1, -1, values.shape[-1]))

    def forward(self, batch):
        batch_size = batch["pieces"].shape[0]
        squares = torch.arange(64, device=batch["pieces"].device).expand(batch_size, -1)
        global_state = (
            self.castling_embedding(batch["castling"])
            + self.ep_embedding(batch["ep_file"])
            + self.halfmove_embedding(batch["halfmove_bucket"])
        )[:, None]
        board = self.piece_embedding(batch["pieces"]) + self.square_embedding(squares) + global_state
        gab_context = self.gab.context(board)
        use_checkpoint = self.training and self.config.get("activation_checkpointing", True)
        for layer, block in enumerate(self.board_blocks):
            bias = self.gab.bias(gab_context, layer)
            if use_checkpoint:
                board = checkpoint(block, board, bias, use_reentrant=False)
            else:
                board = block(board, bias)
        board = self.board_norm(board)
        pooled = board.mean(1)

        actions = batch["safe_actions"]
        source = actions & 63
        target = (actions >> 6) & 63
        promotion = (actions >> 12).clamp(0, 4)
        action_values = (
            self.action_from(source)
            + self.action_to(target)
            + self.action_promotion(promotion)
            + 0.5 * (self.gather(board, source) + self.gather(board, target))
        )
        history = self.history_context(batch)
        adapter_input = action_values + history[:, None]
        a = self.adapter_a[batch["policy_kind"]]
        b = self.adapter_b[batch["policy_kind"]]
        low = torch.einsum("bld,brd->blr", adapter_input, a)
        action_values = action_values + torch.einsum("blr,bdr->bld", low, b) / self.adapter_rank
        action_values = action_values * batch["legal_mask"][:, :, None]
        for block in self.decoder_blocks:
            if use_checkpoint:
                action_values = checkpoint(
                    block,
                    action_values,
                    board,
                    batch["legal_mask"],
                    use_reentrant=False,
                )
            else:
                action_values = block(action_values, board, batch["legal_mask"])

        logits = self.policy(action_values).squeeze(-1).masked_fill(~batch["legal_mask"], -1e4)
        regret_raw = self.regret(action_values)
        regret_mean = F.softplus(regret_raw[:, :, 0])
        regret_log_scale = regret_raw[:, :, 1].clamp(-8.0, 4.0)
        evidence = F.softplus(self.value(pooled))
        quantiles = self.quantiles(pooled).sort(1).values
        board_semantic = F.linear(pooled, self.concept_board)
        legal_weights = batch["legal_mask"].to(action_values.dtype)
        action_pooled = (action_values * legal_weights[:, :, None]).sum(1) / legal_weights.sum(1, keepdim=True).clamp_min(1)
        policy_semantic = F.linear(action_pooled, self.concept_policy)
        board_concepts = F.softmax(F.linear(board_semantic, self.concept_bank), -1)
        policy_concepts = F.softmax(F.linear(policy_semantic, self.concept_bank), -1)
        return {
            "logits": logits,
            "regret_mean": regret_mean,
            "regret_log_scale": regret_log_scale,
            "evidence": evidence,
            "quantiles": quantiles,
            "representation": pooled,
            "board_concepts": board_concepts,
            "policy_concepts": policy_concepts,
        }


def prepare_oracle_batch(records, device, augment=False):
    batch = prepare_batch(records, device, augment)
    batch["teacher_score"] = torch.from_numpy(
        np.ascontiguousarray(records["teacher_score"]).astype(np.float32)
    ).to(device, non_blocking=device.type == "cuda")
    return batch


def oracle_loss(output, batch, config, global_step):
    human_rows = batch["policy_kind"] == POLICY_HUMAN
    guide_rows = (batch["policy_kind"] == POLICY_GUIDE) & batch["teacher_mask"].any(1)
    hard_ce = F.cross_entropy(
        output["logits"].float(),
        batch["target_index"],
        reduction="none",
        label_smoothing=0.01,
    )
    human_policy = masked_mean(hard_ce, human_rows)

    target_regret = batch["regret_target"]
    target_policy = F.softmax(
        (-target_regret / (config["teacher_temperature_cp"] / 400.0)).masked_fill(
            ~batch["teacher_mask"], -1e4
        ),
        1,
    )
    guide_ce = -(target_policy * F.log_softmax(output["logits"].float(), 1)).sum(1)
    guide_policy = masked_mean(guide_ce, guide_rows)

    difference = target_regret - output["regret_mean"]
    regret_nll = 0.5 * (
        difference.square() * torch.exp(-2.0 * output["regret_log_scale"])
        + 2.0 * output["regret_log_scale"]
    )
    legal_regret = masked_mean(regret_nll, batch["teacher_mask"])

    anneal = min(1.0, global_step / max(1, config["evidence_kl_anneal_steps"]))
    evidential = evidential_loss(output["evidence"], batch["wdl"], anneal)

    labelled_rows = batch["teacher_mask"].any(1)
    score_target = batch["teacher_score"] / 400.0
    quantile_levels = torch.linspace(
        0.1, 0.9, output["quantiles"].shape[1], device=score_target.device
    )
    quantile_error = score_target[:, None] - output["quantiles"]
    pinball = torch.maximum(
        quantile_levels * quantile_error,
        (quantile_levels - 1.0) * quantile_error,
    ).mean(1)
    score_quantiles = masked_mean(pinball, labelled_rows)

    board_concepts = output["board_concepts"]
    policy_concepts = output["policy_concepts"]
    concept_alignment = F.kl_div(
        policy_concepts.clamp_min(1e-8).log(),
        board_concepts.detach(),
        reduction="batchmean",
    )
    entropy = -(board_concepts * board_concepts.clamp_min(1e-8).log()).sum(1).mean()
    concept_regularization = concept_alignment - 0.002 * entropy

    alpha = output["evidence"] + 1.0
    wdl_probability = alpha / alpha.sum(1, keepdim=True)
    expected = wdl_probability[:, 2] - wdl_probability[:, 0]
    median = output["quantiles"][:, output["quantiles"].shape[1] // 2]
    wdl_score_consistency = F.smooth_l1_loss(expected, torch.tanh(median / 4.0))

    active_error = difference.abs()[batch["teacher_mask"]]
    active_scale = torch.exp(output["regret_log_scale"])[batch["teacher_mask"]]
    if active_error.numel() > 1:
        error_delta = active_error - active_error.roll(1)
        scale_delta = active_scale - active_scale.roll(1)
        calibration_ranking = F.relu(-error_delta * scale_delta).mean()
    else:
        calibration_ranking = active_error.sum() * 0.0

    weights = config["loss_weights"]
    losses = {
        "human_policy": human_policy,
        "guide_policy": guide_policy,
        "evidential_wdl": evidential,
        "legal_regret": legal_regret,
        "score_quantiles": score_quantiles,
        "concept_regularization": concept_regularization,
        "wdl_score_consistency": wdl_score_consistency,
        "calibration_ranking": calibration_ranking,
    }
    total = sum(weights[name] * value for name, value in losses.items())
    return total, {name: value.detach() for name, value in losses.items()}


class EpochRecordPrefetcher:
    """Disjoint block-shuffled epoch with contiguous NVMe reads per rank."""

    def __init__(
        self,
        shards,
        seed,
        epoch,
        batch_size,
        rank,
        world_size,
        depth,
        workers=4,
    ):
        self.shards = shards
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.global_batch = batch_size * world_size
        self.batches = shards.total // self.global_batch
        rng = np.random.default_rng(seed + epoch * 1_000_003)
        # Shuffle global contiguous batches, not every record. This keeps each
        # rank's mmap gather sequential while using O(records/global_batch)
        # index memory instead of O(records) per DDP process.
        self.batch_order = rng.permutation(self.batches)
        self.record_rotation = int(rng.integers(0, shards.total)) if shards.total else 0
        self.cursor = 0
        self.executor = ThreadPoolExecutor(max_workers=max(1, min(workers, depth)))
        self.futures = deque()
        for _ in range(min(max(1, depth), self.batches)):
            self._submit_next()

    def _submit_next(self):
        if self.cursor < self.batches:
            global_batch_index = int(self.batch_order[self.cursor])
            start = global_batch_index * self.global_batch + self.rank * self.batch_size
            indices = (
                np.arange(start, start + self.batch_size, dtype=np.int64)
                + self.record_rotation
            ) % self.shards.total
            self.cursor += 1
            self.futures.append(self.executor.submit(self.shards.gather, indices))

    def next(self):
        if not self.futures:
            raise StopIteration("epoch record stream is exhausted")
        records = self.futures.popleft().result()
        self._submit_next()
        return records

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()
        self.batch_order = np.empty(0, dtype=np.int64)


def require_distinct_shards(train_paths, validation_paths):
    train = {str(Path(path).resolve()) for path in train_paths}
    validation = {str(Path(path).resolve()) for path in validation_paths}
    overlap = train & validation
    if overlap:
        raise ValueError(
            "training and validation arguments share shard paths: "
            + ", ".join(sorted(overlap))
        )


def optimizer_steps_for_epoch(total_records, effective_global_batch, minimum_steps):
    if effective_global_batch <= 0:
        raise ValueError("effective global batch must be positive")
    steps = total_records // effective_global_batch
    if steps < minimum_steps:
        required = effective_global_batch * minimum_steps
        raise ValueError(
            f"dataset has {total_records:,} records, but at least {required:,} are "
            f"required for {minimum_steps} without-replacement optimizer steps"
        )
    return steps


def reserved_memory_growth_exceeded(current, baseline, tolerance):
    if not baseline or not current:
        return False
    allowance = max(512 * 2**20, int(baseline * tolerance))
    return current > baseline + allowance


def require_finite_metrics(metrics):
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"validation metric {name} is non-finite: {value}")


def make_optimizer(model, learning_rate_value, weight_decay, device):
    kwargs = dict(
        lr=learning_rate_value,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def cuda_memory_snapshot(device):
    if device.type != "cuda":
        return {
            "device": str(device),
            "allocated": 0,
            "reserved": 0,
            "peak_allocated": 0,
            "peak_reserved": 0,
            "total": 0,
            "fraction": 0.0,
        }
    total = torch.cuda.get_device_properties(device).total_memory
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "device": torch.cuda.get_device_name(device),
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated": peak,
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "total": total,
        "fraction": peak / total,
    }


def probe_microbatch(config, hardware, records, device, precision):
    if device.type != "cuda":
        return min(len(records), config["micro_batch_initial"]), cuda_memory_snapshot(device)
    multiple = config["micro_batch_multiple"]
    low = max(1, config["micro_batch_min"] // multiple)
    high = min(len(records), config["micro_batch_max"]) // multiple
    target = hardware["target_vram_fraction"]
    best = None
    best_memory = None
    while low <= high:
        units = (low + high) // 2
        candidate = units * multiple
        model = optimizer = scaler = batch = output = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = HydraOracleV5(config).to(device)
            optimizer = make_optimizer(
                model, config["learning_rate"], config["weight_decay"], device
            )
            batch = prepare_oracle_batch(records[:candidate], device)
            scaler = make_grad_scaler(precision["uses_scaler"])
            with autocast_context(device, precision):
                output = model(batch)
                loss, _ = oracle_loss(output, batch, config, 1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            memory = cuda_memory_snapshot(device)
            if memory["fraction"] <= target:
                best, best_memory = candidate, memory
                low = units + 1
            else:
                high = units - 1
        except torch.OutOfMemoryError:
            high = units - 1
        finally:
            del model, optimizer, scaler, batch, output, loss
            torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("minimum microbatch exceeds the configured A100 VRAM target")
    return best, best_memory


def probe_distillation_microbatch(
    oracle, student_config, distill_config, hardware, records, device, precision
):
    if device.type != "cuda":
        return min(len(records), distill_config["micro_batch_initial"]), cuda_memory_snapshot(device)
    multiple = 16
    low = 1
    high = min(len(records), distill_config["micro_batch_max"]) // multiple
    target = hardware["target_vram_fraction"]
    best = None
    best_memory = None
    while low <= high:
        units = (low + high) // 2
        candidate = units * multiple
        student = optimizer = scaler = batch = oracle_output = outputs = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            student = AegisV4Chessformer(student_config).to(device)
            optimizer = make_optimizer(
                student,
                distill_config["learning_rate"],
                student_config["weight_decay"],
                device,
            )
            batch = prepare_oracle_batch(records[:candidate], device)
            scaler = make_grad_scaler(precision["uses_scaler"])
            with torch.no_grad(), autocast_context(device, precision):
                oracle_output = oracle(batch)
            with autocast_context(device, precision):
                outputs = student(batch)
                loss, _ = student_distillation_loss(
                    oracle_output, outputs, batch, distill_config
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            memory = cuda_memory_snapshot(device)
            if memory["fraction"] <= target:
                best, best_memory = candidate, memory
                low = units + 1
            else:
                high = units - 1
        except torch.OutOfMemoryError:
            high = units - 1
        finally:
            del student, optimizer, scaler, batch, oracle_output, outputs, loss
            torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("distillation microbatch cannot fit below target VRAM fraction")
    return best, best_memory


@torch.no_grad()
def evaluate_oracle(
    model,
    shards,
    device,
    config,
    maximum,
    distributed_context=None,
    heartbeat_path=None,
):
    model.eval()
    total = human_total = guide_total = human_hits = guide_hits = wdl_hits = 0
    regret_error = labelled = nll = 0.0
    world_size = distributed_context.world_size if distributed_context else 1
    rank = distributed_context.rank if distributed_context else 0
    batches = shards.sequential_batches(config.get("validation_batch_size", 256), maximum)
    for batch_index, records in enumerate(batches):
        if batch_index % world_size != rank:
            continue
        batch = prepare_oracle_batch(records, device)
        output = model(batch)
        prediction = output["logits"].argmax(1)
        human = batch["policy_kind"] == POLICY_HUMAN
        guide = (batch["policy_kind"] == POLICY_GUIDE) & batch["teacher_mask"].any(1)
        human_hits += int(((prediction == batch["target_index"]) & human).sum())
        guide_hits += int(((prediction == batch["teacher_best_index"]) & guide).sum())
        human_total += int(human.sum())
        guide_total += int(guide.sum())
        wdl_hits += int((output["evidence"].argmax(1) == batch["wdl"]).sum())
        nll += float(F.cross_entropy(output["logits"].float(), batch["target_index"], reduction="sum"))
        mask = batch["teacher_mask"]
        regret_error += float((output["regret_mean"] - batch["regret_target"]).abs()[mask].sum()) * 400.0
        labelled += int(mask.sum())
        total += len(records)
        if (
            heartbeat_path
            and (distributed_context is None or distributed_context.primary)
            and batch_index % 25 == 0
        ):
            write_heartbeat(
                heartbeat_path,
                "oracle_validation",
                {"validation_batches": batch_index + 1, "validation_records_rank": total},
            )
    values = [
        total,
        human_total,
        guide_total,
        human_hits,
        guide_hits,
        wdl_hits,
        regret_error,
        labelled,
        nll,
    ]
    if distributed_context:
        values = distributed_context.sum_floats(values)
    (
        total,
        human_total,
        guide_total,
        human_hits,
        guide_hits,
        wdl_hits,
        regret_error,
        labelled,
        nll,
    ) = values
    return {
        "records": int(total),
        "nll": nll / max(1, total),
        "human_top1": human_hits / max(1, human_total),
        "guide_best_top1": guide_hits / max(1, guide_total),
        "wdl_top1": wdl_hits / max(1, total),
        "regret_mae_cp": regret_error / max(1, labelled),
        "labelled_actions": int(labelled),
    }


@torch.no_grad()
def evaluate_student_distributed(
    model,
    shards,
    device,
    config,
    maximum,
    distributed_context,
    heartbeat_path=None,
):
    model.eval()
    exits = len(config["exit_layers"])
    human_hits = [0.0] * exits
    guide_hits = [0.0] * exits
    total = human_total = guide_total = wdl_hits = labelled = 0.0
    nll = regret_error = 0.0
    batches = shards.sequential_batches(config.get("validation_batch_size", 256), maximum)
    for batch_index, records in enumerate(batches):
        if batch_index % distributed_context.world_size != distributed_context.rank:
            continue
        batch = prepare_oracle_batch(records, device)
        outputs = model(batch)
        human = batch["policy_kind"] == POLICY_HUMAN
        guide = (batch["policy_kind"] == POLICY_GUIDE) & batch["teacher_mask"].any(1)
        human_total += int(human.sum())
        guide_total += int(guide.sum())
        for index, output in enumerate(outputs):
            prediction = output["logits"].argmax(1)
            human_hits[index] += int(((prediction == batch["target_index"]) & human).sum())
            guide_hits[index] += int(
                ((prediction == batch["teacher_best_index"]) & guide).sum()
            )
        full = outputs[-1]
        nll += float(
            F.cross_entropy(
                full["logits"].float(), batch["target_index"], reduction="sum"
            )
        )
        wdl_hits += int((full["evidence"].argmax(1) == batch["wdl"]).sum())
        mask = batch["teacher_mask"]
        regret_error += (
            float((full["regret_mean"] - batch["regret_target"]).abs()[mask].sum())
            * 400.0
        )
        labelled += int(mask.sum())
        total += len(records)
        if heartbeat_path and distributed_context.primary and batch_index % 25 == 0:
            write_heartbeat(
                heartbeat_path,
                "student_validation",
                {"validation_batches": batch_index + 1, "validation_records_rank": total},
            )
    reduced = distributed_context.sum_floats(
        [total, human_total, guide_total, wdl_hits, labelled, nll, regret_error]
        + human_hits
        + guide_hits
    )
    total, human_total, guide_total, wdl_hits, labelled, nll, regret_error = reduced[:7]
    human_hits = reduced[7 : 7 + exits]
    guide_hits = reduced[7 + exits :]
    return {
        "records": int(total),
        "nll": nll / max(1, total),
        "exit_human_top1": [value / max(1, human_total) for value in human_hits],
        "exit_guide_best_top1": [value / max(1, guide_total) for value in guide_hits],
        "wdl_top1": wdl_hits / max(1, total),
        "regret_mae_cp": regret_error / max(1, labelled),
        "labelled_actions": int(labelled),
    }


def train_oracle(args):
    config, hardware = load_config(args.config, "oracle")
    root = json.loads(Path(args.config).read_text())
    hardware = root["hardware"]
    safety = TrainingSafetyController.load(root["safety_config"])
    heartbeat_path = args.heartbeat or Path(str(args.output) + ".heartbeat.json")
    incident_path = args.incident or Path(str(args.output) + ".incident.json")
    if abs(sum(config["loss_weights"].values()) - 1.0) > 1e-12:
        raise ValueError("oracle loss weights must sum to one")
    if config.get("epoch_mode") != "without_replacement":
        raise ValueError("Apex v5 requires without-replacement epoch semantics")
    require_distinct_shards(args.train, args.validation)
    configure_compiler_safety(hardware)
    distributed_context = DistributedContext(config["seed"], args.deterministic)
    device = distributed_context.device
    precision = resolve_precision(device, hardware)
    train_data = V4RecordShards(args.train)
    validation_data = V4RecordShards(args.validation)
    minimum_records = (
        config["effective_batch_records"]
        * config["minimum_optimizer_steps_per_epoch"]
    )
    if train_data.total < minimum_records:
        raise ValueError(
            f"training set has {train_data.total:,} records; Apex requires at least "
            f"{minimum_records:,} before allocating the full Oracle"
        )
    if validation_data.total < config["minimum_validation_records"]:
        raise ValueError(
            f"validation set has {validation_data.total:,} records; at least "
            f"{config['minimum_validation_records']:,} are required"
        )
    rng = np.random.default_rng(config["seed"] + distributed_context.rank * 1_000_003)
    microbatch = args.micro_batch or config["micro_batch_initial"]
    probe_result = None
    if args.auto_batch:
        probe_records = train_data.sample(rng, config["micro_batch_max"])
        microbatch, probe_result = probe_microbatch(
            config, hardware, probe_records, device, precision
        )
    microbatch = distributed_context.minimum_int(microbatch)
    accumulation = math.ceil(
        config["effective_batch_records"]
        / (microbatch * distributed_context.world_size)
    )
    effective_batch = microbatch * accumulation * distributed_context.world_size
    optimizer_steps_per_epoch = optimizer_steps_for_epoch(
        train_data.total,
        effective_batch,
        config["minimum_optimizer_steps_per_epoch"],
    )
    records_per_epoch = optimizer_steps_per_epoch * effective_batch

    raw_model = HydraOracleV5(config).to(device)
    parameter_count = model_parameter_count(raw_model)
    if config.get("expected_parameters") and parameter_count != config["expected_parameters"]:
        raise RuntimeError(
            f"oracle parameter drift: {parameter_count:,} != {config['expected_parameters']:,}"
        )
    optimizer = make_optimizer(
        raw_model, config["learning_rate"], config["weight_decay"], device
    )
    global_step = start_epoch = 0
    best_nll = float("inf")
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        global_step = int(checkpoint_data.get("global_step", 0))
        start_epoch = int(checkpoint_data.get("epoch", 0))
        best_nll = float(checkpoint_data.get("metrics", {}).get("nll", best_nll))
    train_model = raw_model
    if hardware.get("compile", True) and device.type == "cuda" and not args.no_compile:
        train_model = torch.compile(
            raw_model, mode=hardware.get("compile_mode", "default")
        )
    if distributed_context.distributed:
        train_model = DistributedDataParallel(
            train_model,
            device_ids=[distributed_context.local_rank],
            output_device=distributed_context.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    scaler = make_grad_scaler(precision["uses_scaler"])
    total_steps = config["epochs"] * optimizer_steps_per_epoch
    if distributed_context.primary:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "precision": precision["name"],
                    "world_size": distributed_context.world_size,
                    "parameters": model_parameter_count(raw_model),
                    "microbatch_per_gpu": microbatch,
                    "accumulation": accumulation,
                    "effective_global_batch": effective_batch,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "records_consumed_once_per_epoch": records_per_epoch,
                    "dropped_tail_records": train_data.total - records_per_epoch,
                    "sampling": "global_without_replacement",
                    "probe": probe_result,
                    "train_records": train_data.total,
                    "validation_records": validation_data.total,
                },
                indent=2,
            ),
            flush=True,
        )
    epochs_without_improvement = 0
    reserved_memory_baseline = 0
    try:
        for epoch in range(start_epoch, config["epochs"]):
            prefetcher = EpochRecordPrefetcher(
                train_data,
                config["seed"],
                epoch,
                microbatch,
                distributed_context.rank,
                distributed_context.world_size,
                hardware.get("prefetch_batches", 4),
                hardware.get("prefetch_workers", 4),
            )
            expected_batches = optimizer_steps_per_epoch * accumulation
            if prefetcher.batches < expected_batches:
                prefetcher.close()
                raise RuntimeError("without-replacement epoch produced too few rank batches")
            train_model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            epoch_loss = 0.0
            try:
                for _ in range(optimizer_steps_per_epoch):
                    lr = learning_rate(
                        global_step,
                        total_steps,
                        config["learning_rate"],
                        config["warmup_steps"],
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for accumulation_index in range(accumulation):
                        records = prefetcher.next()
                        batch = prepare_oracle_batch(records, device, augment=True)
                        synchronize = accumulation_index == accumulation - 1
                        sync_context = (
                            train_model.no_sync()
                            if distributed_context.distributed and not synchronize
                            else contextlib.nullcontext()
                        )
                        with sync_context:
                            with autocast_context(device, precision):
                                output = train_model(batch)
                                loss, _ = oracle_loss(output, batch, config, global_step)
                                scaled_loss = loss / accumulation
                            scaler.scale(scaled_loss).backward()
                        step_loss += float(loss.detach())
                    scaler.unscale_(optimizer)
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            raw_model.parameters(), config["gradient_clip"]
                        )
                    )
                    mean_step_loss = step_loss / accumulation
                    safety_decision = safety.check_step(mean_step_loss, gradient_norm)
                    unsafe = distributed_context.any_bool(not safety_decision.safe)
                    if unsafe:
                        if distributed_context.primary:
                            atomic_json(
                                incident_path,
                                {
                                    "schema": 1,
                                    "phase": "oracle",
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "decision": safety.snapshot(),
                                    "reason": safety_decision.reason
                                    or "another DDP rank reported unsafe state",
                                },
                            )
                        raise RuntimeError(
                            "autonomous safety abort: "
                            + (
                                safety_decision.reason
                                or "another DDP rank reported unsafe state"
                            )
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += mean_step_loss
                    global_step += 1
                    if (
                        distributed_context.primary
                        and global_step
                        % safety.config["heartbeat_interval_steps"]
                        == 0
                    ):
                        write_heartbeat(
                            heartbeat_path,
                            "oracle",
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "loss": mean_step_loss,
                                "gradient_norm": gradient_norm,
                                "learning_rate": lr,
                                "safety": safety.snapshot(),
                                "memory": cuda_memory_snapshot(device),
                            },
                        )
            finally:
                prefetcher.close()
            metrics = evaluate_oracle(
                raw_model,
                validation_data,
                device,
                config,
                args.validation_records,
                distributed_context,
                heartbeat_path,
            )
            require_finite_metrics(metrics)
            validation_decision = safety.check_validation(metrics["nll"])
            memory = cuda_memory_snapshot(device)
            elapsed = time.monotonic() - started
            mean_epoch_loss = distributed_context.sum_floats([epoch_loss])[0] / distributed_context.world_size
            improved = metrics["nll"] < best_nll - config["early_stopping_min_delta"]
            if improved:
                best_nll = metrics["nll"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            current_reserved = memory.get("reserved", 0)
            if epoch <= start_epoch + 1:
                reserved_memory_baseline = max(reserved_memory_baseline, current_reserved)
            memory_growth_abort = distributed_context.any_bool(
                epoch > start_epoch + 1
                and reserved_memory_growth_exceeded(
                    current_reserved,
                    reserved_memory_baseline,
                    config["reserved_memory_growth_tolerance"],
                )
            )
            early_stop = (
                epochs_without_improvement >= config["early_stopping_patience"]
                or validation_decision.action == "early_stop"
            )
            if distributed_context.primary:
                print(
                    f"epoch={epoch + 1} loss={mean_epoch_loss / optimizer_steps_per_epoch:.5f} "
                    f"nll={metrics['nll']:.5f} human={metrics['human_top1']:.4f} "
                    f"guide={metrics['guide_best_top1']:.4f} wdl={metrics['wdl_top1']:.4f} "
                    f"regret={metrics['regret_mae_cp']:.1f}cp vram={memory['fraction']:.3f} "
                    f"records_per_second={records_per_epoch / elapsed:.0f}",
                    flush=True,
                )
                payload = {
                    "format": "UNARCHV1_ORACLE_TRAINING_V1_DDP",
                    "config": config,
                    "hardware": hardware,
                    "precision": precision["name"],
                    "world_size": distributed_context.world_size,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "microbatch_per_gpu": microbatch,
                    "accumulation": accumulation,
                    "effective_global_batch": effective_batch,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "records_per_epoch": records_per_epoch,
                    "sampling": "global_without_replacement",
                    "epochs_without_improvement": epochs_without_improvement,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "memory_rank0": memory,
                    "train_manifest": train_data.manifest(),
                    "validation_manifest": validation_data.manifest(),
                }
                atomic_torch_save(payload, args.output)
                if improved:
                    atomic_torch_save(payload, str(args.output) + ".best")
                if early_stop:
                    print(
                        f"early stop: validation NLL failed to improve for "
                        f"{epochs_without_improvement} epochs",
                        flush=True,
                    )
                if memory_growth_abort:
                    print(
                        "abort: CUDA reserved memory grew beyond the configured stable envelope",
                        flush=True,
                    )
            distributed_context.barrier()
            if memory_growth_abort:
                raise RuntimeError("CUDA reserved-memory growth guard triggered")
            if early_stop:
                break
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        distributed_context.close()


def student_distillation_loss(oracle, student_outputs, batch, config):
    temperature = config["policy_temperature"]
    oracle_policy = F.softmax(oracle["logits"].detach().float() / temperature, 1)
    policy_losses = []
    for output in student_outputs:
        policy_losses.append(
            -(oracle_policy * F.log_softmax(output["logits"].float() / temperature, 1)).sum(1).mean()
            * temperature
            * temperature
        )
    oracle_policy_loss = torch.stack(policy_losses).mean()

    wdl_temperature = config["wdl_temperature"]
    oracle_alpha = oracle["evidence"].detach().float() + 1.0
    oracle_probability = oracle_alpha / oracle_alpha.sum(1, keepdim=True)
    oracle_wdl = F.softmax(
        oracle_probability.clamp_min(1e-8).log() / wdl_temperature, dim=1
    )
    student_wdl_losses = []
    for output in student_outputs:
        alpha = output["evidence"].float() + 1.0
        probability = alpha / alpha.sum(1, keepdim=True)
        softened = F.log_softmax(
            probability.clamp_min(1e-8).log() / wdl_temperature, dim=1
        )
        student_wdl_losses.append(
            F.kl_div(softened, oracle_wdl, reduction="batchmean")
            * wdl_temperature
            * wdl_temperature
        )
    oracle_wdl_loss = torch.stack(student_wdl_losses).mean()

    student_full = student_outputs[-1]
    oracle_regret_loss = masked_mean(
        F.smooth_l1_loss(
            student_full["regret_mean"], oracle["regret_mean"].detach(), reduction="none"
        ),
        batch["legal_mask"],
    )
    hard_ce = F.cross_entropy(student_full["logits"].float(), batch["target_index"], reduction="none")
    human = batch["policy_kind"] == POLICY_HUMAN
    human_ground_truth = masked_mean(hard_ce, human)
    guide = (batch["policy_kind"] == POLICY_GUIDE) & batch["teacher_mask"].any(1)
    guide_ground_truth = masked_mean(
        F.cross_entropy(
            student_full["logits"].float(), batch["teacher_best_index"], reduction="none"
        ),
        guide,
    )
    full_policy = F.softmax(student_full["logits"].detach().float(), 1)
    exit_consistency = torch.stack(
        [
            -(full_policy * F.log_softmax(output["logits"].float(), 1)).sum(1).mean()
            for output in student_outputs[:-1]
        ]
    ).mean()
    oracle_entropy = -(
        oracle["board_concepts"] * oracle["board_concepts"].clamp_min(1e-8).log()
    ).sum(1)
    student_entropy = -(
        student_full["board_concepts"]
        * student_full["board_concepts"].clamp_min(1e-8).log()
    ).sum(1)
    concept_statistics = F.smooth_l1_loss(
        student_entropy / math.log(student_full["board_concepts"].shape[1]),
        oracle_entropy.detach() / math.log(oracle["board_concepts"].shape[1]),
    )
    losses = {
        "oracle_policy": oracle_policy_loss,
        "oracle_wdl": oracle_wdl_loss,
        "oracle_regret": oracle_regret_loss,
        "human_ground_truth": human_ground_truth,
        "guide_ground_truth": guide_ground_truth,
        "exit_consistency": exit_consistency,
        "concept_alignment": concept_statistics,
    }
    weights = config["loss_weights"]
    total = sum(weights[name] * value for name, value in losses.items())
    return total, {name: value.detach() for name, value in losses.items()}


def distill_student(args):
    root = json.loads(Path(args.config).read_text())
    safety = TrainingSafetyController.load(root["safety_config"])
    heartbeat_path = args.heartbeat or Path(str(args.output) + ".heartbeat.json")
    incident_path = args.incident or Path(str(args.output) + ".incident.json")
    oracle_config = root["oracle"]
    distill_config = root["student_distillation"]
    if distill_config.get("epoch_mode") != "without_replacement":
        raise ValueError("Apex v5 distillation requires without-replacement epochs")
    require_distinct_shards(args.train, args.validation)
    configure_compiler_safety(root["hardware"])
    student_config, hardware = load_config(
        args.student_config or distill_config["student_config"], "chessformer"
    )
    distributed_context = DistributedContext(
        oracle_config["seed"] + 17, args.deterministic
    )
    device = distributed_context.device
    precision = resolve_precision(device, root["hardware"])
    train_data = V4RecordShards(args.train)
    validation_data = V4RecordShards(args.validation)
    minimum_records = (
        distill_config["effective_batch_records"]
        * distill_config["minimum_optimizer_steps_per_epoch"]
    )
    if train_data.total < minimum_records:
        raise ValueError(
            f"distillation set has {train_data.total:,} records; at least "
            f"{minimum_records:,} are required"
        )
    if validation_data.total < distill_config["minimum_validation_records"]:
        raise ValueError(
            f"distillation validation set has {validation_data.total:,} records; at least "
            f"{distill_config['minimum_validation_records']:,} are required"
        )
    oracle_checkpoint = torch.load(args.oracle, map_location=device, weights_only=False)
    oracle = HydraOracleV5(oracle_checkpoint["config"]).to(device)
    oracle.load_state_dict(oracle_checkpoint["model"])
    oracle.eval()
    oracle.requires_grad_(False)
    rng = np.random.default_rng(
        oracle_config["seed"] + 1 + distributed_context.rank * 1_000_003
    )
    microbatch = args.micro_batch or distill_config["micro_batch_initial"]
    probe_result = None
    if args.auto_batch:
        probe_records = train_data.sample(rng, distill_config["micro_batch_max"])
        microbatch, probe_result = probe_distillation_microbatch(
            oracle,
            student_config,
            distill_config,
            root["hardware"],
            probe_records,
            device,
            precision,
        )
    microbatch = distributed_context.minimum_int(microbatch)
    student = AegisV4Chessformer(student_config).to(device)
    optimizer = make_optimizer(
        student,
        distill_config["learning_rate"],
        student_config["weight_decay"],
        device,
    )
    accumulation = math.ceil(
        distill_config["effective_batch_records"]
        / (microbatch * distributed_context.world_size)
    )
    effective_batch = microbatch * accumulation * distributed_context.world_size
    optimizer_steps_per_epoch = optimizer_steps_for_epoch(
        train_data.total,
        effective_batch,
        distill_config["minimum_optimizer_steps_per_epoch"],
    )
    records_per_epoch = optimizer_steps_per_epoch * effective_batch
    train_student = student
    if root["hardware"].get("compile", True) and device.type == "cuda" and not args.no_compile:
        train_student = torch.compile(
            student, mode=root["hardware"].get("compile_mode", "default")
        )
    if distributed_context.distributed:
        train_student = DistributedDataParallel(
            train_student,
            device_ids=[distributed_context.local_rank],
            output_device=distributed_context.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    scaler = make_grad_scaler(precision["uses_scaler"])
    total_steps = distill_config["epochs"] * optimizer_steps_per_epoch
    global_step = 0
    best_nll = float("inf")
    if distributed_context.primary:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "precision": precision["name"],
                    "world_size": distributed_context.world_size,
                    "oracle_parameters": model_parameter_count(oracle),
                    "student_parameters": model_parameter_count(student),
                    "microbatch_per_gpu": microbatch,
                    "accumulation": accumulation,
                    "effective_global_batch": effective_batch,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "records_consumed_once_per_epoch": records_per_epoch,
                    "dropped_tail_records": train_data.total - records_per_epoch,
                    "sampling": "global_without_replacement",
                    "probe": probe_result,
                },
                indent=2,
            ),
            flush=True,
        )
    epochs_without_improvement = 0
    reserved_memory_baseline = 0
    try:
        for epoch in range(distill_config["epochs"]):
            prefetcher = EpochRecordPrefetcher(
                train_data,
                oracle_config["seed"] + 1,
                epoch,
                microbatch,
                distributed_context.rank,
                distributed_context.world_size,
                root["hardware"].get("prefetch_batches", 4),
                root["hardware"].get("prefetch_workers", 4),
            )
            expected_batches = optimizer_steps_per_epoch * accumulation
            if prefetcher.batches < expected_batches:
                prefetcher.close()
                raise RuntimeError("without-replacement distillation epoch is too short")
            train_student.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_loss = 0.0
            started = time.monotonic()
            try:
                for _ in range(optimizer_steps_per_epoch):
                    lr = learning_rate(
                        global_step,
                        total_steps,
                        distill_config["learning_rate"],
                        distill_config["warmup_steps"],
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for accumulation_index in range(accumulation):
                        records = prefetcher.next()
                        batch = prepare_oracle_batch(records, device, augment=True)
                        with torch.no_grad(), autocast_context(device, precision):
                            oracle_output = oracle(batch)
                        synchronize = accumulation_index == accumulation - 1
                        sync_context = (
                            train_student.no_sync()
                            if distributed_context.distributed and not synchronize
                            else contextlib.nullcontext()
                        )
                        with sync_context:
                            with autocast_context(device, precision):
                                student_outputs = train_student(batch)
                                loss, _ = student_distillation_loss(
                                    oracle_output, student_outputs, batch, distill_config
                                )
                            scaler.scale(loss / accumulation).backward()
                        step_loss += float(loss.detach())
                    scaler.unscale_(optimizer)
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(), student_config["gradient_clip"]
                        )
                    )
                    mean_step_loss = step_loss / accumulation
                    safety_decision = safety.check_step(mean_step_loss, gradient_norm)
                    unsafe = distributed_context.any_bool(not safety_decision.safe)
                    if unsafe:
                        if distributed_context.primary:
                            atomic_json(
                                incident_path,
                                {
                                    "schema": 1,
                                    "phase": "student_distillation",
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "decision": safety.snapshot(),
                                    "reason": safety_decision.reason
                                    or "another DDP rank reported unsafe state",
                                },
                            )
                        raise RuntimeError(
                            "autonomous safety abort: "
                            + (
                                safety_decision.reason
                                or "another DDP rank reported unsafe state"
                            )
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += mean_step_loss
                    global_step += 1
                    if (
                        distributed_context.primary
                        and global_step
                        % safety.config["heartbeat_interval_steps"]
                        == 0
                    ):
                        write_heartbeat(
                            heartbeat_path,
                            "student_distillation",
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "loss": mean_step_loss,
                                "gradient_norm": gradient_norm,
                                "learning_rate": lr,
                                "safety": safety.snapshot(),
                                "memory": cuda_memory_snapshot(device),
                            },
                        )
            finally:
                prefetcher.close()
            metrics = evaluate_student_distributed(
                student,
                validation_data,
                device,
                student_config,
                args.validation_records,
                distributed_context,
                heartbeat_path,
            )
            require_finite_metrics(metrics)
            validation_decision = safety.check_validation(metrics["nll"])
            memory = cuda_memory_snapshot(device)
            elapsed = time.monotonic() - started
            mean_epoch_loss = distributed_context.sum_floats([epoch_loss])[0] / distributed_context.world_size
            improved = (
                metrics["nll"]
                < best_nll - distill_config["early_stopping_min_delta"]
            )
            if improved:
                best_nll = metrics["nll"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            current_reserved = memory.get("reserved", 0)
            if epoch <= 1:
                reserved_memory_baseline = max(reserved_memory_baseline, current_reserved)
            memory_growth_abort = distributed_context.any_bool(
                epoch > 1
                and reserved_memory_growth_exceeded(
                    current_reserved,
                    reserved_memory_baseline,
                    distill_config["reserved_memory_growth_tolerance"],
                )
            )
            early_stop = (
                epochs_without_improvement
                >= distill_config["early_stopping_patience"]
                or validation_decision.action == "early_stop"
            )
            if distributed_context.primary:
                print(
                    f"epoch={epoch + 1} distill_loss={mean_epoch_loss / optimizer_steps_per_epoch:.5f} "
                    f"nll={metrics['nll']:.5f} vram={memory['fraction']:.3f} records_per_second="
                    f"{records_per_epoch / elapsed:.0f}",
                    flush=True,
                )
                payload = {
                    "format": "UNARCHV1_STUDENT_DISTILLATION_V1_DDP",
                    "oracle_checkpoint": str(args.oracle),
                    "oracle_metrics": oracle_checkpoint.get("metrics", {}),
                    "student_config": student_config,
                    "distillation_config": distill_config,
                    "hardware": root["hardware"],
                    "precision": precision["name"],
                    "world_size": distributed_context.world_size,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "microbatch_per_gpu": microbatch,
                    "accumulation": accumulation,
                    "effective_global_batch": effective_batch,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "records_per_epoch": records_per_epoch,
                    "sampling": "global_without_replacement",
                    "epochs_without_improvement": epochs_without_improvement,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "memory_rank0": memory,
                    "train_manifest": train_data.manifest(),
                    "validation_manifest": validation_data.manifest(),
                }
                atomic_torch_save(payload, args.output)
                if improved:
                    atomic_torch_save(payload, str(args.output) + ".best")
                if early_stop:
                    print(
                        f"early stop: student validation NLL failed to improve for "
                        f"{epochs_without_improvement} epochs",
                        flush=True,
                    )
                if memory_growth_abort:
                    print(
                        "abort: distillation CUDA reserved memory exceeded its stable envelope",
                        flush=True,
                    )
            distributed_context.barrier()
            if memory_growth_abort:
                raise RuntimeError("distillation reserved-memory growth guard triggered")
            if early_stop:
                break
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        distributed_context.close()


def calibrate_student_checkpoint(args):
    root = json.loads(Path(args.config).read_text())
    oracle_config = root["oracle"]
    device = configure_torch(oracle_config["seed"], args.deterministic)
    data = V4RecordShards(args.validation)
    checkpoint_data = torch.load(args.student, map_location=device, weights_only=False)
    config = checkpoint_data["student_config"]
    model = AegisV4Chessformer(config).to(device)
    model.load_state_dict(checkpoint_data["model"])
    from train_chessformer_v4_a100 import fit_regret_calibration

    calibration = fit_regret_calibration(
        model,
        data,
        device,
        config,
        args.validation_records,
    )
    payload = dict(checkpoint_data)
    payload["format"] = "UNARCHV1_STUDENT_CALIBRATED_V1"
    payload["calibration"] = calibration
    payload["calibration_manifest"] = data.manifest()
    output = args.output or Path(str(args.student) + ".calibrated")
    atomic_torch_save(payload, output)
    write_metrics(
        args.metrics_json,
        {
            "input_checkpoint": str(args.student),
            "output_checkpoint": str(output),
            "calibration": calibration,
            "calibration_manifest": data.manifest(),
        },
    )


def write_metrics(path, metrics):
    text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")


def evaluate_checkpoint(args, student=False):
    root = json.loads(Path(args.config).read_text())
    oracle_config = root["oracle"]
    device = configure_torch(oracle_config["seed"], args.deterministic)
    data = V4RecordShards(args.validation)
    checkpoint_data = torch.load(
        args.student if student else args.oracle,
        map_location=device,
        weights_only=False,
    )
    if student:
        config = checkpoint_data["student_config"]
        model = AegisV4Chessformer(config).to(device)
        model.load_state_dict(checkpoint_data["model"])
        from train_chessformer_v4_a100 import evaluate

        calibration = checkpoint_data.get("calibration")
        metrics = evaluate(
            model,
            data,
            device,
            config,
            calibration or {"upper_regret_quantile": 0.0},
            args.validation_records,
        )
        if calibration is None:
            metrics.pop("regret_upper_coverage", None)
            metrics["regret_calibration"] = "missing; coverage intentionally omitted"
        else:
            metrics["regret_calibration"] = calibration
    else:
        config = checkpoint_data["config"]
        model = HydraOracleV5(config).to(device)
        model.load_state_dict(checkpoint_data["model"])
        metrics = evaluate_oracle(
            model, data, device, config, args.validation_records
        )
    metrics = {
        "checkpoint": str(args.student if student else args.oracle),
        "checkpoint_format": checkpoint_data.get("format"),
        "final_holdout": data.manifest(),
        "metrics": metrics,
    }
    write_metrics(args.metrics_json, metrics)


def synthetic_records(count, seed=11):
    from train_chessformer_v4_a100 import synthetic_records as v4_synthetic

    records = v4_synthetic(count, seed)
    for index in range(count):
        if index % 2:
            records["teacher_score"][index] = 100
    return records


def selfcheck(args):
    config, hardware = load_config(args.config, "oracle")
    config = {
        **config,
        "d_model": 64,
        "board_layers": 2,
        "board_heads": 4,
        "board_ffn": 128,
        "gab_token_projection": 4,
        "gab_hidden": 16,
        "gab_templates": 8,
        "decoder_layers": 1,
        "decoder_heads": 4,
        "decoder_ffn": 128,
        "history_width": 16,
        "policy_adapter_rank": 8,
        "concept_count": 16,
        "concept_width": 8,
        "activation_checkpointing": False,
    }
    device = configure_torch(config["seed"], True)
    model = HydraOracleV5(config).to(device)
    optimizer = make_optimizer(model, config["learning_rate"], config["weight_decay"], device)
    records = synthetic_records(8)
    for step in range(2):
        batch = prepare_oracle_batch(records, device, augment=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, _ = oracle_loss(output, batch, config, step)
        loss.backward()
        optimizer.step()
    assert output["logits"].shape == (8, 218)
    assert output["quantiles"].shape == (8, config["score_quantiles"])
    assert torch.isfinite(loss)
    print(
        f"selfcheck PASS device={device} parameters={model_parameter_count(model):,} "
        f"loss={float(loss):.5f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "selfcheck",
            "train-oracle",
            "distill-student",
            "calibrate-student",
            "evaluate-oracle",
            "evaluate-student",
        ),
    )
    parser.add_argument("--config", default="config/unarchitectured_v1_training.json")
    parser.add_argument("--student-config")
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--validation", nargs="+", default=[])
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--output", type=Path, default=Path("hydra-apex-v5.pt"))
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--incident", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--validation-records", type=int, default=200_000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "selfcheck":
        selfcheck(arguments)
    elif arguments.command == "train-oracle":
        if not arguments.train or not arguments.validation:
            raise SystemExit("train-oracle requires --train and --validation v4 guide/human shards")
        train_oracle(arguments)
    elif arguments.command == "distill-student":
        if not arguments.oracle or not arguments.train or not arguments.validation:
            raise SystemExit("distill-student requires --oracle, --train, and --validation")
        distill_student(arguments)
    elif arguments.command == "calibrate-student":
        if not arguments.student or not arguments.validation:
            raise SystemExit("calibrate-student requires --student and --validation")
        calibrate_student_checkpoint(arguments)
    elif arguments.command == "evaluate-oracle":
        if not arguments.oracle or not arguments.validation:
            raise SystemExit("evaluate-oracle requires --oracle and --validation")
        evaluate_checkpoint(arguments, student=False)
    else:
        if not arguments.student or not arguments.validation:
            raise SystemExit("evaluate-student requires --student and --validation")
        evaluate_checkpoint(arguments, student=True)
