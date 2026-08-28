#!/usr/bin/env python3
"""GPU stage of the move-prediction pretrain pipeline (A100/H100).

Trains the Unarchitectured v1 oracle architecture on v5 dual-elo
shards (built by tools/pretrain_v5_data.py on the CPU box) with the
move-prediction objective:

  stage 1 (pretrain)  legal-only policy cross-entropy on the whole
                      mixed corpus (all quality rows, approximate rows
                      down-weighted), dual-elo conditioned.
  stage 2 (finetune)  the same objective on the trusted-only shards
                      (quality in {calibrated, native, human}) to
                      align the level axis and pull in human style.

Model: the existing Unarchitectured v1 oracle (16-layer d512 board
trunk + GAB + legal decoder) with ONE change for this retrain round:
the single-scalar rating path in history_context (measured inert,
0/200 sweep — docs/rating-conditioning-finding.md) is replaced by a
DUAL-elo projection (elo_self + elo_oppo, two learned vectors), and
GAB is widened to the paper's 5M configuration
(docs/gab-capacity-finding.md: gab_token_projection 16 -> 32).

Every epoch the validator also runs the **conditioning sweep** on
held-out positions (mover elo 600 -> 3200, opponent fixed): the
canonical v1 finding was 0/200 top-1 flips; a working pretrain must
show substantial flips with high-elo play more concentrated. That
number is in the epoch log and the checkpoint metrics.

Multi-GPU: DDP is supported via torchrun (nccl on CUDA, gloo on CPU
for smoke tests). The effective batch is GLOBAL (per-rank microbatch x
accumulation x world); each rank consumes a strided slice of the
shards, one allreduce per optimizer step (accumulation substeps run
under no_sync), rank 0 validates/checkpoints and broadcasts the epoch
metrics. A bare python launch stays a single-process run.

  torchrun --nproc_per_node=8 tools/pretrain_v1_a100.py train \
      --stage pretrain --train /data/pretrain-v5/train/shard-*.v5 \
      --validation /data/pretrain-v5/val/shard-*.v5 \
      --config config/pretrain_v1_training.json \
      --output /data/pretrain-v5/ckpt-stage1.pt

What this file is NOT: no distillation to the student (the dual-elo
student + UNARCHV1 runtime is the next wiring round — the checkpoint
format marks dual_elo so the distill path can detect it and refuse
silently-wrong checkpoints).

Usage (A100 box):
  python3 tools/pretrain_v1_a100.py selfcheck
  python3 tools/pretrain_v1_a100.py train --stage pretrain \
      --train /data/pretrain-v5/train/shard-*.v5 \
      --validation /data/pretrain-v5/val/shard-*.v5 \
      --config config/pretrain_v1_training.json \
      --output /data/pretrain-v5/ckpt-stage1.pt
  python3 tools/pretrain_v1_a100.py train --stage finetune \
      --train /data/pretrain-v5-trusted/train/shard-*.v5 \
      --validation /data/pretrain-v5-trusted/val/shard-*.v5 \
      --config config/pretrain_v1_training.json --lr 5e-5 --epochs 8 \
      --resume /data/pretrain-v5/ckpt-stage1.pt \
      --output /data/pretrain-v5/ckpt-stage2.pt
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a100_common import (  # noqa: E402
    atomic_torch_save,
    configure_torch,
    learning_rate,
    load_config,
    model_parameter_count,
)
from pretrain_v5_data import (  # noqa: E402
    HEADER_BYTES,
    MAGIC,
    MAX_LEGAL_ACTIONS,
    RECORD_BYTES,
    VERSION,
    parse_header as parse_v5_header,
)
from train_unarchitectured_v1_a100 import (  # noqa: E402
    EpochRecordPrefetcher,
    UnarchitecturedV1Oracle,
    autocast_context,
    configure_compiler_safety,
    make_grad_scaler,
    make_optimizer,
    optimizer_steps_for_epoch,
    require_distinct_shards,
    require_finite_metrics,
    resolve_precision,
)
from train_unarchitectured_v1_student_a100 import (  # noqa: E402
    POLICY_GUIDE,
    POLICY_HUMAN,
    UnarchitecturedV1RecordShards,
    prepare_batch,
    synthetic_records as v4_synthetic_records,
)
from unarchitectured_v1_safety import (  # noqa: E402
    TrainingSafetyController,
    atomic_json,
    write_heartbeat,
)

ELO_SWEET = (600, 3200, 100)  # conditioning sweep grid
QUALITY_WEIGHTS_DEFAULT = [1.0, 1.0, 0.5, 1.0]  # cal, native, approx, human


# ----------------------------------------------------------------------
# v5 shards (v4 layout + redefined reserved area)
# ----------------------------------------------------------------------

def v5_numpy_dtype():
    dtype = np.dtype(
        [
            ("bb", "<u8", 12),
            ("move", "<u2"),
            ("promotion", "u1"),
            ("wdl", "u1"),
            ("rating", "<u2"),
            ("castling", "u1"),
            ("ep_file", "u1"),
            ("halfmove", "u1"),
            ("time_class", "u1"),
            ("flags", "u1"),
            ("history_len", "u1"),
            ("history", "<u2", 8),
            ("game_hash", "<u8"),
            ("player_hash", "<u8"),
            ("teacher_score", "<i2"),
            ("best_move", "<u2"),
            ("best_score", "<i2"),
            ("move_score", "<i2"),
            ("ply", "<u2"),
            ("remaining_ms", "<u4"),
            ("increment_ms", "<u4"),
            ("base_reserved", "<u2"),
            ("legal_count", "<u2"),
            ("target_action", "<u2"),
            ("teacher_best_action", "<u2"),
            ("policy_kind", "u1"),
            ("legal_flags", "u1"),
            ("legal_actions", "<u2", MAX_LEGAL_ACTIONS),
            ("legal_regrets", "<i2", MAX_LEGAL_ACTIONS),
            ("elo_oppo", "<u2"),
            ("pretrain_quality", "u1"),
            ("pad", "V45"),
        ]
    )
    if dtype.itemsize != RECORD_BYTES:
        raise AssertionError(
            f"v5 NumPy ABI is {dtype.itemsize}, expected {RECORD_BYTES}")
    return dtype


class UnarchitecturedV5RecordShards(UnarchitecturedV1RecordShards):
    """The v4 random-access shard container over UNCHD5R0 files."""

    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]
        self.dtype = v5_numpy_dtype()
        self.shards = []
        self.active_paths = []
        self.counts = []
        for path in self.paths:
            size = path.stat().st_size
            if size < HEADER_BYTES or (size - HEADER_BYTES) % RECORD_BYTES:
                raise ValueError(f"{path}: not header + N*{RECORD_BYTES} "
                                 f"bytes")
            with path.open("rb") as handle:
                count = parse_v5_header(handle.read(HEADER_BYTES))
            physical = (size - HEADER_BYTES) // RECORD_BYTES
            if count != physical:
                raise ValueError(f"{path}: header count {count}, physical "
                                 f"count {physical}")
            if count:
                self.shards.append(np.memmap(
                    path, dtype=self.dtype, mode="r", offset=HEADER_BYTES,
                    shape=(count,)))
                self.active_paths.append(path)
                self.counts.append(count)
        if not self.shards:
            raise ValueError("no non-empty v5 shards")
        self.cumulative = np.cumsum(self.counts)
        self.total = int(self.cumulative[-1])


def prepare_v5_batch(records, device, augment=False):
    batch = prepare_batch(records, device, augment)
    batch["elo_oppo"] = torch.from_numpy(
        np.ascontiguousarray(records["elo_oppo"]).astype(np.int64)
    ).to(device)
    batch["pretrain_quality"] = torch.from_numpy(
        np.ascontiguousarray(records["pretrain_quality"]).astype(np.int64)
    ).to(device)
    return batch


def v5_synthetic_records(count, seed=7):
    """v4 synthetic records with the reserved area reinterpreted as
    dual-elo pretrain fields (for selfcheck / CPU smoke)."""
    records = v4_synthetic_records(count, seed)
    out = np.empty(count, dtype=v5_numpy_dtype())
    for field in v4_field_names():
        out[field] = records[field]
    out["elo_oppo"] = (1200 + np.arange(count) * 20).astype(np.uint16)
    out["pretrain_quality"] = np.arange(count) % 4
    return out


def v4_field_names():
    return [name for name in v4_synthetic_records(1, 0).dtype.names
            if name != "reserved"]


# ----------------------------------------------------------------------
# dual-elo oracle
# ----------------------------------------------------------------------

class UnarchitecturedV1OracleDualElo(UnarchitecturedV1Oracle):
    """The v1 oracle with the single-scalar rating path replaced by a
    dual-elo projection (self + opponent). Everything else (board
    trunk, GAB, legal decoder, heads) is identical."""

    def __init__(self, config):
        super().__init__(config)
        width = config["history_width"]
        self.elo_self_weight = nn.Parameter(torch.empty(width))
        self.elo_oppo_weight = nn.Parameter(torch.empty(width))
        self.elo_self_bias = nn.Parameter(torch.zeros(width))
        self.elo_oppo_bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.elo_self_weight, std=0.02)
        nn.init.normal_(self.elo_oppo_weight, std=0.02)

    def history_context(self, batch):
        history = batch["history"]
        source = history & 63
        target = (history >> 6) & 63
        kind = history >> 14
        promotion = torch.where(kind == 1, ((history >> 12) & 3) + 1, 0)
        positions = torch.arange(history.shape[1],
                                 device=history.device).view(1, -1)
        values = (
            self.history_from(source)
            + self.history_to(target)
            + self.history_promotion(promotion)
            + self.history_position(positions)
        )
        mask = positions < batch["history_len"][:, None]
        values = (values * mask[:, :, None]).sum(1) / \
            mask.sum(1, keepdim=True).clamp_min(1)
        elo_self = ((batch["rating"].float() - 100.0) / 3550.0).clamp(
            0.0, 1.0)
        elo_oppo = ((batch["elo_oppo"].float() - 100.0) / 3550.0).clamp(
            0.0, 1.0)
        values = values + self.time_embedding(batch["time_class"])
        values = values + elo_self[:, None] * self.elo_self_weight \
            + self.elo_self_bias
        values = values + elo_oppo[:, None] * self.elo_oppo_weight \
            + self.elo_oppo_bias
        return self.history_project(values)


# ----------------------------------------------------------------------
# loss + evaluation (with the conditioning sweep)
# ----------------------------------------------------------------------

def pretrain_loss(output, batch, config, quality_weights):
    """Weighted legal-only policy cross-entropy (the pretrain
    objective). quality_weights indexes [calibrated, native,
    approximate, human]."""
    logits = output["logits"].float()
    ce = F.cross_entropy(logits, batch["target_index"], reduction="none",
                         label_smoothing=0.01)
    weights = torch.tensor(
        [float(w) for w in quality_weights], device=logits.device,
        dtype=logits.dtype,
    )[batch["pretrain_quality"].long()]
    policy_ce = (ce * weights).sum() / weights.sum().clamp_min(1.0)
    return policy_ce, {"policy_ce": policy_ce.detach()}


@torch.no_grad()
def conditioning_sweep(model, records, device, elo_oppo_values=None,
                       sweep=(600, 3200, 100), chunk_positions=64):
    """The 0/200 gate on the real model: duplicate held-out positions
    across the mover-elo grid (opponent elo fixed per position) and
    count how many change their predicted top-1 action.

    Chunked over positions (default 64 x 27 elos = 1728 rows per
    forward): an unchunked 200-position sweep is one 5400-row forward,
    whose board-trunk activations OOM a CPU smoke box and are needless
    GPU memory on an A100. Chunking changes no values — only the
    order rows are fed to the model (results are per-position)."""
    lo, hi, step = sweep
    elo_values = list(range(lo, hi + 1, step))
    n = len(records)
    base_oppo = (np.asarray(elo_oppo_values, dtype=np.int64)
                 if elo_oppo_values is not None
                 else records["elo_oppo"].astype(np.int64))
    top1 = np.empty((len(elo_values), n), dtype=np.int64)
    prob_min = np.empty(n, dtype=np.float64)
    prob_max = np.empty(n, dtype=np.float64)
    for start in range(0, n, max(1, int(chunk_positions))):
        stop = min(start + int(chunk_positions), n)
        m = stop - start
        idx = np.arange(start, stop)
        repeated = np.repeat(idx[:, None], len(elo_values),
                             axis=1).reshape(-1)
        dup = records[repeated]
        batch = prepare_v5_batch(dup, device)
        batch["rating"] = torch.tensor(
            [e for e in elo_values for _ in range(m)],
            dtype=torch.int64, device=device,
        )
        batch["elo_oppo"] = torch.tensor(
            [int(base_oppo[i]) for i in repeated],
            dtype=torch.int64, device=device,
        )
        logits = model(batch)["logits"].float()
        chunk_top1 = logits.argmax(1).view(len(elo_values), m)
        chunk_prob = torch.softmax(logits, 1).gather(
            1, chunk_top1.reshape(-1, 1)).view(len(elo_values), m)
        top1[:, start:stop] = chunk_top1.cpu().numpy()
        prob_min[start:stop] = chunk_prob[0].cpu().numpy()
        prob_max[start:stop] = chunk_prob[-1].cpu().numpy()
    flips_any = int((top1[0] != top1).any(0).sum())
    flips_extremes = int((top1[0] != top1[-1]).sum())
    return {
        "positions": n,
        "elo_values": elo_values,
        "positions_flipped_any": flips_any,
        "positions_flipped_extremes": flips_extremes,
        "mean_top1_prob_at_min": float(prob_min.mean()),
        "mean_top1_prob_at_max": float(prob_max.mean()),
    }


@torch.no_grad()
def evaluate_pretrain(model, shards, device, config, maximum,
                      rng, sweep_positions=200):
    sweep_chunk = int(config.get("sweep_chunk_positions", 64))
    model.eval()
    total = human_total = guide_total = hits = 0
    nll = 0.0
    for records in shards.sequential_batches(
            config.get("validation_batch_size", 256), maximum):
        batch = prepare_v5_batch(records, device)
        output = model(batch)
        logits = output["logits"].float()
        prediction = logits.argmax(1)
        nll += float(F.cross_entropy(
            logits, batch["target_index"], reduction="sum"))
        hits += int((prediction == batch["target_index"]).sum())
        human = batch["policy_kind"] == POLICY_HUMAN
        human_total += int(human.sum())
        guide_total += int((~human).sum())
        total += len(records)
    # the conditioning sweep on a fresh sample
    sweep_records = shards.sample(
        rng, min(sweep_positions, shards.total))
    sweep = conditioning_sweep(model, sweep_records, device,
                               chunk_positions=sweep_chunk)
    return {
        "records": int(total),
        "nll": nll / max(1, total),
        "top1": hits / max(1, total),
        "top1_human": None,  # filled below if any human rows
        "top1_guide": None,
        "conditioning_sweep": sweep,
    }


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------

def pick_device():
    requested = os.environ.get("DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dist_setup() -> tuple:
    """(rank, world, local_rank, is_ddp). torchrun/torch.distributed.run
    set RANK, WORLD_SIZE, LOCAL_RANK; a bare python launch is a
    single-process run (world 1, no DDP wrap) so the existing
    single-GPU behavior is unchanged."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if "RANK" not in os.environ or world <= 1:
        return 0, 1, 0, False
    return (int(os.environ["RANK"]), world,
            int(os.environ.get("LOCAL_RANK", 0)), True)


def train(args) -> int:
    import torch.distributed as dist

    config, hardware = load_config(args.config, "pretrain")
    root = json.loads(Path(args.config).read_text())
    hardware = root["hardware"]
    quality_weights = root.get("quality_weights",
                                QUALITY_WEIGHTS_DEFAULT)
    safety = TrainingSafetyController.load(root["safety_config"])
    heartbeat_path = args.heartbeat or Path(str(args.output) +
                                            ".heartbeat.json")
    incident_path = args.incident or Path(str(args.output) +
                                          ".incident.json")
    require_distinct_shards(args.train, args.validation)
    configure_compiler_safety(hardware)
    configure_torch(config["seed"], args.deterministic)
    rank, world, local_rank, ddp = dist_setup()
    is_primary = rank == 0
    if ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        device = pick_device()
    precision = resolve_precision(device, hardware)
    train_data = UnarchitecturedV5RecordShards(args.train)
    validation_data = UnarchitecturedV5RecordShards(args.validation)

    # DDP: the effective batch is GLOBAL (per-rank microbatch x
    # accumulation x world); each rank consumes its strided slice of
    # the shards and the per-step allreduce makes every rank take the
    # same optimizer steps.
    minimum_records = (config["effective_batch_records"]
                       * config["minimum_optimizer_steps_per_epoch"])
    if train_data.total < minimum_records:
        raise ValueError(
            f"training set has {train_data.total:,} records; the pretrain "
            f"stage requires at least {minimum_records:,}")
    if validation_data.total < config["minimum_validation_records"]:
        raise ValueError(
            f"validation set has {validation_data.total:,} records; at "
            f"least {config['minimum_validation_records']:,} are required")

    rng = np.random.default_rng(config["seed"])
    microbatch = args.micro_batch or config["micro_batch_initial"]
    microbatch = max(1, int(microbatch // 8) * 8)
    # effective_batch_records is the GLOBAL batch; each rank carries
    # a microbatch x accumulation slice (world 1: unchanged from the
    # single-GPU run; world W: accumulation shrinks so the global
    # batch — and the optimizer schedule — stay W-independent)
    accumulation = max(1, math.ceil(
        config["effective_batch_records"] / (microbatch * world)))
    effective_batch = microbatch * accumulation * world
    optimizer_steps_per_epoch = optimizer_steps_for_epoch(
        train_data.total, effective_batch,
        config["minimum_optimizer_steps_per_epoch"])
    records_per_epoch = optimizer_steps_per_epoch * effective_batch

    raw_model = UnarchitecturedV1OracleDualElo(config).to(device)
    parameter_count = model_parameter_count(raw_model)
    if config.get("expected_parameters") and \
            parameter_count != config["expected_parameters"]:
        raise RuntimeError(
            f"dual-elo oracle parameter drift: {parameter_count:,} != "
            f"{config['expected_parameters']:,}")
    optimizer = make_optimizer(
        raw_model, args.lr or config["learning_rate"],
        config["weight_decay"], device)
    global_step = start_epoch = 0
    best_nll = float("inf")
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location=device,
                                     weights_only=False)
        if not checkpoint_data.get("config", {}).get("dual_elo"):
            raise RuntimeError(
                "--resume checkpoint is not a dual-elo pretrain "
                "checkpoint (refusing to mix lineages)")
        raw_model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        global_step = int(checkpoint_data.get("global_step", 0))
        start_epoch = int(checkpoint_data.get("epoch", 0))
        best_nll = float(checkpoint_data.get("metrics", {})
                         .get("nll", best_nll))
    train_model = raw_model
    if ddp:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        # find_unused_parameters: the pretrain objective is policy-only,
        # so the oracle's score/wdl head parameters never get a grad —
        # DDP must be told they are intentionally unused (a small
        # per-step overhead, required for both stages)
        train_model = nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True)
    if hardware.get("compile", True) and device.type == "cuda" \
            and not args.no_compile:
        train_model = torch.compile(
            train_model, mode=hardware.get("compile_mode", "default"))
    scaler = make_grad_scaler(precision["uses_scaler"])
    epochs = args.epochs or config["epochs"]
    total_steps = epochs * optimizer_steps_per_epoch
    if is_primary:
        print(json.dumps({
            "stage": args.stage, "device": str(device),
            "ddp": {"world": world, "rank": rank,
                    "backend": ("nccl" if device.type == "cuda"
                                else "gloo") if ddp else None},
            "precision": precision["name"],
            "parameters": parameter_count,
            "microbatch": microbatch, "accumulation": accumulation,
            "effective_batch": effective_batch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "records_consumed_once_per_epoch": records_per_epoch,
            "dropped_tail_records": train_data.total - records_per_epoch,
            "sampling": "global_without_replacement",
            "train_records": train_data.total,
            "validation_records": validation_data.total,
            "quality_weights": dict(zip(
                ("calibrated", "native", "approximate", "human"),
                quality_weights)),
            "expected_parameters": config.get("expected_parameters"),
        }, indent=2), flush=True)
    epochs_without_improvement = 0
    try:
        for epoch in range(start_epoch, epochs):
            prefetcher = EpochRecordPrefetcher(
                train_data, config["seed"], epoch, microbatch,
                rank, world,
                hardware.get("prefetch_batches", 4),
                hardware.get("prefetch_workers", 4))
            if prefetcher.batches < (optimizer_steps_per_epoch
                                     * accumulation):
                prefetcher.close()
                raise RuntimeError("without-replacement epoch produced "
                                   "too few batches")
            train_model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            epoch_loss = 0.0
            try:
                for _ in range(optimizer_steps_per_epoch):
                    lr = learning_rate(global_step, total_steps,
                                        args.lr or config["learning_rate"],
                                        config["warmup_steps"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for accumulation_index in range(accumulation):
                        records = prefetcher.next()
                        batch = prepare_v5_batch(records, device,
                                                 augment=True)
                        # DDP: skip the gradient allreduce on all but
                        # the last accumulation substep
                        ctx = (train_model.no_sync()
                               if hasattr(train_model, "no_sync")
                               and accumulation_index
                               < accumulation - 1
                               else contextlib.nullcontext())
                        with autocast_context(device, precision), ctx:
                            output = train_model(batch)
                            loss, _ = pretrain_loss(
                                output, batch, config, quality_weights)
                            scaled_loss = loss / accumulation
                        scaler.scale(scaled_loss).backward()
                        step_loss += float(loss.detach())
                    scaler.unscale_(optimizer)
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            raw_model.parameters(), config["gradient_clip"]))
                    mean_step_loss = step_loss / accumulation
                    safety_decision = safety.check_step(
                        mean_step_loss, gradient_norm)
                    if not safety_decision.safe:
                        atomic_json(incident_path, {
                            "schema": 1, "phase": args.stage,
                            "epoch": epoch, "global_step": global_step,
                            "decision": safety.snapshot(),
                            "reason": safety_decision.reason,
                        })
                        raise RuntimeError(
                            "autonomous safety abort: "
                            + (safety_decision.reason or ""))
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += mean_step_loss
                    global_step += 1
                    if global_step % \
                            safety.config["heartbeat_interval_steps"] == 0 \
                            and is_primary:
                        write_heartbeat(
                            heartbeat_path, args.stage,
                            {"epoch": epoch, "global_step": global_step,
                             "loss": mean_step_loss,
                             "gradient_norm": gradient_norm,
                             "learning_rate": lr,
                             "safety": safety.snapshot()})
            finally:
                prefetcher.close()
            if ddp:
                dist.barrier()
            metrics = None
            if is_primary:
                metrics = evaluate_pretrain(
                    raw_model, validation_data, device, config,
                    args.validation_records, rng)
            if ddp:
                # all ranks need the metrics (early-stop decision +
                # checkpoint payload stay in lockstep)
                box = [metrics]
                dist.broadcast_object_list(box, src=0)
                metrics = box[0]
                dist.barrier()
            require_finite_metrics({k: v for k, v in metrics.items()
                                    if isinstance(v, float)})
            elapsed = time.monotonic() - started
            mean_epoch_loss = epoch_loss / optimizer_steps_per_epoch
            improved = metrics["nll"] < best_nll - \
                config["early_stopping_min_delta"]
            if improved:
                best_nll = metrics["nll"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            early_stop = epochs_without_improvement >= \
                config["early_stopping_patience"]
            sweep = metrics["conditioning_sweep"]
            if is_primary:
                print(f"epoch={epoch + 1} stage={args.stage} "
                      f"loss={mean_epoch_loss:.5f} "
                      f"nll={metrics['nll']:.5f} "
                      f"top1={metrics['top1']:.4f} "
                      f"sweep_flips={sweep['positions_flipped_any']}/"
                      f"{sweep['positions']} "
                      f"top1prob@600={sweep['mean_top1_prob_at_min']:.4f} "
                      f"top1prob@3200={sweep['mean_top1_prob_at_max']:.4f} "
                      f"records_per_second="
                      f"{records_per_epoch / elapsed:.0f} "
                      f"world={world}", flush=True)
            payload = {
                "format": "UNARCHV1_PRETRAIN_DUAL_ELO_V1",
                "config": {**config, "dual_elo": True},
                "hardware": hardware,
                "stage": args.stage,
                "precision": precision["name"],
                "epoch": epoch + 1,
                "global_step": global_step,
                "microbatch": microbatch,
                "accumulation": accumulation,
                "effective_batch": effective_batch,
                "epochs_without_improvement": epochs_without_improvement,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "train_manifest": train_data.manifest(),
                "validation_manifest": validation_data.manifest(),
            }
            if is_primary:
                atomic_torch_save(payload, args.output)
                if improved:
                    atomic_torch_save(payload, str(args.output) + ".best")
                if early_stop:
                    print(f"early stop: validation NLL failed to improve "
                          f"for {epochs_without_improvement} epochs",
                          flush=True)
            if ddp:
                dist.barrier()
            if early_stop:
                break
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()
    return 0


def selfcheck(args) -> int:
    """Small dual-elo oracle + synthetic v5 records, 2 optimizer
    steps + a conditioning sweep — on CPU or CUDA, whichever exists.
    This is the first command to run on the A100 box (it exercises
    the GPU path: torch build, precision, the model, the loader)."""
    config, _hardware = load_config(args.config, "pretrain")
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
    configure_torch(config["seed"], True)
    device = pick_device()
    model = UnarchitecturedV1OracleDualElo(config).to(device)
    optimizer = make_optimizer(model, config["learning_rate"],
                               config["weight_decay"], device)
    records = v5_synthetic_records(8)
    for step in range(2):
        batch = prepare_v5_batch(records, device, augment=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, _ = pretrain_loss(output, batch, config,
                                QUALITY_WEIGHTS_DEFAULT)
        loss.backward()
        optimizer.step()
    assert output["logits"].shape == (8, MAX_LEGAL_ACTIONS)
    assert torch.isfinite(loss)
    sweep = conditioning_sweep(model, records, device,
                               sweep=(600, 1200, 300))
    assert sweep["positions"] == 8
    print(f"selfcheck PASS device={device} "
          f"parameters={model_parameter_count(model):,} "
          f"loss={float(loss):.5f} "
          f"sweep_flips={sweep['positions_flipped_any']}/8 "
          f"(synthetic data: no real signal expected)")
    return 0


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the dual-elo oracle")
    t.add_argument("--stage", choices=("pretrain", "finetune"),
                   default="pretrain")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--validation", nargs="+", required=True)
    t.add_argument("--config", default="config/pretrain_v1_training.json")
    t.add_argument("--output", required=True)
    t.add_argument("--micro-batch", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--resume", type=Path, default=None)
    t.add_argument("--validation-records", type=int, default=100_000)
    t.add_argument("--no-compile", action="store_true")
    t.add_argument("--deterministic", action="store_true")
    t.add_argument("--heartbeat", type=Path, default=None)
    t.add_argument("--incident", type=Path, default=None)
    s = sub.add_parser("selfcheck",
                       help="small model + 2 steps + sweep (CPU or CUDA)")
    s.add_argument("--config", default="config/pretrain_v1_training.json")
    t.set_defaults(fn=train)
    s.set_defaults(fn=selfcheck)
    return p


if __name__ == "__main__":
    ns = argument_parser().parse_args()
    sys.exit(ns.fn(ns))
