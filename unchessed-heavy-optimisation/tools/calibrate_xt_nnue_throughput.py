#!/usr/bin/env python3
"""Time a few hundred real XT-NNUE training steps on the target GPU and
report samples/sec, so the full 30-epoch/600k-step budget (and its cost) can
be decided from a measurement instead of a guess. Run this before committing
to a full training run -- the dense (batch x 64 x 64) threat-relation
computation has no directly comparable prior throughput number from the
project's other NNUE training runs.

Usage:
    python tools/calibrate_xt_nnue_throughput.py --train shard0.bin --steps 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a100_common import FixedRecordShards, configure_torch, load_config
from train_nnue_xt_a100 import (
    NNUE_REC_SPEC,
    ThreatIndexer,
    XtNnue,
    constants,
    make_batch,
    make_optimizer,
    xt_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/a100_hybrid_training.json")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config, hardware = load_config(args.config, "xt_nnue")
    device = configure_torch(config["seed"], deterministic=False)
    dtype = np.dtype(NNUE_REC_SPEC)
    shards = FixedRecordShards(args.train, dtype)
    print(f"device={device} records={shards.total:,} batch_size={config['batch_size']}")

    rng = np.random.default_rng(config["seed"])
    fixed = constants(device)
    indexer = ThreatIndexer(device)
    raw_model = XtNnue(config).to(device)
    optimizer = make_optimizer(raw_model, config, device)
    train_model = raw_model
    if hardware.get("compile", True) and device.type == "cuda" and not args.no_compile:
        train_model = torch.compile(raw_model, mode="max-autotune", dynamic=True)

    import contextlib

    def run_steps(n, label):
        started = time.monotonic()
        for _ in range(n):
            records = shards.sample(rng, config["batch_size"])
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        samples = n * config["batch_size"]
        print(
            f"{label}: {n} steps, {samples:,} samples, {elapsed:.1f}s, "
            f"{samples / elapsed:,.0f} samples/s, {elapsed / n * 1000:.1f} ms/step"
        )
        return samples / elapsed

    # torch.compile pays its autotune cost on the first calls -- warm up
    # separately so it doesn't pollute the measured rate.
    run_steps(args.warmup_steps, "warmup (excluded from rate)")
    rate = run_steps(args.steps, "measured")

    total_samples = config["epochs"] * config["steps_per_epoch"] * config["batch_size"]
    projected_hours = total_samples / rate / 3600
    print()
    print(f"projected full run ({config['epochs']} epochs, {total_samples:,} samples): "
          f"{projected_hours:.1f} hours at this rate")
    if device.type == "cuda":
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
