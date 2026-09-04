#!/usr/bin/env python3
"""Time real-size Unarchitectured v1 oracle/student forward+backward steps on
the target GPU, so the config's 24+16 epoch training budget (and its cost)
can be decided from a measurement instead of a guess.

Uses synthetic data (same generator selfcheck uses) at the *real* model size
from config/unarchitectured_v1_training.json -- selfcheck deliberately shrinks
the model for a fast CPU smoke test, which tells us nothing about GPU
throughput at the actual 58M-param oracle / real student size.

Usage:
    python tools/calibrate_unarchitectured_v1_throughput.py --steps 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a100_common import configure_torch, model_parameter_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unarchitectured_v1_training.json")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    return parser.parse_args()


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_args()

import torch


def time_steps(label, step_fn, warmup, steps, microbatch):
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(steps):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    samples = steps * microbatch
    rate = samples / elapsed
    print(
        f"{label}: {steps} steps, microbatch={microbatch}, {elapsed:.2f}s, "
        f"{rate:,.1f} samples/s, {elapsed / steps * 1000:.1f} ms/step"
    )
    if torch.cuda.is_available():
        print(f"  peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    return rate


def project(rate, effective_batch, minimum_steps_per_epoch, epochs):
    total_samples = effective_batch * minimum_steps_per_epoch * epochs
    hours = total_samples / rate / 3600
    print(
        f"  projected {epochs}-epoch run at gate-floor size "
        f"({total_samples:,} samples): {hours:.2f} hours at this rate"
    )


def calibrate_oracle(root, device, steps, warmup):
    from train_unarchitectured_v1_a100 import (
        UnarchitecturedV1Oracle,
        make_optimizer,
        oracle_loss,
        prepare_oracle_batch,
        synthetic_records,
    )

    config = root["oracle"]
    microbatch = config["micro_batch_initial"]
    model = UnarchitecturedV1Oracle(config).to(device)
    optimizer = make_optimizer(model, config["learning_rate"], config["weight_decay"], device)
    records = synthetic_records(microbatch)
    print(f"oracle: parameters={model_parameter_count(model):,} (config expects {config['expected_parameters']:,})")

    step_counter = [0]

    def step():
        batch = prepare_oracle_batch(records, device, augment=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, _ = oracle_loss(output, batch, config, step_counter[0])
        loss.backward()
        optimizer.step()
        step_counter[0] += 1

    rate = time_steps("oracle", step, warmup, steps, microbatch)
    project(rate, config["effective_batch_records"], config["minimum_optimizer_steps_per_epoch"], config["epochs"])


def calibrate_student(root, device, steps, warmup):
    from train_unarchitectured_v1_student_a100 import (
        UnarchitecturedV1Student,
        make_optimizer,
        prepare_batch,
        synthetic_records,
        training_loss,
    )

    student_config_path = root["student_distillation"]["student_config"]
    student_arch = json.loads(Path(student_config_path).read_text())["chessformer"]
    distill_config = root["student_distillation"]
    microbatch = distill_config["micro_batch_initial"]
    model = UnarchitecturedV1Student(student_arch).to(device)
    optimizer = make_optimizer(model, student_arch, device)
    records = synthetic_records(microbatch)
    print(f"student: parameters={model_parameter_count(model):,}")

    step_counter = [0]

    def step():
        batch = prepare_batch(records, device, augment=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, _ = training_loss(outputs, batch, student_arch, step_counter[0])
        loss.backward()
        optimizer.step()
        step_counter[0] += 1

    rate = time_steps("student", step, warmup, steps, microbatch)
    project(
        rate,
        distill_config["effective_batch_records"],
        distill_config["minimum_optimizer_steps_per_epoch"],
        distill_config["epochs"],
    )


def main():
    args = parse_args()
    root = json.loads(Path(args.config).read_text())
    device = configure_torch(root["oracle"]["seed"], deterministic=False)
    print(f"device={device}")
    if not args.skip_oracle:
        calibrate_oracle(root, device, args.steps, args.warmup_steps)
    if not args.skip_student:
        calibrate_student(root, device, args.steps, args.warmup_steps)


if __name__ == "__main__":
    main()
