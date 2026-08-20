#!/usr/bin/env python3
"""Shared zero-dependency-at-import helpers for A100 training scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path, section: str) -> tuple[dict, dict]:
    root = json.loads(Path(path).read_text(encoding="utf-8"))
    if root.get("schema") != 1 or section not in root:
        raise ValueError(f"invalid A100 config or missing section {section!r}")
    return root[section], root.get("hardware", {})


def configure_torch(seed: int, deterministic: bool = False):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))


class FixedRecordShards:
    """Memory-mapped fixed-record shards with random global-index gathering."""

    def __init__(self, paths, dtype):
        import numpy as np

        self.paths = [Path(path) for path in paths]
        self.dtype = dtype
        self.shards = []
        self.active_paths = []
        self.counts = []
        for path in self.paths:
            size = path.stat().st_size
            if size % dtype.itemsize:
                raise ValueError(f"{path}: size is not a multiple of {dtype.itemsize}")
            count = size // dtype.itemsize
            if count:
                self.shards.append(np.memmap(path, dtype=dtype, mode="r"))
                self.active_paths.append(path)
                self.counts.append(count)
        if not self.shards:
            raise ValueError("no non-empty shards")
        self.cumulative = np.cumsum(self.counts)
        self.total = int(self.cumulative[-1])

    def sample(self, rng, count):
        import numpy as np

        global_indices = rng.integers(0, self.total, count, endpoint=False)
        shard_ids = np.searchsorted(self.cumulative, global_indices, side="right")
        previous = np.where(shard_ids == 0, 0, self.cumulative[shard_ids - 1])
        local_indices = global_indices - previous
        output = np.empty(count, dtype=self.dtype)
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            output[mask] = self.shards[int(shard_id)][local_indices[mask]]
        return output

    def sequential_batches(self, batch_size, max_records=None):
        emitted = 0
        for shard in self.shards:
            for start in range(0, len(shard), batch_size):
                if max_records is not None and emitted >= max_records:
                    return
                end = min(start + batch_size, len(shard))
                if max_records is not None:
                    end = min(end, start + max_records - emitted)
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
            }
            for path, count in zip(self.active_paths, self.counts)
        ]


def learning_rate(step, total_steps, base_lr, warmup_steps, minimum_ratio=0.05):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def atomic_torch_save(payload, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def numpy_to_device(array, device):
    """Transfer contiguous NumPy data through pinned host memory on CUDA."""
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=device.type == "cuda")


def model_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
