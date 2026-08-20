#!/usr/bin/env python3
"""Verify Verda CPU/GPU hosts before Apex v5 data generation or training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from v5_180core_datagen import available_cores, numa_groups, physical_core_representatives


def memory_info() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(value.split()[0]) * 1024
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
    }


def mount_info(path: str | Path) -> dict:
    path = Path(path).resolve()
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    candidates = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3:
            source, mount, filesystem = fields[:3]
            try:
                path.relative_to(mount)
                candidates.append((len(mount), source, mount, filesystem))
            except ValueError:
                pass
    _, source, mount, filesystem = max(candidates, default=(0, "unknown", "/", "unknown"))
    return {
        "path": str(path),
        "source": source,
        "mount": mount,
        "filesystem": filesystem,
        "total_bytes": usage.total,
        "free_bytes": usage.free,
    }


def parse_nvidia_csv(text: str) -> list[dict]:
    output = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi CSV line: {line!r}")
        output.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "driver_version": fields[3],
            }
        )
    return output


def gpu_info() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_nvidia_csv(completed.stdout)


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        ),
    }


SUPPORTED_GPU = re.compile(
    r"GB300|B300|B200|H200|H100|A100|V100|RTX PRO 6000|L40S|RTX 6000 Ada|RTX A6000",
    re.IGNORECASE,
)


def inspect(
    role: str,
    data_path: str | Path,
    expected_physical_cores: int,
    expected_gpus: int = 1,
    expected_logical_cpus: int = 0,
) -> dict:
    logical = available_cores()
    physical = physical_core_representatives(logical)
    report = {
        "role": role,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "logical_affinity_cpus": logical,
        "logical_affinity_count": len(logical),
        "physical_core_representatives": physical,
        "physical_core_count": len(physical),
        "numa_groups": numa_groups(logical),
        "memory": memory_info(),
        "data_mount": mount_info(data_path),
        "checks": {},
    }
    if role == "cpu":
        if expected_logical_cpus:
            report["checks"]["logical_affinity_cpus"] = (
                len(logical) >= expected_logical_cpus
            )
        if expected_physical_cores:
            report["checks"]["physical_cores"] = (
                len(physical) >= expected_physical_cores
            )
    else:
        try:
            report["gpus"] = gpu_info()
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
            report["gpus"] = []
            report["gpu_error"] = str(error)
        report["torch"] = torch_info()
        report["checks"]["gpu_count"] = len(report["gpus"]) == expected_gpus
        report["checks"]["supported_gpu_family"] = bool(report["gpus"]) and all(
            SUPPORTED_GPU.search(gpu["name"]) for gpu in report["gpus"]
        )
        report["checks"]["homogeneous_gpus"] = bool(report["gpus"]) and len(
            {(gpu["name"], gpu["memory_total_mib"]) for gpu in report["gpus"]}
        ) == 1
    report["checks"]["data_path_exists"] = Path(data_path).exists()
    report["checks"]["data_free_100gb"] = report["data_mount"]["free_bytes"] >= 100 * 2**30
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--data-path", default="/data")
    parser.add_argument("--expected-physical-cores", type=int, default=0)
    parser.add_argument("--expected-logical-cpus", type=int, default=0)
    parser.add_argument("--expected-gpus", type=int, default=1)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    report = inspect(
        args.role,
        args.data_path,
        args.expected_physical_cores,
        args.expected_gpus,
        args.expected_logical_cpus,
    )
    if args.require_torch:
        gpu_name = report.get("gpus", [{}])[0].get("name", "") if report.get("gpus") else ""
        precision_supported = report.get("torch", {}).get("bf16_supported") or "V100" in gpu_name
        report["checks"]["torch_cuda_training_precision"] = bool(
            report.get("torch", {}).get("cuda_available") and precision_supported
        )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not all(report["checks"].values()):
        raise SystemExit("Verda preflight failed one or more required checks")


if __name__ == "__main__":
    main()
