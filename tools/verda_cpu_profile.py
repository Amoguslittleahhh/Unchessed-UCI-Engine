#!/usr/bin/env python3
"""Resolve Verda 4-360 vCPU data-generation profiles from CPU affinity."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def available_vcpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def select_profile(profiles: dict, visible_vcpus: int) -> dict:
    for profile in profiles["profiles"]:
        if profile["vcpus"] == visible_vcpus:
            return profile
    supported = [profile["vcpus"] for profile in profiles["profiles"]]
    raise ValueError(
        f"affinity exposes {visible_vcpus} vCPUs; expected one of Verda sizes {supported}"
    )


def resolve(base: dict, profiles: dict, visible_cpu_ids: list[int]) -> dict:
    profile = select_profile(profiles, len(visible_cpu_ids))
    defaults = profiles["defaults"]
    output = copy.deepcopy(base)
    reserve = int(profile["reserve_vcpus"])
    worker_vcpus = int(profile["vcpus"]) - reserve
    output["cpu"].update(
        {
            "verda_vcpus": int(profile["vcpus"]),
            "requested_cores": worker_vcpus,
            "reserve_cores": reserve,
            "cores_per_worker": int(defaults["cores_per_worker"]),
            "physical_cores_only": bool(defaults["physical_cores_only"]),
            "numa_aware": bool(defaults["numa_aware"]),
            "maximum_memory_fraction": float(defaults["maximum_memory_fraction"]),
            "affinity_cpu_ids": visible_cpu_ids,
        }
    )
    output["teacher"]["threads"] = int(defaults["teacher_threads"])
    output["cpu"]["resolved_workers"] = worker_vcpus // output["cpu"]["cores_per_worker"]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("detect", "resolve"))
    parser.add_argument("--profiles", default="config/verda_cpu_profiles.json")
    parser.add_argument("--base-config", default="config/v5_180core_datagen.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vcpus", type=int)
    args = parser.parse_args()
    cpu_ids = list(range(args.vcpus)) if args.vcpus else available_vcpus()
    if args.command == "detect":
        result: object = {"count": len(cpu_ids), "cpu_ids": cpu_ids}
    else:
        profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
        base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
        if profiles.get("schema") != 1 or base.get("schema") != 1:
            raise SystemExit("CPU profile and datagen config schema 1 required")
        result = resolve(base, profiles, cpu_ids)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
