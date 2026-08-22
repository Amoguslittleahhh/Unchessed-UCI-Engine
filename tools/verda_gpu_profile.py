#!/usr/bin/env python3
"""Resolve a tested Apex v5 Oracle profile for 1-8 homogeneous Verda GPUs."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from pathlib import Path


def deep_update(target: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def parse_nvidia_csv(text: str) -> list[dict]:
    output = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}")
        output.append(
            {"index": int(fields[0]), "name": fields[1], "memory_total_mib": int(fields[2])}
        )
    return output


def detect_gpus() -> list[dict]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_nvidia_csv(completed.stdout)


def oracle_parameter_count(config: dict) -> int:
    d = config["d_model"]
    board_layers = config["board_layers"]
    board_ffn = config["board_ffn"]
    decoder_layers = config["decoder_layers"]
    decoder_ffn = config["decoder_ffn"]
    history_width = config["history_width"]
    embeddings = (13 + 64 + 16 + 9 + 16) * d
    gab = (
        d * config["gab_token_projection"]
        + 64 * config["gab_token_projection"] * config["gab_hidden"]
        + 2 * config["gab_hidden"]
        + board_layers
        * config["gab_hidden"]
        * config["board_heads"]
        * config["gab_templates"]
        + config["gab_templates"] * 64 * 64
    )
    board_block = 4 * d * d + 4 * d + 3 * d * board_ffn + 2 * board_ffn
    board = embeddings + gab + board_layers * board_block + d
    action_embeddings = (64 + 64 + 5) * d
    decoder_block = 8 * d * d + 3 * d * decoder_ffn + 6 * d + 2 * decoder_ffn
    decoder = action_embeddings + decoder_layers * decoder_block
    history = (64 + 64 + 5 + config["history_plies"] + config["time_classes"]) * history_width
    history += 2 * history_width + history_width * d + d
    adapters = 2 * 2 * d * config["policy_adapter_rank"]
    heads = d + 1 + 2 * d + 2 + 3 * d + 3 + config["score_quantiles"] * d + config["score_quantiles"]
    concepts = config["concept_count"] * config["concept_width"] + 2 * d * config["concept_width"]
    return board + decoder + history + adapters + heads + concepts


def select_profile(profiles: dict, gpus: list[dict]) -> dict:
    if not 1 <= len(gpus) <= profiles["maximum_gpus"]:
        raise ValueError(f"GPU count {len(gpus)} is outside supported range 1..{profiles['maximum_gpus']}")
    names = {gpu["name"] for gpu in gpus}
    memories = {gpu["memory_total_mib"] for gpu in gpus}
    if len(names) != 1 or len(memories) != 1:
        raise ValueError("Apex DDP requires homogeneous GPU model and VRAM across the node")
    gpu = gpus[0]
    for profile in profiles["profiles"]:
        if re.search(profile["match"], gpu["name"], re.IGNORECASE):
            if gpu["memory_total_mib"] < profile["minimum_vram_mib"]:
                continue
            return profile
    raise ValueError(f"no checked-in Verda profile matches {gpu['name']} ({gpu['memory_total_mib']} MiB)")


def resolve(base: dict, profiles: dict, gpus: list[dict]) -> dict:
    profile = select_profile(profiles, gpus)
    output = copy.deepcopy(base)
    deep_update(output["hardware"], profile.get("hardware", {}))
    deep_update(output["oracle"], profile.get("oracle", {}))
    output["hardware"]["gpu_count"] = len(gpus)
    output["hardware"]["distributed_backend"] = "nccl" if len(gpus) > 1 else "none"
    output["hardware"]["resolved_gpu_profile"] = profile["id"]
    output["hardware"]["gpu_name"] = gpus[0]["name"]
    output["hardware"]["gpu_memory_mib"] = gpus[0]["memory_total_mib"]
    output["oracle"]["expected_parameters"] = oracle_parameter_count(output["oracle"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("detect", "resolve"))
    parser.add_argument("--profiles", default="config/verda_gpu_profiles.json")
    parser.add_argument("--base-config", default="config/unarchitectured_v1_training.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-name")
    parser.add_argument("--memory-mib", type=int)
    parser.add_argument("--gpus", type=int)
    args = parser.parse_args()
    gpus = (
        [
            {"index": index, "name": args.gpu_name, "memory_total_mib": args.memory_mib}
            for index in range(args.gpus)
        ]
        if args.gpu_name and args.memory_mib and args.gpus
        else detect_gpus()
    )
    if args.command == "detect":
        result: object = gpus
    else:
        profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
        base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
        if profiles.get("schema") != 1 or base.get("schema") != 1:
            raise SystemExit("profile and training config schema 1 required")
        result = resolve(base, profiles, gpus)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
