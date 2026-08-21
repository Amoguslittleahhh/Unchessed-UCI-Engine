#!/usr/bin/env python3
"""Calculate Hydra Apex v5 offline-oracle and runtime-student budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import hydra_v4_architecture_report as v4
import verda_gpu_profile as gpu_profiles


def validate_config(config):
    if config.get("schema") != 5:
        raise ValueError("Hydra Apex schema 5 required")
    oracle = config["offline_oracle"]
    if oracle["runtime_deployed"]:
        raise ValueError("the large oracle must remain training-only")
    if oracle["maximum_legal_tokens"] != 218 or oracle["promotion_classes"] != 5:
        raise ValueError("oracle must consume the exact promotion-aware legal set")
    if oracle["d_model"] % oracle["board_heads"]:
        raise ValueError("oracle width must divide across board heads")
    if oracle["d_model"] % oracle["legal_decoder_heads"]:
        raise ValueError("oracle width must divide across legal decoder heads")
    distillation = config["distillation"]
    weights = [value for key, value in distillation.items() if key not in ("oracle_policy_temperature", "oracle_wdl_temperature", "sum")]
    if abs(sum(weights) - 1.0) > 1e-12 or distillation["sum"] != 1.0:
        raise ValueError("v5 distillation weights must sum to one")
    cpu = config["data_generation"]
    worker_vcpus = cpu["default_workers"] * cpu["threads_per_worker"]
    if worker_vcpus + cpu["reserved_service_vcpus"] != cpu["default_node_vcpus"]:
        raise ValueError("default worker and service vCPUs must cover the selected Verda node")
    gpu = config["a100_training"]
    if gpu["fp8_allowed"]:
        raise ValueError("A100 training must not claim unsupported FP8 execution")
    if abs(gpu["target_vram_fraction"] + gpu["minimum_free_vram_fraction"] - 1.0) > 1e-12:
        raise ValueError("VRAM target and safety reserve must sum to one")
    safety = config["safety"]
    if safety["noncandidate_pruning_allowed"] or not safety["full_legal_fallback_required"]:
        raise ValueError("oracle policy cannot remove alpha-beta legal fallback")


def oracle_budget(config):
    board_tokens = config["board_tokens"]
    legal_tokens = config["maximum_legal_tokens"]
    d = config["d_model"]
    ffn = config["board_ffn"]
    layers = config["board_layers"]
    decoder_layers = config["legal_decoder_layers"]
    decoder_ffn = config["legal_decoder_ffn"]
    promotion = config["promotion_classes"]
    history_width = config["history_width"]
    concept_count = config["concept_count"]
    concept_width = config["concept_width"]

    board_embeddings = (config["piece_vocabulary"] + board_tokens + 16 + 9 + 16) * d
    gab = (
        d * config["gab_d1"]
        + board_tokens * config["gab_d1"] * config["gab_d2"]
        + 2 * config["gab_d2"]
        + layers * config["gab_d2"] * config["board_heads"] * config["gab_templates"]
        + config["gab_templates"] * board_tokens * board_tokens
    )
    board_block = 4 * d * d + 4 * d + 3 * d * ffn + 2 * ffn
    board_trunk = board_embeddings + gab + layers * board_block + d

    action_embeddings = (64 + 64 + promotion) * d
    decoder_block = 8 * d * d + 3 * d * decoder_ffn + 6 * d + 2 * decoder_ffn
    decoder = action_embeddings + decoder_layers * decoder_block

    history_rows = 2 * 64 + promotion + config["history_plies"] + config["time_classes"]
    history = history_rows * history_width + 2 * history_width + history_width * d + d
    adapters = config["private_policy_adapters"] * 2 * d * config["policy_adapter_rank"]
    heads = (
        d + 1
        + d * config["regret_outputs"] + config["regret_outputs"]
        + d * config["value_classes"] + config["value_classes"]
        + d * config["score_quantiles"] + config["score_quantiles"]
    )
    concepts = concept_count * concept_width + 2 * d * concept_width
    parameters = board_trunk + decoder + history + adapters + heads + concepts

    board_block_macs = (
        4 * board_tokens * d * d
        + 3 * board_tokens * d * ffn
        + 2 * board_tokens * board_tokens * d
    )
    board_flops = 2 * layers * board_block_macs
    decoder_block_macs = (
        4 * legal_tokens * d * d
        + 2 * legal_tokens * legal_tokens * d
        + 2 * legal_tokens * d * d
        + 2 * board_tokens * d * d
        + 2 * legal_tokens * board_tokens * d
        + 3 * legal_tokens * d * decoder_ffn
    )
    decoder_flops = 2 * decoder_layers * decoder_block_macs
    return {
        "board_trunk_parameters": board_trunk,
        "legal_decoder_parameters": decoder,
        "history_parameters": history,
        "private_adapter_parameters": adapters,
        "head_parameters": heads,
        "concept_parameters": concepts,
        "parameters": parameters,
        "bf16_weight_bytes": parameters * 2,
        "fp32_adamw_persistent_bytes_approx": parameters * 16,
        "board_forward_flops_approx": board_flops,
        "legal_decoder_forward_flops_approx": decoder_flops,
        "forward_flops_approx": board_flops + decoder_flops,
    }


def verda_profile_matrix(profiles, training_config):
    matrix = []
    for profile in profiles["profiles"]:
        resolved = json.loads(json.dumps(training_config))
        gpu_profiles.deep_update(resolved["hardware"], profile.get("hardware", {}))
        gpu_profiles.deep_update(resolved["oracle"], profile.get("oracle", {}))
        matrix.append(
            {
                "id": profile["id"],
                "match": profile["match"],
                "minimum_vram_mib": profile["minimum_vram_mib"],
                "precision": resolved["hardware"]["precision"],
                "oracle_parameters": gpu_profiles.oracle_parameter_count(
                    resolved["oracle"]
                ),
                "minimum_training_records": (
                    resolved["oracle"]["minimum_optimizer_steps_per_epoch"]
                    * resolved["oracle"]["effective_batch_records"]
                ),
            }
        )
    return matrix


def build_report(config, v4_config, profiles=None, training_config=None):
    validate_config(config)
    oracle = oracle_budget(config["offline_oracle"])
    student = v4.chessformer_budget(v4_config["chessformer"])
    report = {
        "schema": 5,
        "name": config["name"],
        "offline_oracle": oracle,
        "runtime_student": student,
        "parameter_compression_factor": oracle["parameters"] / student["parameters"],
        "cpu_datagen": {
            "node_vcpus": config["data_generation"]["default_node_vcpus"],
            "reserved_service_vcpus": config["data_generation"]["reserved_service_vcpus"],
            "workers": config["data_generation"]["default_workers"],
            "threads_per_worker": config["data_generation"]["threads_per_worker"],
            "aggregate_hash_mib": config["data_generation"]["default_workers"]
            * config["data_generation"]["teacher_hash_mb_per_worker"],
            "nodes_per_action": config["data_generation"]["default_nodes_per_action"],
        },
        "a100": config["a100_training"],
        "warning": "calculated architecture/memory/operation budgets only; not measured throughput, accuracy, NPS, Elo, or SPRT evidence",
    }
    if profiles is not None and training_config is not None:
        report["verda_profiles"] = verda_profile_matrix(profiles, training_config)
    return report


def markdown(report):
    oracle = report["offline_oracle"]
    student = report["runtime_student"]
    cpu = report["cpu_datagen"]
    lines = [
        "# Unchessed Hydra Apex v5 calculated budget",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Offline oracle parameters | {oracle['parameters']:,} |",
        f"| Oracle BF16 weights | {oracle['bf16_weight_bytes'] / 2**20:.2f} MiB |",
        f"| Oracle FP32 AdamW persistent state | {oracle['fp32_adamw_persistent_bytes_approx'] / 2**30:.2f} GiB |",
        f"| Oracle maximum legal-set forward | {oracle['forward_flops_approx'] / 1e9:.2f} GFLOP |",
        f"| Runtime student parameters | {student['parameters']:,} |",
        f"| Oracle/student parameter ratio | {report['parameter_compression_factor']:.2f}x |",
        f"| CPU teacher workers | {cpu['workers']} x {cpu['threads_per_worker']} thread on {cpu['node_vcpus']} vCPUs |",
        f"| Reserved CPU service vCPUs | {cpu['reserved_service_vcpus']} |",
        f"| Aggregate configured teacher hash | {cpu['aggregate_hash_mib'] / 1024:.2f} GiB |",
        f"| Exact teacher nodes per legal action | {cpu['nodes_per_action']:,} |",
        f"| Base-profile target VRAM occupancy | {report['a100']['target_vram_fraction'] * 100:.0f}% |",
        f"| Base-profile reserved free VRAM | {report['a100']['minimum_free_vram_fraction'] * 100:.0f}% |",
    ]
    if report.get("verda_profiles"):
        lines.extend(
            [
                "",
                "| Resolved Verda profile | Minimum VRAM | Precision | Oracle parameters | Minimum records |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for profile in report["verda_profiles"]:
            lines.append(
                f"| {profile['id']} | {profile['minimum_vram_mib']:,} MiB | "
                f"{profile['precision']} | {profile['oracle_parameters']:,} | "
                f"{profile['minimum_training_records']:,} |"
            )
    lines.extend(
        [
            "",
            "The oracle is training-only. These are calculations, not measured hardware throughput, model accuracy, NPS, Elo, or SPRT evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unchessed_hydra_v5.json")
    parser.add_argument("--student-config", default="config/unchessed_hydra_v4.json")
    parser.add_argument("--profiles", default="config/verda_gpu_profiles.json")
    parser.add_argument("--training-config", default="config/a100_hydra_v5_training.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    student_config = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    training_config = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    report = build_report(config, student_config, profiles, training_config)
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown(report)
    if args.check:
        if args.json is None or args.markdown is None:
            raise SystemExit("--check requires output paths")
        if args.json.read_text(encoding="utf-8") != json_text:
            raise SystemExit(f"generated output differs: {args.json}")
        if args.markdown.read_text(encoding="utf-8") != markdown_text:
            raise SystemExit(f"generated output differs: {args.markdown}")
    else:
        print(json_text, end="")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json_text, encoding="utf-8")
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown_text, encoding="utf-8")


if __name__ == "__main__":
    main()
