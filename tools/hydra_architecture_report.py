#!/usr/bin/env python3
"""Compute parameter, memory, and operation budgets for Unchessed Hydra v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def xt_budget(config):
    position_features = config["position_features"]
    position_width = config["position_width"]
    threat_dimensions = (
        config["threat_attacker_classes"]
        * config["threat_target_classes"]
        * config["threat_relations"]
    )
    threat_width = config["threat_latent_width"]
    rank = config["threat_cp_rank"]
    factor_parameters = rank * (
        config["threat_attacker_classes"]
        + config["threat_target_classes"]
        + config["threat_relations"]
        + threat_width
    )
    residual_parameters = config["threat_residual_buckets"] * threat_width
    head_per_stack = (
        config["head_input"] * config["head_l1"]
        + config["head_l1"]
        + (2 * config["head_l1"]) * config["head_l2"]
        + config["head_l2"]
        + config["head_l2"]
        + 1
    )
    position_bytes = (
        position_features * position_width * config["position_storage_bits"] // 8
    )
    threat_bytes = (
        threat_dimensions * threat_width * config["threat_export_bits"] // 8
    )
    head_bytes = head_per_stack * config["phase_stacks"]
    state_bytes = 2 * (position_width + threat_width) * 2
    return {
        "threat_dimensions": threat_dimensions,
        "training_factor_parameters": factor_parameters,
        "training_hashed_residual_parameters": residual_parameters,
        "materialized_threat_parameters": threat_dimensions * threat_width,
        "head_parameters": head_per_stack * config["phase_stacks"],
        "runtime_position_bytes": position_bytes,
        "runtime_threat_bytes": threat_bytes,
        "runtime_head_bytes_approx": head_bytes,
        "runtime_total_bytes_approx": position_bytes + threat_bytes + head_bytes,
        "incremental_state_bytes_per_ply": state_bytes,
        "average_relation_adds_at_38_6": round(38.6 * threat_width),
    }


def chessformer_budget(config):
    tokens = config["tokens"]
    width = config["d_model"]
    layers = config["layers"]
    heads = config["heads"]
    ffn = config["ffn"]
    templates = config["gab_templates"]
    d1 = config["gab_d1"]
    d2 = config["gab_d2"]

    embeddings = (
        config["piece_vocabulary"] * width
        + tokens * width
        + 16 * width
        + 9 * width
        + width * width
        + 2 * width
    )
    gab = (
        width * d1
        + tokens * d1 * d2
        + d2
        + d2
        + layers * d2 * heads * templates
        + templates * tokens * tokens
    )
    block = (
        3 * width * width
        + width * width
        + width
        + 2 * width
        + width * 2 * ffn
        + 2 * ffn
        + ffn * width
        + width
    )
    heads_parameters = (
        width
        + width * width
        + width
        + width * width
        + width * width
        + width * config["promotion_classes"]
        + config["promotion_classes"]
        + width * config["value_classes"]
        + config["value_classes"]
    )
    parameters = embeddings + gab + layers * block + heads_parameters

    block_macs = (
        4 * tokens * width * width
        + 3 * tokens * width * ffn
        + 2 * tokens * tokens * width
    )
    gab_macs = (
        tokens * width * d1
        + tokens * d1 * d2
        + layers * d2 * heads * templates
        + layers * heads * templates * tokens * tokens
    )
    policy_macs = 3 * tokens * width * width + tokens * tokens * width
    value_macs = tokens * width + width * config["value_classes"]
    macs = layers * block_macs + gab_macs + policy_macs + value_macs
    return {
        "parameters": parameters,
        "runtime_int8_bytes_approx": parameters,
        "training_bf16_parameter_bytes": 2 * parameters,
        "root_forward_macs_approx": macs,
        "root_forward_flops_approx": 2 * macs,
        "attention_matrix_elements_per_layer": heads * tokens * tokens,
    }


def markdown(report):
    xt = report["xt_nnue"]
    cf = report["chessformer"]
    return "\n".join(
        [
            "# Unchessed Hydra v1 architecture budget",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| XT positional table | {xt['runtime_position_bytes'] / 2**20:.2f} MiB |",
            f"| XT materialized int8 threat table | {xt['runtime_threat_bytes'] / 2**20:.2f} MiB |",
            f"| XT total runtime parameters | {xt['runtime_total_bytes_approx'] / 2**20:.2f} MiB |",
            f"| XT state per search ply | {xt['incremental_state_bytes_per_ply']:,} bytes |",
            f"| Chessformer parameters | {cf['parameters']:,} |",
            f"| Chessformer int8 runtime target | {cf['runtime_int8_bytes_approx'] / 2**20:.2f} MiB |",
            f"| Chessformer root forward | {cf['root_forward_flops_approx'] / 1e6:.1f} MFLOP |",
            "",
            "These are architecture calculations, not measured latency, accuracy, or Elo.",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unchessed_hydra_v1.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise SystemExit("unsupported architecture schema")
    report = {
        "schema": 1,
        "name": config["name"],
        "xt_nnue": xt_budget(config["xt_nnue"]),
        "chessformer": chessformer_budget(config["chessformer"]),
        "joint_loss_weight_sum": sum(config["joint_training"]["loss_weights"].values()),
        "warning": "calculated architecture budget only; not latency, accuracy, or Elo",
    }
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown(report)
    if args.check:
        if args.json is None or args.markdown is None:
            raise SystemExit("--check requires --json and --markdown")
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
