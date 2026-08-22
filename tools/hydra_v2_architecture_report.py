#!/usr/bin/env python3
"""Calculate Hydra Aegis v2 memory and adaptive-compute budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def shallow_head_parameters(input_width, l1, l2, outputs=1):
    return input_width * l1 + l1 + (2 * l1) * l2 + l2 + l2 * outputs + outputs


def xt_budget(config):
    position_bytes = (
        config["position_features"]
        * config["position_width"]
        * config["position_storage_bits"]
        // 8
    )
    direct_bytes = (
        config["direct_threat_dimensions"]
        * config["direct_threat_width"]
        * config["relation_storage_bits"]
        // 8
    )
    xray_bytes = (
        config["xray_hyperedge_dimensions"]
        * config["xray_hyperedge_width"]
        * config["relation_storage_bits"]
        // 8
    )
    pawn_bytes = (
        config["pawn_topology_dimensions"]
        * config["pawn_topology_width"]
        * config["relation_storage_bits"]
        // 8
    )
    fast_input = 4 * config["position_width"]
    relation_width = (
        config["direct_threat_width"]
        + config["xray_hyperedge_width"]
        + config["pawn_topology_width"]
    )
    full_semantic_input = fast_input + 2 * relation_width + 6
    full_padded_input = ((full_semantic_input + 31) // 32) * 32
    fast_head = shallow_head_parameters(
        fast_input,
        config["fast_head_l1"],
        config["head_l2"],
        1 + config["fast_uncertainty_outputs"],
    )
    full_head = shallow_head_parameters(
        full_padded_input, config["full_head_l1"], config["head_l2"]
    )
    head_parameters = config["phase_stacks"] * (fast_head + full_head)
    state_width = config["position_width"] + relation_width
    state_bytes = 2 * state_width * 2
    total = position_bytes + direct_bytes + xray_bytes + pawn_bytes + head_parameters
    return {
        "runtime_position_bytes": position_bytes,
        "runtime_direct_threat_bytes": direct_bytes,
        "runtime_xray_bytes": xray_bytes,
        "runtime_pawn_topology_bytes": pawn_bytes,
        "fast_head_input": fast_input,
        "full_semantic_input": full_semantic_input,
        "full_padded_input": full_padded_input,
        "head_parameters": head_parameters,
        "incremental_state_bytes_per_ply": state_bytes,
        "runtime_total_bytes_approx": total,
    }


def transformer_core_parameters(config):
    tokens = config["tokens"]
    d = config["d_model"]
    layers = config["layers"]
    heads = config["heads"]
    ffn = config["ffn"]
    templates = config["gab_templates"]
    d1 = config["gab_d1"]
    d2 = config["gab_d2"]
    embeddings = (config["piece_vocabulary"] + tokens + 16 + 9) * d + d * d + 2 * d
    gab = (
        d * d1
        + tokens * d1 * d2
        + 2 * d2
        + layers * d2 * heads * templates
        + templates * tokens * tokens
    )
    block = 4 * d * d + 4 * d + 3 * d * ffn + 2 * ffn
    final_heads = 3 * d * d + 3 * d + d * config["promotion_classes"]
    final_heads += config["promotion_classes"] + d * config["value_classes"]
    final_heads += config["value_classes"]
    return embeddings + gab + layers * block + final_heads


def exit_flops(config, layers, width):
    tokens = config["tokens"]
    heads = config["heads"]
    templates = config["gab_templates"]
    d1 = config["gab_d1"]
    d2 = config["gab_d2"]
    ffn = width
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
    # Project body/source/target, then score only legal moves at inference.
    policy_macs = 3 * tokens * width * width
    policy_macs += config["maximum_legal_moves"] * width
    value_macs = tokens * width + width * config["value_classes"]
    return 2 * (layers * block_macs + gab_macs + policy_macs + value_macs)


def chessformer_budget(config):
    base_parameters = transformer_core_parameters(config)
    d = config["d_model"]
    intermediate_exits = len(config["exit_layers"]) - 1
    exit_parameters = intermediate_exits * (
        3 * d * d + 3 * d + d * config["promotion_classes"]
        + config["promotion_classes"] + d * config["value_classes"]
        + config["value_classes"]
    )
    concept_parameters = (
        config["concept_count"] * config["concept_width"]
        + 2 * d * config["concept_width"]
    )
    total = base_parameters + exit_parameters + concept_parameters
    exits = {}
    for layers, width in zip(config["exit_layers"], config["matryoshka_widths"]):
        exits[f"layer_{layers}_width_{width}"] = {
            "flops_approx": exit_flops(config, layers, width),
            "mflops_approx": round(exit_flops(config, layers, width) / 1e6, 3),
        }
    dense_policy_dots = config["policy_squares"] ** 2
    return {
        "base_parameters": base_parameters,
        "intermediate_exit_parameters": exit_parameters,
        "concept_parameters": concept_parameters,
        "parameters": total,
        "runtime_int8_bytes_approx": total,
        "exit_budgets": exits,
        "dense_policy_dot_products": dense_policy_dots,
        "legal_only_policy_dot_products": config["maximum_legal_moves"],
        "policy_dot_reduction_factor": dense_policy_dots / config["maximum_legal_moves"],
    }


def markdown(report):
    xt = report["xt_nnue"]
    cf = report["chessformer"]
    lines = [
        "# Unchessed Hydra Aegis v2 calculated budget",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| XT total runtime | {xt['runtime_total_bytes_approx'] / 2**20:.2f} MiB |",
        f"| XT per-ply state | {xt['incremental_state_bytes_per_ply']:,} bytes |",
        f"| Chessformer parameters | {cf['parameters']:,} |",
        f"| Chessformer int8 target | {cf['runtime_int8_bytes_approx'] / 2**20:.2f} MiB |",
        f"| Legal-only policy dot reduction | {cf['policy_dot_reduction_factor']:.2f}x |",
    ]
    for name, values in cf["exit_budgets"].items():
        lines.append(f"| {name} forward | {values['mflops_approx']:.1f} MFLOP |")
    lines.extend(
        [
            "",
            "These are deterministic architecture calculations, not measured latency, accuracy, or Elo.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unchessed_hydra_v2.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("schema") != 2:
        raise SystemExit("Hydra Aegis schema 2 required")
    report = {
        "schema": 2,
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
