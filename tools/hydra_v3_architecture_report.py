#!/usr/bin/env python3
"""Calculate Hydra Aegis v3 memory, data, and adaptive-compute budgets.

Every number emitted here is an architecture calculation.  This tool does not
claim measured latency, validation accuracy, playing strength, or Elo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def shallow_head_parameters(input_width, l1, l2, outputs=1):
    """Parameters in Linear(input,l1), dual SCReLU/CReLU, Linear(2*l1,l2), output."""
    return input_width * l1 + l1 + (2 * l1) * l2 + l2 + l2 * outputs + outputs


def validate_config(root):
    if root.get("schema") != 3:
        raise ValueError("Hydra Aegis schema 3 required")
    if len(root.get("file_magic", "").encode("ascii")) != 8:
        raise ValueError("file_magic must be exactly eight ASCII bytes")
    if len(root.get("data_magic", "").encode("ascii")) != 8:
        raise ValueError("data_magic must be exactly eight ASCII bytes")
    xt = root["xt_nnue"]
    if xt["xray_hyperedge_dimensions"] != 12 * 12 * 12 * 8:
        raise ValueError("x-ray dimensions must encode 12^3 classes x 8 directions")
    if xt["direct_threat_dimensions"] != 12 * 12 * 15 * 15:
        raise ValueError("direct dimensions must encode 12^2 classes x 15^2 deltas")
    cf = root["chessformer"]
    exits = cf["exit_layers"]
    widths = cf["matryoshka_widths"]
    if len(exits) != len(widths) or exits != sorted(exits) or widths != sorted(widths):
        raise ValueError("elastic exits and widths must be paired and monotonic")
    if exits[-1] != cf["layers"] or widths[-1] != cf["d_model"]:
        raise ValueError("last elastic exit must be the full model")
    data = root["data"]
    if data["header_bytes"] != 64 or data["record_bytes"] != 160:
        raise ValueError("v3 data ABI is fixed at a 64-byte header and 160-byte records")
    loss_sum = sum(root["joint_training"]["loss_weights"].values())
    if abs(loss_sum - 1.0) > 1e-12:
        raise ValueError(f"joint loss weights sum to {loss_sum}, not 1.0")


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

    # Each position accumulator contributes SCReLU and CReLU channels for both
    # perspectives. Relation groups contribute one clipped channel per side.
    fast_input = 4 * config["position_width"]
    direct_semantic_input = fast_input + 2 * config["direct_threat_width"] + 6
    direct_padded_input = ((direct_semantic_input + 31) // 32) * 32
    full_semantic_input = direct_semantic_input + 2 * (
        config["xray_hyperedge_width"] + config["pawn_topology_width"]
    )
    full_padded_input = ((full_semantic_input + 31) // 32) * 32

    fast_one = shallow_head_parameters(
        fast_input,
        config["fast_head_l1"],
        config["head_l2"],
        config["fast_outputs_per_member"],
    )
    fast_heads = config["fast_ensemble_members"] * fast_one
    direct_head = shallow_head_parameters(
        direct_padded_input,
        config["direct_head_l1"],
        config["head_l2"],
        1 + config["direct_uncertainty_outputs"],
    )
    full_head = shallow_head_parameters(
        full_padded_input,
        config["full_head_l1"],
        config["head_l2"],
    )
    head_parameters = config["phase_stacks"] * (fast_heads + direct_head + full_head)
    relation_width = (
        config["direct_threat_width"]
        + config["xray_hyperedge_width"]
        + config["pawn_topology_width"]
    )
    state_bytes = 2 * (config["position_width"] + relation_width) * 2
    calibration_bytes = (
        config["phase_stacks"]
        * config["conformal_depth_buckets"]
        * config["conformal_tail_bounds"]
        * 4
    )
    total = (
        position_bytes
        + direct_bytes
        + xray_bytes
        + pawn_bytes
        + head_parameters
        + calibration_bytes
    )
    return {
        "runtime_position_bytes": position_bytes,
        "runtime_direct_threat_bytes": direct_bytes,
        "runtime_xray_bytes": xray_bytes,
        "runtime_pawn_topology_bytes": pawn_bytes,
        "fast_head_input": fast_input,
        "direct_semantic_input": direct_semantic_input,
        "direct_padded_input": direct_padded_input,
        "full_semantic_input": full_semantic_input,
        "full_padded_input": full_padded_input,
        "head_parameters": head_parameters,
        "calibration_bytes": calibration_bytes,
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
    policy_macs = 3 * tokens * width * width
    policy_macs += config["maximum_legal_moves"] * width
    value_macs = tokens * width + width * config["value_classes"]
    rank = config["policy_adapter_rank"]
    adapter_macs = 3 * 2 * tokens * width * rank
    history_macs = config["history_width"] * width
    return 2 * (
        layers * block_macs
        + gab_macs
        + policy_macs
        + value_macs
        + adapter_macs
        + history_macs
    )


def chessformer_budget(config):
    base_parameters = transformer_core_parameters(config)
    d = config["d_model"]
    intermediate_exits = len(config["exit_layers"]) - 1
    one_exit = (
        3 * d * d
        + 3 * d
        + d * config["promotion_classes"]
        + config["promotion_classes"]
        + d * config["value_classes"]
        + config["value_classes"]
    )
    exit_parameters = intermediate_exits * one_exit
    concept_parameters = (
        config["concept_count"] * config["concept_width"]
        + 2 * d * config["concept_width"]
    )
    adapter_parameters = (
        config["private_policy_adapters"]
        * 3
        * 2
        * d
        * config["policy_adapter_rank"]
    )
    history_embedding_rows = (
        2 * config["policy_squares"]
        + config["promotion_classes"]
        + config["time_classes"]
        + config["history_plies"]
    )
    history_parameters = history_embedding_rows * config["history_width"]
    history_parameters += config["history_width"] * d + d
    calibration_parameters = config["calibrated_exits"] * (
        config["value_classes"] + config["private_policy_adapters"]
    )
    total = (
        base_parameters
        + exit_parameters
        + concept_parameters
        + adapter_parameters
        + history_parameters
        + calibration_parameters
    )
    exits = {}
    for layers, width in zip(config["exit_layers"], config["matryoshka_widths"]):
        flops = exit_flops(config, layers, width)
        exits[f"layer_{layers}_width_{width}"] = {
            "flops_approx": flops,
            "mflops_approx": round(flops / 1e6, 3),
        }
    dense_policy_dots = config["policy_squares"] ** 2
    return {
        "base_parameters": base_parameters,
        "intermediate_exit_parameters": exit_parameters,
        "concept_parameters": concept_parameters,
        "private_adapter_parameters": adapter_parameters,
        "history_adapter_parameters": history_parameters,
        "calibration_parameters": calibration_parameters,
        "parameters": total,
        "runtime_int8_bytes_approx": total,
        "exit_budgets": exits,
        "dense_policy_dot_products": dense_policy_dots,
        "legal_only_policy_dot_products": config["maximum_legal_moves"],
        "policy_dot_reduction_factor": dense_policy_dots / config["maximum_legal_moves"],
    }


def data_budget(config):
    required = (
        "contains_promotion_identity",
        "contains_wdl",
        "contains_game_hash",
        "contains_player_hash",
        "contains_time_class",
        "contains_teacher_regret",
    )
    if not all(config[key] for key in required):
        raise ValueError("v3 records must carry every mandatory semantic field")
    return {
        "header_bytes": config["header_bytes"],
        "record_bytes": config["record_bytes"],
        "payload_expansion_over_v2": config["record_bytes"] / 104,
        "records_per_gib": (2**30 - config["header_bytes"]) // config["record_bytes"],
    }


def markdown(report):
    xt = report["xt_nnue"]
    cf = report["chessformer"]
    data = report["data"]
    lines = [
        "# Unchessed Hydra Aegis v3 calculated budget",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| XT total runtime | {xt['runtime_total_bytes_approx'] / 2**20:.2f} MiB |",
        f"| XT per-ply state | {xt['incremental_state_bytes_per_ply']:,} bytes |",
        f"| XT conformal calibration | {xt['calibration_bytes']:,} bytes |",
        f"| Chessformer parameters | {cf['parameters']:,} |",
        f"| Chessformer int8 target | {cf['runtime_int8_bytes_approx'] / 2**20:.2f} MiB |",
        f"| Private policy adapters | {cf['private_adapter_parameters']:,} parameters |",
        f"| History adapter | {cf['history_adapter_parameters']:,} parameters |",
        f"| Legal-only policy dot reduction | {cf['policy_dot_reduction_factor']:.2f}x |",
        f"| v3 record | {data['record_bytes']} bytes + one {data['header_bytes']}-byte shard header |",
        f"| Records per GiB | {data['records_per_gib']:,} |",
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


def build_report(config):
    validate_config(config)
    return {
        "schema": 3,
        "name": config["name"],
        "xt_nnue": xt_budget(config["xt_nnue"]),
        "chessformer": chessformer_budget(config["chessformer"]),
        "data": data_budget(config["data"]),
        "joint_loss_weight_sum": sum(config["joint_training"]["loss_weights"].values()),
        "warning": "calculated architecture budget only; not latency, accuracy, or Elo",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unchessed_hydra_v3.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = build_report(config)
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
