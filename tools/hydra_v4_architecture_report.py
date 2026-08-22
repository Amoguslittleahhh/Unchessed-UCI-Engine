#!/usr/bin/env python3
"""Calculate Hydra Aegis v4 legal-policy and adaptive-compute budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import hydra_v3_architecture_report as v3


def validate_config(root):
    if root.get("schema") != 4:
        raise ValueError("Hydra Aegis schema 4 required")
    if len(root.get("file_magic", "").encode("ascii")) != 8:
        raise ValueError("file_magic must be exactly eight ASCII bytes")
    if len(root.get("data_magic", "").encode("ascii")) != 8:
        raise ValueError("data_magic must be exactly eight ASCII bytes")
    xt = root["xt_nnue"]
    if not xt.get("exact_delta_oracle") or not xt.get("dirty_update_requires_oracle_equality"):
        raise ValueError("v4 requires an exact hypergraph delta oracle")
    cf = root["chessformer"]
    if cf["policy_action_vocabulary"] != 64 * 64 * cf["promotion_classes"]:
        raise ValueError("policy action vocabulary must preserve five promotion classes")
    if cf["maximum_legal_moves"] != 218:
        raise ValueError("frozen legal action capacity must be 218")
    if len(cf["exit_layers"]) != len(cf["matryoshka_widths"]):
        raise ValueError("elastic exits and widths must be paired")
    data = root["data"]
    calculated_record = (
        data["v3_semantic_prefix_bytes"]
        + 8
        + data["legal_action_bytes"]
        + data["legal_regret_bytes"]
        + 48
    )
    if data["header_bytes"] != 64 or data["record_bytes"] != calculated_record:
        raise ValueError("v4 data ABI width mismatch")
    if data["maximum_legal_moves"] != cf["maximum_legal_moves"]:
        raise ValueError("model/data legal capacities disagree")
    search = root["search_interface"]
    if search["noncandidate_pruning_allowed"] or not search["full_legal_fallback_required"]:
        raise ValueError("statistical candidate sets cannot remove alpha-beta legal fallback")
    loss_sum = sum(root["joint_training"]["loss_weights"].values())
    if abs(loss_sum - 1.0) > 1e-12:
        raise ValueError(f"joint loss weights sum to {loss_sum}, not 1.0")


def chessformer_budget(config):
    if not config.get("shared_nested_exit_heads"):
        raise ValueError("v4 requires shared nested heads across elastic exits")
    tokens = config["tokens"]
    d = config["d_model"]
    layers = config["layers"]
    heads = config["heads"]
    ffn = config["ffn"]
    d1 = config["gab_d1"]
    d2 = config["gab_d2"]
    templates = config["gab_templates"]
    promotion = config["promotion_classes"]
    regret_width = config["legal_regret_width"]
    outputs = config["legal_regret_outputs"]

    embedding_parameters = (
        config["piece_vocabulary"] + tokens + 16 + 9 + 16
    ) * d
    gab_parameters = (
        d * d1
        + tokens * d1 * d2
        + d2
        + d2
        + layers * d2 * heads * templates
        + templates * tokens * tokens
    )
    block_parameters = 4 * d * d + 4 * d + 3 * d * ffn + 2 * ffn
    trunk_parameters = embedding_parameters + gab_parameters + layers * block_parameters + d
    shared_policy_parameters = 3 * d * d + d + promotion
    value_parameters = d * config["value_classes"] + config["value_classes"]
    adapter_parameters = (
        config["private_policy_adapters"]
        * 3
        * 2
        * d
        * config["policy_adapter_rank"]
    )
    history_rows = 2 * config["policy_squares"] + promotion + config["history_plies"] + config["time_classes"]
    history_parameters = history_rows * config["history_width"]
    history_parameters += 2 * config["history_width"]
    history_parameters += config["history_width"] * d + d
    regret_parameters = (
        2 * d * regret_width
        + promotion * regret_width
        + regret_width * outputs
        + outputs
    )
    concept_parameters = config["concept_count"] * config["concept_width"] + 2 * d * config["concept_width"]
    calibration_parameters = config["calibrated_exits"] * (
        config["value_classes"] + config["private_policy_adapters"]
    )
    total = (
        trunk_parameters
        + shared_policy_parameters
        + value_parameters
        + adapter_parameters
        + history_parameters
        + regret_parameters
        + concept_parameters
        + calibration_parameters
    )
    exits = {}
    for exit_layers, model_width in zip(config["exit_layers"], config["matryoshka_widths"]):
        flops = v3.exit_flops(config, exit_layers, model_width)
        regret_macs = (
            2 * tokens * model_width * regret_width
            + config["maximum_legal_moves"] * regret_width * outputs
        )
        flops += 2 * regret_macs
        exits[f"layer_{exit_layers}_width_{model_width}"] = {
            "flops_approx": flops,
            "mflops_approx": round(flops / 1e6, 3),
        }
    dense_policy_dots = config["policy_squares"] ** 2
    return {
        "trunk_parameters": trunk_parameters,
        "shared_policy_parameters": shared_policy_parameters,
        "value_parameters": value_parameters,
        "concept_parameters": concept_parameters,
        "private_adapter_parameters": adapter_parameters,
        "history_adapter_parameters": history_parameters,
        "legal_regret_parameters": regret_parameters,
        "calibration_parameters": calibration_parameters,
        "parameters": total,
        "runtime_int8_bytes_approx": total,
        "exit_budgets": exits,
        "dense_policy_dot_products": dense_policy_dots,
        "legal_only_policy_dot_products": config["maximum_legal_moves"],
        "policy_dot_reduction_factor": dense_policy_dots / config["maximum_legal_moves"],
    }


def data_budget(config):
    return {
        "header_bytes": config["header_bytes"],
        "record_bytes": config["record_bytes"],
        "semantic_prefix_bytes": config["v3_semantic_prefix_bytes"],
        "legal_action_bytes": config["legal_action_bytes"],
        "legal_regret_bytes": config["legal_regret_bytes"],
        "payload_expansion_over_v3": config["record_bytes"] / 160,
        "records_per_gib": (2**30 - config["header_bytes"]) // config["record_bytes"],
    }


def build_report(config):
    validate_config(config)
    return {
        "schema": 4,
        "name": config["name"],
        "xt_nnue": v3.xt_budget(config["xt_nnue"]),
        "chessformer": chessformer_budget(config["chessformer"]),
        "data": data_budget(config["data"]),
        "search_interface": {
            "candidate_set_maximum": config["search_interface"]["candidate_set_maximum"],
            "candidate_minimum_coverage": config["search_interface"][
                "candidate_set_minimum_coverage"
            ],
            "noncandidate_pruning_allowed": False,
            "full_legal_fallback_required": True,
        },
        "joint_loss_weight_sum": sum(config["joint_training"]["loss_weights"].values()),
        "warning": "calculated architecture budget only; not latency, accuracy, NPS, Elo, or SPRT evidence",
    }


def markdown(report):
    xt = report["xt_nnue"]
    cf = report["chessformer"]
    data = report["data"]
    lines = [
        "# Unchessed Hydra Aegis v4 calculated budget",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| XT total runtime | {xt['runtime_total_bytes_approx'] / 2**20:.2f} MiB |",
        f"| XT per-ply state | {xt['incremental_state_bytes_per_ply']:,} bytes |",
        f"| Chessformer parameters | {cf['parameters']:,} |",
        f"| Chessformer int8 target | {cf['runtime_int8_bytes_approx'] / 2**20:.2f} MiB |",
        f"| Legal regret head | {cf['legal_regret_parameters']:,} parameters |",
        f"| Legal-only policy dot reduction | {cf['policy_dot_reduction_factor']:.2f}x |",
        f"| v4 record | {data['record_bytes']:,} bytes + one {data['header_bytes']}-byte shard header |",
        f"| Records per GiB | {data['records_per_gib']:,} |",
        f"| Candidate-set cap | {report['search_interface']['candidate_set_maximum']} moves |",
        "| Noncandidate pruning | forbidden |",
    ]
    for name, values in cf["exit_budgets"].items():
        lines.append(f"| {name} forward | {values['mflops_approx']:.1f} MFLOP |")
    lines.extend(
        [
            "",
            "These are deterministic architecture calculations, not measured latency, accuracy, NPS, Elo, or SPRT evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/unchessed_hydra_v4.json")
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
