#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def load(root: Path) -> dict:
    return json.loads((root / "analysis.json").read_text(encoding="utf-8"))


def paired(root: Path) -> dict:
    data = load(root)
    by_arm = {arm: rows for arm, rows in ((arm, [r for r in data["games"] if r["arm"] == arm]) for arm in ("standard", "fusion"))}
    out = {"root": str(root), "arms": {}, "paired": {}}
    for arm, rows in by_arm.items():
        latencies = [r["first_full_ply"] for r in rows if r["first_full_ply"] is not None]
        fusion_reasons = [r for r in rows if r["first_suspect_reason"] == "legacy_accelerated_fusion"]
        out["arms"][arm] = {
            "games": len(rows),
            "full_rate": len(latencies) / len(rows) if rows else None,
            "latencies": latencies,
            "mean_ply": statistics.mean(latencies) if latencies else None,
            "median_ply": statistics.median(latencies) if latencies else None,
            "fusion_trigger_rate": len(fusion_reasons) / len(rows) if rows else 0.0,
            "reasons": data["arms"][arm]["reasons"],
        }
    standard = by_arm["standard"]
    fusion = by_arm["fusion"]
    deltas = []
    for left, right in zip(standard, fusion):
        if left["first_full_ply"] is not None and right["first_full_ply"] is not None:
            deltas.append(right["first_full_ply"] - left["first_full_ply"])
    out["paired"] = {
        "complete_pairs": len(deltas),
        "fusion_minus_standard_ply": deltas,
        "mean_delta_ply": statistics.mean(deltas) if deltas else None,
        "median_delta_ply": statistics.median(deltas) if deltas else None,
        "fusion_earlier_pairs": sum(delta < 0 for delta in deltas),
        "same_pairs": sum(delta == 0 for delta in deltas),
        "fusion_later_pairs": sum(delta > 0 for delta in deltas),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"normal_80ms": paired(args.normal), "fast_30ms": paired(args.fast)}
    (args.output / "latency_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (label, result) in zip(axes, report.items()):
        for arm, color in (("standard", "#4C78A8"), ("fusion", "#F58518")):
            values = result["arms"][arm]["latencies"]
            ax.scatter([arm] * len(values), values, s=28, alpha=0.82, color=color, label=arm.title())
            if values:
                ax.hlines(statistics.mean(values), -0.25 if arm == "standard" else 0.75, 0.25 if arm == "standard" else 1.25, color=color, linewidth=2)
        ax.set_title(label.replace("_", " "))
        ax.set_xticks([0, 1], ["Standard", "Fusion"])
        ax.set_ylabel("First Full confirmation (ply)")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(args.output / "latency_comparison.png", dpi=220)
    plt.close(fig)

    lines = ["# Unarchitectured Metal asymmetric latency summary", "", "This report is derived from real UCI games against Stockfish 16, not simulated observations.", ""]
    for label, result in report.items():
        lines.extend([f"## {label}", "", "| Arm | Games | Full rate | Mean first-Full ply | Median first-Full ply | Fusion-trigger rate |", "|---|---:|---:|---:|---:|---:|"])
        for arm in ("standard", "fusion"):
            row = result["arms"][arm]
            lines.append(f"| {arm} | {row['games']} | {row['full_rate']:.3f} | {row['mean_ply'] if row['mean_ply'] is not None else 'NA'} | {row['median_ply'] if row['median_ply'] is not None else 'NA'} | {row['fusion_trigger_rate']:.3f} |")
        lines.extend(["", f"Paired complete games: {result['paired']['complete_pairs']}; fusion-minus-standard latency mean: {result['paired']['mean_delta_ply']} plies; earlier/same/later pairs: {result['paired']['fusion_earlier_pairs']}/{result['paired']['same_pairs']}/{result['paired']['fusion_later_pairs']}.", ""])
    (args.output / "latency_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
