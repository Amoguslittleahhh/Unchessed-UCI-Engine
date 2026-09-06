#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def parse_log(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "info string [UnchessedTelemetry] "
        if marker not in raw:
            continue
        rows.append(dict(FIELD_RE.findall(raw.split(marker, 1)[1])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "unarchitectured-metal-asymmetric-latency")
    args = parser.parse_args()
    root = args.root
    summaries = []
    for arm in ("standard", "fusion"):
        for path in sorted((root / arm).glob("game_*_unchessed.log")):
            rows = parse_log(path)
            obs = [r for r in rows if r.get("event") == "opponent_observation"]
            decisions = [r for r in rows if r.get("event") == "persona_decision"]
            full = [r for r in decisions if r.get("mode_after") == "FULL"]
            suspects = [r for r in obs if r.get("suspect") == "1"]
            scores = [int(r.get("accelerated_score_milli", "0")) for r in obs]
            evidence = [int(r.get("accelerated_evidence_milli", "0")) for r in obs]
            summaries.append({
                "arm": arm,
                "game_log": str(path.relative_to(root)),
                "observation_count": len(obs),
                "first_suspect_ply": int(suspects[0]["ply"]) if suspects else None,
                "first_suspect_reason": suspects[0].get("suspect_reason") if suspects else None,
                "first_full_ply": int(full[0]["ply"]) if full else None,
                "max_fusion_score_milli": max(scores) if scores else None,
                "max_fusion_evidence_milli": max(evidence) if evidence else None,
                "last_estimate_elo": int(obs[-1]["estimate_elo"]) if obs else None,
                "last_confidence_cp": int(obs[-1]["confidence_cp"]) if obs else None,
            })
    by_arm = defaultdict(list)
    for row in summaries:
        by_arm[row["arm"]].append(row)
    report = {"games": summaries, "arms": {}}
    for arm, rows in by_arm.items():
        latencies = [r["first_full_ply"] for r in rows if r["first_full_ply"] is not None]
        scores = [r["max_fusion_score_milli"] for r in rows if r["max_fusion_score_milli"] is not None]
        report["arms"][arm] = {
            "games": len(rows),
            "full_confirmation_rate": len(latencies) / len(rows) if rows else None,
            "first_full_ply_mean": statistics.mean(latencies) if latencies else None,
            "first_full_ply_median": statistics.median(latencies) if latencies else None,
            "max_fusion_score_mean": statistics.mean(scores) if scores else None,
            "reasons": dict(Counter(r["first_suspect_reason"] for r in rows)),
        }
    print(json.dumps(report, indent=2))
    (root / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
