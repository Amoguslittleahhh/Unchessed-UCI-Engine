#!/usr/bin/env python3
"""Fail closed when Rust, architecture, and GPU feature schemas drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rust_constant(source: str, name: str) -> int | None:
    match = re.search(rf"pub const {re.escape(name)}: usize = ([0-9_]+);", source)
    return int(match.group(1).replace("_", "")) if match else None


def audit(root=ROOT):
    config_path = root / "config/unarchitectured_v1.json"
    rust_path = root / "unchessed-core/src/threat_features.rs"
    gpu_path = root / "tools/train_nnue_xt_v3_a100.py"
    config = json.loads(config_path.read_text())
    xt = config["xt_nnue"]
    rust = rust_path.read_text()
    gpu = gpu_path.read_text()
    expected = {
        "direct_threat_dimensions": 12 * 12 * 15 * 15,
        "xray_hyperedge_dimensions": 12 * 12 * 12 * 8,
        "pawn_topology_dimensions": 4096,
        "direct_threat_width": 32,
        "xray_hyperedge_width": 16,
        "pawn_topology_width": 16,
    }
    checks = {
        f"config_{name}": xt.get(name) == value for name, value in expected.items()
    }
    checks.update(
        {
            "rust_direct_dimensions_formula": "THREAT_CLASSES * THREAT_CLASSES * THREAT_RELATIONS" in rust,
            "rust_xray_dimensions_formula": "THREAT_CLASSES * THREAT_CLASSES * THREAT_CLASSES" in rust,
            "rust_pawn_dimensions": rust_constant(rust, "PAWN_TOPOLOGY_DIMENSIONS") == 4096,
            "rust_direct_width": rust_constant(rust, "THREAT_LATENT_WIDTH") == 32,
            "rust_xray_width": rust_constant(rust, "XRAY_LATENT_WIDTH") == 16,
            "rust_pawn_width": rust_constant(rust, "PAWN_TOPOLOGY_WIDTH") == 16,
            "gpu_xray_dimensions": "XRAY_DIMENSIONS = 12 * 12 * 12 * 8" in gpu,
            "gpu_pawn_dimensions": "PAWN_TOPOLOGY_DIMENSIONS = 4096" in gpu,
            "shared_topology_hash_multiplier": "0x045D9F3B" in gpu
            and "0x045d_9f3b" in rust.lower(),
            "exact_delta_oracle": bool(xt.get("exact_delta_oracle")),
            "dirty_update_oracle_gate": bool(
                xt.get("dirty_update_requires_oracle_equality")
            ),
        }
    )
    return {
        "schema": 1,
        "architecture": "Unarchitectured v1",
        "passed": all(checks.values()),
        "checks": checks,
        "expected": expected,
        "source_sha256": {
            "config": sha256(config_path),
            "rust_features": sha256(rust_path),
            "gpu_features": sha256(gpu_path),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not report["passed"]:
        raise SystemExit("Unarchitectured v1 feature schema audit failed")


if __name__ == "__main__":
    main()
