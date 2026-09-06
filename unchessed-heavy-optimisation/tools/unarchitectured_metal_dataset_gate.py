#!/usr/bin/env python3
"""Autonomously reject unsafe train/tune/final datasets before GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import date
from pathlib import Path

from unarchitectured_metal_data import POLICY_GUIDE, POLICY_HUMAN, iter_shard


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_provenance_files(paths, split):
    declared = split.get("shards")
    if not isinstance(declared, list) or not declared:
        return False
    expected = {item["name"]: item["sha256"] for item in declared}
    if len(expected) != len(declared):
        return False
    actual = {Path(path).name: file_sha256(path) for path in paths}
    return actual == expected


def position_key(record) -> int:
    base = record.base
    payload = struct.pack(
        "<12QBBB",
        *base.bitboards,
        base.castling,
        base.ep_file,
        min(15, base.halfmove // 8),
    )
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def summarize(paths):
    positions = set()
    games = set()
    players = set()
    records = human = guide = guide_labelled = duplicates = labelled = 0
    for path in paths:
        for record in iter_shard(path):
            records += 1
            is_guide = record.policy_kind == POLICY_GUIDE
            has_regret = bool(record.legal_flags & 1)
            human += record.policy_kind == POLICY_HUMAN
            guide += is_guide
            guide_labelled += is_guide and has_regret
            labelled += has_regret
            games.add(record.base.game_hash)
            players.add(record.base.player_hash)
            key = position_key(record)
            duplicates += key in positions
            positions.add(key)
    return {
        "records": records,
        "human": human,
        "guide": guide,
        "guide_regret_labelled": guide_labelled,
        "regret_labelled": labelled,
        "positions": positions,
        "games": games,
        "players": players,
        "duplicate_positions": duplicates,
        "duplicate_fraction": duplicates / max(1, records),
    }


def verify_dates(provenance):
    splits = provenance["splits"]
    train_start = date.fromisoformat(splits["train"]["start_date"])
    train_end = date.fromisoformat(splits["train"]["end_date"])
    tune_start = date.fromisoformat(splits["tune"]["start_date"])
    tune_end = date.fromisoformat(splits["tune"]["end_date"])
    final_start = date.fromisoformat(splits["final"]["start_date"])
    final_end = date.fromisoformat(splits["final"]["end_date"])
    return (
        train_start <= train_end < tune_start <= tune_end < final_start <= final_end
    )


def audit(train, tune, final, safety, training, provenance):
    summaries = {
        "train": summarize(train),
        "tune": summarize(tune),
        "final": summarize(final),
    }
    train_summary = summaries["train"]
    minimum_records = (
        training["oracle"]["effective_batch_records"]
        * training["oracle"]["minimum_optimizer_steps_per_epoch"]
    )
    checks = {
        "minimum_training_records": train_summary["records"] >= minimum_records,
        "minimum_tune_records": summaries["tune"]["records"]
        >= training["oracle"]["minimum_validation_records"],
        "minimum_final_records": summaries["final"]["records"]
        >= training["oracle"]["minimum_validation_records"],
        "minimum_human_fraction": train_summary["human"] / max(1, train_summary["records"])
        >= safety["data"]["minimum_human_fraction"],
        "minimum_guide_fraction": train_summary["guide"] / max(1, train_summary["records"])
        >= safety["data"]["minimum_guide_fraction"],
        "all_train_guides_have_regret": (
            not safety["data"].get("require_all_guide_regrets", True)
            or train_summary["guide_regret_labelled"] == train_summary["guide"]
        ),
        "minimum_tune_guide_fraction": summaries["tune"]["guide_regret_labelled"]
        / max(1, summaries["tune"]["records"])
        >= safety["data"]["minimum_tune_guide_fraction"],
        "minimum_final_guide_fraction": summaries["final"]["guide_regret_labelled"]
        / max(1, summaries["final"]["records"])
        >= safety["data"]["minimum_final_guide_fraction"],
        "maximum_duplicate_fraction": all(
            value["duplicate_fraction"]
            <= safety["data"]["maximum_duplicate_position_fraction"]
            for value in summaries.values()
        ),
        "game_disjoint": not (
            (summaries["train"]["games"] & summaries["tune"]["games"])
            or (summaries["train"]["games"] & summaries["final"]["games"])
            or (summaries["tune"]["games"] & summaries["final"]["games"])
        ),
        "player_disjoint": not (
            (summaries["train"]["players"] & summaries["tune"]["players"])
            or (summaries["train"]["players"] & summaries["final"]["players"])
            or (summaries["tune"]["players"] & summaries["final"]["players"])
        ),
        "position_disjoint": not (
            (summaries["train"]["positions"] & summaries["tune"]["positions"])
            or (summaries["train"]["positions"] & summaries["final"]["positions"])
            or (summaries["tune"]["positions"] & summaries["final"]["positions"])
        ),
        "future_ordered": verify_dates(provenance),
        "train_provenance_hashes": verify_provenance_files(
            train, provenance["splits"]["train"]
        ),
        "tune_provenance_hashes": verify_provenance_files(
            tune, provenance["splits"]["tune"]
        ),
        "final_provenance_hashes": verify_provenance_files(
            final, provenance["splits"]["final"]
        ),
    }
    serializable = {
        split: {key: value for key, value in summary.items() if not isinstance(value, set)}
        for split, summary in summaries.items()
    }
    return {
        "schema": 1,
        "architecture": "Unarchitectured Metal",
        "passed": all(checks.values()),
        "checks": checks,
        "summaries": serializable,
        "minimum_training_records": minimum_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--tune", nargs="+", required=True)
    parser.add_argument("--final", nargs="+", required=True)
    parser.add_argument(
        "--safety", default="config/unarchitectured_metal_safety.json"
    )
    parser.add_argument(
        "--training", default="config/unarchitectured_metal_training.json"
    )
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.train,
        args.tune,
        args.final,
        json.loads(Path(args.safety).read_text()),
        json.loads(Path(args.training).read_text()),
        json.loads(Path(args.provenance).read_text()),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    if args.strict and not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise SystemExit("dataset safety gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
