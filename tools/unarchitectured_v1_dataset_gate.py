#!/usr/bin/env python3
"""Autonomously reject unsafe train/tune/final datasets before GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import date
from pathlib import Path

from aegis_v4_data import POLICY_GUIDE, POLICY_HUMAN, iter_shard


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
    records = human = guide = duplicates = labelled = 0
    for path in paths:
        for record in iter_shard(path):
            records += 1
            human += record.policy_kind == POLICY_HUMAN
            guide += record.policy_kind == POLICY_GUIDE
            labelled += bool(record.legal_flags & 1)
            games.add(record.base.game_hash)
            players.add(record.base.player_hash)
            key = position_key(record)
            duplicates += key in positions
            positions.add(key)
    return {
        "records": records,
        "human": human,
        "guide": guide,
        "regret_labelled": labelled,
        "positions": positions,
        "games": games,
        "players": players,
        "duplicate_positions": duplicates,
        "duplicate_fraction": duplicates / max(1, records),
    }


def verify_dates(provenance):
    splits = provenance["splits"]
    train_end = date.fromisoformat(splits["train"]["end_date"])
    tune_start = date.fromisoformat(splits["tune"]["start_date"])
    tune_end = date.fromisoformat(splits["tune"]["end_date"])
    final_start = date.fromisoformat(splits["final"]["start_date"])
    return train_end < tune_start and tune_end < final_start


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
    }
    serializable = {
        split: {key: value for key, value in summary.items() if not isinstance(value, set)}
        for split, summary in summaries.items()
    }
    return {
        "schema": 1,
        "architecture": "Unarchitectured v1",
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
        "--safety", default="config/unarchitectured_v1_safety.json"
    )
    parser.add_argument(
        "--training", default="config/unarchitectured_v1_training.json"
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
