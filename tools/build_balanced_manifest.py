#!/usr/bin/env python3
"""Build a stratified, player-capped PGN sampling manifest.

The output is JSON Lines. Each row identifies a source PGN byte range and one
player perspective selected into an
Elo-band x time-control x color x result reservoir. PGNs are not copied.

Usage:
  python tools/build_balanced_manifest.py \
      --config config/elo_sampling.json \
      --output balanced.jsonl games1.pgn [games2.pgn ...]

This is the metadata pass only. Position extraction should additionally
balance game phase, difficulty, tacticality, and policy entropy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HEADER = re.compile(rb'^\[([^ ]+) "(.*)"\]\s*$')


def iter_games(path: Path):
    """Yield (offset, length, headers) without loading the whole PGN."""
    with path.open("rb") as stream:
        start = None
        headers = {}
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                if start is not None:
                    yield start, offset - start, headers
                return
            if line.startswith(b"[Event "):
                if start is not None:
                    yield start, offset - start, headers
                start = offset
                headers = {}
            if start is not None and line.startswith(b"["):
                match = HEADER.match(line.rstrip(b"\r\n"))
                if match:
                    key = match.group(1).decode("ascii", "replace")
                    value = match.group(2).decode("utf-8", "replace")
                    headers[key] = value


def parse_rating(value):
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return None
    return rating if rating > 0 else None


def parse_base_seconds(value):
    if not value or value in {"-", "?"}:
        return None
    base = value.split("+", 1)[0]
    if "/" in base:  # e.g. 40/7200
        base = base.rsplit("/", 1)[-1]
    try:
        return int(base)
    except ValueError:
        return None


def named_range(value, ranges, low_key, high_key):
    for item in ranges:
        if item[low_key] <= value <= item[high_key]:
            return item["name"]
    return None


@dataclass
class Reservoir:
    quota: int
    player_cap: int
    rng: random.Random
    seen: int = 0
    rows: list[dict] = field(default_factory=list)
    players: Counter = field(default_factory=Counter)
    player_seen: Counter = field(default_factory=Counter)

    def add(self, row):
        self.seen += 1
        player = row["player"]
        self.player_seen[player] += 1
        if self.players[player] >= self.player_cap:
            # Maintain a fair per-player reservoir instead of permanently
            # keeping that player's first games in the file.
            if self.rng.randrange(self.player_seen[player]) < self.player_cap:
                slots = [
                    i for i, existing in enumerate(self.rows)
                    if existing["player"] == player
                ]
                if slots:
                    self.rows[self.rng.choice(slots)] = row
            return
        if len(self.rows) < self.quota:
            self.rows.append(row)
            self.players[player] += 1
            return
        slot = self.rng.randrange(self.seen)
        if slot >= self.quota:
            return
        old_player = self.rows[slot]["player"]
        self.players[old_player] -= 1
        self.rows[slot] = row
        self.players[player] += 1


def build_manifest(config, paths):
    rng = random.Random(config.get("seed", 20260819))
    reservoirs = {}
    accepted_results = set(config["results"])
    stats = Counter()

    for source in paths:
        path = Path(source)
        for offset, length, headers in iter_games(path):
            stats["games_seen"] += 1
            result = headers.get("Result")
            base = parse_base_seconds(headers.get("TimeControl"))
            if result not in accepted_results or base is None:
                stats["games_missing_metadata"] += 1
                continue
            tc = named_range(
                base, config["time_controls"], "min_seconds", "max_seconds"
            )
            if tc is None:
                stats["games_unmapped_time_control"] += 1
                continue

            for color, opponent_color in (("White", "Black"), ("Black", "White")):
                rating = parse_rating(headers.get(color + "Elo"))
                opponent_rating = parse_rating(headers.get(opponent_color + "Elo"))
                player = headers.get(color)
                if rating is None or not player:
                    stats["perspectives_missing_metadata"] += 1
                    continue
                if config.get("exact_elo_cells", False):
                    minimum = int(config.get("min_elo", 100))
                    maximum = int(config.get("max_elo", 3650))
                    if not minimum <= rating <= maximum:
                        stats["perspectives_unmapped_elo"] += 1
                        continue
                    band = f"{rating:04d}"
                else:
                    band = named_range(rating, config["elo_bands"], "min", "max")
                    if band is None:
                        stats["perspectives_unmapped_elo"] += 1
                        continue
                key = (band, tc, color.lower(), result)
                reservoir = reservoirs.setdefault(
                    key,
                    Reservoir(
                        quota=int(config["per_cell"]),
                        player_cap=int(config["per_player_cell"]),
                        rng=rng,
                    ),
                )
                reservoir.add(
                    {
                        "source": str(path),
                        "offset": offset,
                        "length": length,
                        "perspective": color.lower(),
                        "player": player,
                        "opponent": headers.get(opponent_color, ""),
                        "rating": rating,
                        "opponent_rating": opponent_rating,
                        "time_control": headers.get("TimeControl"),
                        "base_seconds": base,
                        "result": result,
                        "cell": {
                            "elo": band,
                            "time_control": tc,
                            "color": color.lower(),
                            "result": result,
                        },
                    }
                )
                stats["perspectives_considered"] += 1

    rows = []
    cell_summary = {}
    for key in sorted(reservoirs):
        reservoir = reservoirs[key]
        reservoir.rows.sort(key=lambda row: (row["source"], row["offset"]))
        rows.extend(reservoir.rows)
        cell_summary["|".join(key)] = {
            "seen": reservoir.seen,
            "selected": len(reservoir.rows),
            "distinct_players": len([n for n in reservoir.players.values() if n]),
        }
    stats["perspectives_selected"] = len(rows)
    return rows, {"totals": dict(stats), "cells": cell_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", nargs="+", help="input PGN files")
    parser.add_argument("--config", default="config/elo_sampling.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", help="optional summary JSON path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    rows, summary = build_manifest(config, args.pgn)
    with open(args.output, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(summary["totals"], sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
