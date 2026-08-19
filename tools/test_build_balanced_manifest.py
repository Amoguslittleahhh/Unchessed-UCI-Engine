#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build_balanced_manifest.py")
SPEC = importlib.util.spec_from_file_location("balanced_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def game(number, rating, result="1-0"):
    return (
        f'[Event "G{number}"]\n'
        f'[White "W{number}"]\n'
        f'[Black "B{number}"]\n'
        f'[WhiteElo "{rating}"]\n'
        f'[BlackElo "{rating}"]\n'
        f'[TimeControl "600+5"]\n'
        f'[Result "{result}"]\n\n'
        f'1. e4 e5 {result}\n\n'
    )


class BalancedManifestTests(unittest.TestCase):
    def test_sparse_tails_get_same_quota_as_crowded_middle(self):
        config = {
            "elo_bands": [
                {"name": "low", "min": 0, "max": 799},
                {"name": "middle", "min": 800, "max": 2199},
                {"name": "high", "min": 2200, "max": 100000},
            ],
            "time_controls": [
                {"name": "rapid", "min_seconds": 0, "max_seconds": 999999}
            ],
            "results": ["1-0"],
            "per_cell": 3,
            "per_player_cell": 1,
            "seed": 7,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.pgn"
            text = "".join(
                [game(i, 500) for i in range(5)]
                + [game(100 + i, 1500) for i in range(40)]
                + [game(200 + i, 2600) for i in range(5)]
            )
            path.write_text(text, encoding="utf-8")
            rows, summary = MODULE.build_manifest(config, [path])

        counts = Counter(row["cell"]["elo"] for row in rows)
        # Three samples for each color are separate cells, so each Elo band
        # contributes six selected perspectives regardless of source crowding.
        self.assertEqual(counts, {"low": 6, "middle": 6, "high": 6})
        self.assertEqual(summary["totals"]["games_seen"], 50)
        self.assertEqual(summary["totals"]["perspectives_selected"], 18)
        self.assertTrue(all(row["length"] > 0 for row in rows))

    def test_exact_elo_cells_keep_adjacent_ratings_separate(self):
        config = {
            "exact_elo_cells": True,
            "min_elo": 100,
            "max_elo": 3650,
            "elo_bands": [],
            "time_controls": [
                {"name": "rapid", "min_seconds": 0, "max_seconds": 999999}
            ],
            "results": ["1-0"],
            "per_cell": 10,
            "per_player_cell": 10,
            "seed": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact.pgn"
            path.write_text(game(1, 500) + game(2, 501), encoding="utf-8")
            rows, _ = MODULE.build_manifest(config, [path])
        self.assertEqual({row["cell"]["elo"] for row in rows}, {"0500", "0501"})

    def test_missing_metadata_is_excluded(self):
        config = {
            "elo_bands": [{"name": "all", "min": 0, "max": 100000}],
            "time_controls": [
                {"name": "all", "min_seconds": 0, "max_seconds": 999999}
            ],
            "results": ["1-0"],
            "per_cell": 10,
            "per_player_cell": 10,
            "seed": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pgn"
            path.write_text('[Event "Missing"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
            rows, summary = MODULE.build_manifest(config, [path])
        self.assertEqual(rows, [])
        self.assertEqual(summary["totals"]["games_missing_metadata"], 1)


if __name__ == "__main__":
    unittest.main()
