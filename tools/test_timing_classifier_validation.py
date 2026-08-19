#!/usr/bin/env python3

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("timing_classifier_validation.py")
SPEC = importlib.util.spec_from_file_location("timing_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


CONFIG = {
    "schema": 1,
    "seed": 7,
    "window_size": 32,
    "skip_initial_player_moves": 1,
    "production_threshold": 0.45,
    "bootstrap_iterations": 50,
    "permutation_iterations": 100,
    "matching": {
        "elo_bin_width": 200,
        "max_records_per_account": 2,
        "max_records_per_account_cell": 1,
    },
    "gates": {
        "minimum_source_bot_accounts": 1,
        "minimum_matched_bot_accounts": 1,
        "minimum_account_auc_lower_95": 0.0,
        "maximum_threshold_fpr_upper_95": 1.0,
    },
}


def game(game_id, white, black, white_bot=False, black_bot=False):
    white_title = '[WhiteTitle "BOT"]\n' if white_bot else ""
    black_title = '[BlackTitle "BOT"]\n' if black_bot else ""
    # White's successive positive times are 1,2,4,8,4,2,1 seconds after the
    # skipped first move; Black has a separate non-constant pattern.
    white_clocks = [180, 179, 177, 173, 165, 161, 159, 158]
    black_clocks = [180, 178, 175, 174, 168, 165, 163, 157]
    moves = []
    for index, (white_clock, black_clock) in enumerate(
        zip(white_clocks, black_clocks), 1
    ):
        moves.append(
            f"{index}. e4 {{ [%clk 0:{white_clock // 60:02d}:{white_clock % 60:02d}] }} "
            f"e5 {{ [%clk 0:{black_clock // 60:02d}:{black_clock % 60:02d}] }}"
        )
    return (
        '[Event "Rated Blitz game"]\n'
        f'[Site "https://lichess.org/{game_id}"]\n'
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        '[WhiteElo "1500"]\n'
        '[BlackElo "1500"]\n'
        f"{white_title}{black_title}"
        '[Variant "Standard"]\n'
        '[TimeControl "180+0"]\n'
        '[UTCDate "2026.08.19"]\n\n'
        + " ".join(moves)
        + " 1-0\n\n"
    )


class TimingValidationTests(unittest.TestCase):
    def test_autocorrelation_matches_direct_formula(self):
        values = [-5.0, -4.0, -2.0, -1.0, -2.5, -4.5, -5.5]
        actual = MODULE.lag1_autocorrelation(values)
        left = values[:-1]
        right = values[1:]
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        numerator = sum(
            (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
        )
        denominator = math.sqrt(
            sum((a - left_mean) ** 2 for a in left)
            * sum((b - right_mean) ** 2 for b in right)
        )
        self.assertAlmostEqual(actual, numerator / denominator, places=14)
        self.assertIsNone(MODULE.lag1_autocorrelation(values[:5]))

    def test_extractor_selects_affirmative_bots_and_pseudonymises(self):
        with tempfile.TemporaryDirectory() as directory:
            bot_path = Path(directory) / "bot.pgn"
            human_path = Path(directory) / "human.pgn"
            bot_path.write_text(game("botgame", "Robot", "Opponent", True), encoding="utf-8")
            human_path.write_text(
                game("humangame", "Alice", "Bob")
                + game("excludedbot", "OtherRobot", "Carol", True),
                encoding="utf-8",
            )
            records, summary = MODULE.extract_records(
                CONFIG, [bot_path], [human_path]
            )
        labels = [record["label"] for record in records]
        self.assertEqual(labels.count("bot"), 1)
        self.assertEqual(labels.count("unmarked"), 3)
        self.assertEqual(summary["accounts"], {"bot": 1, "unmarked": 3})
        serialized = json.dumps(records)
        for raw_name in ("Robot", "Opponent", "Alice", "Bob", "Carol"):
            self.assertNotIn(raw_name, serialized)
        self.assertTrue(all(record["positive_samples"] == 7 for record in records))

    def test_variation_clock_does_not_shift_mainline_colors(self):
        text = (
            "1. e4 { [%clk 0:03:00] } "
            "(1. d4 { [%clk 0:00:01] } d5 { [%clk 0:00:01] }) "
            "e5 { [%clk 0:02:59] }"
        )
        clocks = MODULE.parse_clocks("[Event \"x\"]\n\n" + text)
        self.assertEqual([clock.seconds for clock in clocks], [180.0, 179.0])

    def test_matching_is_exact_and_account_capped(self):
        records = []
        for index in range(4):
            for label in ("bot", "unmarked"):
                records.append(
                    {
                        "schema": 1,
                        "account": f"{label}-same",
                        "game": f"{label}-{index}",
                        "label": label,
                        "rating": 1500,
                        "time_control": "180+0",
                        "acf1": index / 10,
                    }
                )
        # This unmatched time-control record must never be paired.
        records.append(
            {
                "schema": 1,
                "account": "unmarked-extra",
                "game": "extra",
                "label": "unmarked",
                "rating": 1500,
                "time_control": "300+0",
                "acf1": 0.0,
            }
        )
        bots, unmarked = MODULE.matched_records(records, CONFIG)
        self.assertEqual(len(bots), 1)
        self.assertEqual(len(unmarked), 1)
        self.assertEqual(bots[0]["time_control"], unmarked[0]["time_control"])

    def test_auc_handles_wins_losses_and_ties(self):
        self.assertEqual(MODULE.roc_auc([1.0], [0.0]), 1.0)
        self.assertEqual(MODULE.roc_auc([0.0], [1.0]), 0.0)
        self.assertEqual(MODULE.roc_auc([1.0], [1.0]), 0.5)
        self.assertIsNone(MODULE.roc_auc([], [1.0]))

    def test_snapshot_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "records.jsonl"
            manifest = Path(directory) / "manifest.json"
            records.write_text("trusted\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "derived_outputs": {
                            "records.jsonl": {
                                "bytes": records.stat().st_size,
                                "sha256": MODULE.file_sha256(records),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            verified = MODULE.verify_snapshot(records, manifest)
            self.assertEqual(verified["records_sha256"], MODULE.file_sha256(records))
            records.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.verify_snapshot(records, manifest)


if __name__ == "__main__":
    unittest.main()
