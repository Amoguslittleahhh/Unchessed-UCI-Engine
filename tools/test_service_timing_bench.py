#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("service_timing_bench.py")
SPEC = importlib.util.spec_from_file_location("service_timing_bench", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG = {"skip_initial_player_moves": 1, "window_size": 32}


def chesscom_game():
    white = [180, 179, 177, 173, 165, 161, 159, 158]
    black = [180, 178, 175, 174, 168, 165, 163, 157]
    moves = []
    for number, (wclock, bclock) in enumerate(zip(white, black), 1):
        moves.append(
            f"{number}. e4 {{[%clk 0:{wclock // 60:02d}:{wclock % 60:02d}]}} "
            f"e5 {{[%clk 0:{bclock // 60:02d}:{bclock % 60:02d}]}}"
        )
    return (
        '[Event "Live Chess"]\n[Site "Chess.com"]\n'
        '[White "Alice"]\n[Black "Bob"]\n[TimeControl "180"]\n\n'
        + " ".join(moves)
        + " 1-0\n"
    )


def fics_game():
    white = [0.0, 1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0]
    black = [0.0, 2.0, 3.0, 1.0, 6.0, 3.0, 2.0, 6.0]
    moves = []
    for number, (wtime, btime) in enumerate(zip(white, black), 1):
        moves.append(
            f"{number}. e4 {{[%emt {wtime}]}} e5 {{[%emt {btime}]}}"
        )
    return (
        '[Event "FICS rated blitz game"]\n[Site "FICS freechess.org"]\n'
        '[White "Robot"]\n[Black "Human"]\n[WhiteIsComp "Yes"]\n'
        '[TimeControl "180+0"]\n\n'
        + " ".join(moves)
        + " 1-0\n"
    )


class ServiceTimingBenchTests(unittest.TestCase):
    def test_chesscom_parser_handles_zero_increment_and_hides_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chesscom.pgn"
            path.write_text(chesscom_game(), encoding="utf-8")
            rows, details = MODULE.parse_chesscom([path], CONFIG)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["label"] for row in rows}, {"unmarked"})
        self.assertEqual({row["time_control"] for row in rows}, {"180+0"})
        self.assertNotIn("Alice", str(rows))
        self.assertEqual(details["games_seen"], 1)

    def test_fics_parser_uses_affirmative_computer_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fics.pgn"
            path.write_text(fics_game(), encoding="utf-8")
            rows, details = MODULE.parse_fics([path], CONFIG)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["label"] for row in rows], ["bot", "unmarked"])
        self.assertNotIn("Robot", str(rows))
        self.assertEqual(details["games_seen"], 1)

    def test_account_summary_uses_one_median_per_account(self):
        rows = [
            {"account": "a", "label": "unmarked", "acf1": 0.1, "time_class": "blitz"},
            {"account": "a", "label": "unmarked", "acf1": 0.9, "time_class": "blitz"},
            {"account": "b", "label": "unmarked", "acf1": 0.2, "time_class": "blitz"},
        ]
        summary = MODULE.class_summary(rows, 0.45)
        self.assertEqual(summary["accounts"], 2)
        self.assertAlmostEqual(summary["account_median_acf1"], 0.35)
        self.assertEqual(summary["account_threshold_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
