#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("uci_epd_suite.py")
SPEC = importlib.util.spec_from_file_location("uci_epd_suite", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UciEpdSuiteTests(unittest.TestCase):
    def test_parse_epd_coordinate_answers_and_quoted_id(self):
        position = MODULE.parse_epd_line(
            '8/8/8/8/8/8/4K3/7k w - - bm e2e3 e2f3; am e2d1; id "case; one";',
            7,
        )
        self.assertEqual(position.fen, "8/8/8/8/8/8/4K3/7k w - - 0 1")
        self.assertEqual(position.best_moves, ("e2e3", "e2f3"))
        self.assertEqual(position.avoid_moves, ("e2d1",))
        self.assertEqual(position.identifier, "case; one")
        self.assertEqual(position.line_number, 7)

    def test_solved_supports_best_and_avoid_moves(self):
        best = MODULE.parse_epd_line(
            '8/8/8/8/8/8/4K3/7k w - - bm e2e3; id "best";', 1
        )
        avoid = MODULE.parse_epd_line(
            '8/8/8/8/8/8/4K3/7k w - - am e2d1; id "avoid";', 2
        )
        san = MODULE.parse_epd_line(
            '8/8/8/8/8/8/4K3/7k w - - bm Ke3; id "san";', 3
        )
        self.assertTrue(MODULE.solved(best, "e2e3"))
        self.assertFalse(MODULE.solved(best, "e2f3"))
        self.assertFalse(MODULE.solved(avoid, "e2d1"))
        self.assertTrue(MODULE.solved(avoid, "e2e3"))
        self.assertIsNone(MODULE.solved(san, "e2e3"))

    def test_uci_engine_collects_search_metadata(self):
        fake = r'''
import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == "uci":
        print("id name Fake Reviewer", flush=True)
        print("id author Tests", flush=True)
        print("uciok", flush=True)
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("go "):
        print("info depth 7 score cp 23 nodes 321 time 4 pv e2e4", flush=True)
        print("bestmove e2e4", flush=True)
    elif command == "quit":
        break
'''
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake.py"
            script.write_text(fake, encoding="utf-8")
            engine = MODULE.UciEngine([sys.executable, str(script)], 2.0)
            try:
                engine.handshake({"Threads": "1"})
                position = MODULE.parse_epd_line(
                    '8/8/8/8/8/8/4K3/7k w - - bm e2e4; id "fake";', 1
                )
                result = engine.search(position, "go depth 7")
            finally:
                engine.close()
        self.assertEqual(engine.name, "Fake Reviewer")
        self.assertEqual(result.bestmove, "e2e4")
        self.assertEqual(result.depth, 7)
        self.assertEqual(result.nodes, 321)
        self.assertEqual(result.time_ms, 4)
        self.assertEqual((result.score_kind, result.score), ("cp", 23))


if __name__ == "__main__":
    unittest.main()
