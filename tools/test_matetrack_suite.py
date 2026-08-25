"""Verify the matetrack suite and keep the Rust copy in sync.

Three things are guarded:

1. **Every EPD entry really is a unique forced mate.** Re-derived with
   python-chess, never trusted from the file. Building this suite already
   caught one bad fixture -- a "ladder mate" where the king simply walked to
   the seventh rank -- so the check is not hypothetical.

2. **The Rust suite in `search.rs` matches the EPD file.** The positions are
   transcribed by hand into the Rust test; a typo there would leave the gate
   green while testing something else.

3. **Pattern coverage.** A suite of seven back-rank mates would pass while the
   engine was blind to smothered mate, so distinct mating mechanisms are
   required.

Needs only python-chess.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
EPD = ROOT / "benchmarks" / "matetrack.epd"
SEARCH_RS = ROOT / "unchessed-core" / "src" / "search.rs"

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def load_epd():
    entries = []
    for line in EPD.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        epd = line.split(" bm ")[0].strip()
        bm = line.split(" bm ")[1].split(";")[0].strip()
        ident = line.split('id "')[1].split('"')[0]
        dm = int(line.split(" dm ")[1].split(";")[0])
        entries.append({"id": ident, "epd": epd, "bm": bm, "dm": dm})
    return entries


def rust_suite():
    src = SEARCH_RS.read_text()
    start = src.index("fn matetrack_suite_finds_every_forced_mate")
    block = src[start : src.index("\n    }", start)]
    return re.findall(
        r'\(\s*"([\w-]+)",\s*"([^"]+)",\s*"([a-h][1-8][a-h][1-8][qrbn]?)",?\s*\)',
        block,
    )


class EpdTests(unittest.TestCase):
    def setUp(self):
        self.entries = load_epd()

    def test_suite_is_not_empty(self):
        self.assertGreaterEqual(len(self.entries), 5)

    def test_every_entry_is_a_legal_unique_mate(self):
        for e in self.entries:
            with self.subTest(id=e["id"]):
                board = chess.Board(e["epd"] + " 0 1")
                self.assertTrue(board.is_valid(), "illegal position")
                move = board.parse_san(e["bm"])
                self.assertIn(move, board.legal_moves)

                mates = []
                for candidate in list(board.legal_moves):
                    board.push(candidate)
                    if board.is_checkmate():
                        mates.append(candidate.uci())
                    board.pop()
                self.assertEqual(
                    mates,
                    [move.uci()],
                    f"mate must be unique; found {mates}",
                )

    def test_declared_depth_matches(self):
        for e in self.entries:
            with self.subTest(id=e["id"]):
                board = chess.Board(e["epd"] + " 0 1")
                board.push(board.parse_san(e["bm"]))
                self.assertEqual(e["dm"], 1)
                self.assertTrue(board.is_checkmate())

    def test_no_duplicate_positions(self):
        seen = [e["epd"] for e in self.entries]
        self.assertEqual(len(seen), len(set(seen)))

    def test_ids_are_unique(self):
        ids = [e["id"] for e in self.entries]
        self.assertEqual(len(ids), len(set(ids)))

    def test_covers_distinct_mating_patterns(self):
        """Seven back-rank mates would be a weak suite."""
        patterns = {e["id"].split("-")[0] for e in self.entries}
        self.assertGreaterEqual(len(patterns), 4, f"only {patterns}")
        self.assertIn("smothered", patterns)

    def test_both_colours_are_represented(self):
        turns = {chess.Board(e["epd"] + " 0 1").turn for e in self.entries}
        self.assertEqual(turns, {chess.WHITE, chess.BLACK})


class RustSyncTests(unittest.TestCase):
    def setUp(self):
        self.entries = {e["id"]: e for e in load_epd()}
        self.rust = rust_suite()

    def test_rust_test_exists(self):
        self.assertIn(
            "matetrack_suite_finds_every_forced_mate", SEARCH_RS.read_text()
        )

    def test_same_positions_in_both(self):
        self.assertEqual(len(self.rust), len(self.entries))
        self.assertEqual({r[0] for r in self.rust}, set(self.entries))

    def test_rust_fens_and_moves_match_the_epd(self):
        for ident, fen, uci in self.rust:
            with self.subTest(id=ident):
                entry = self.entries[ident]
                self.assertEqual(" ".join(fen.split()[:4]), entry["epd"])
                board = chess.Board(entry["epd"] + " 0 1")
                self.assertEqual(board.parse_san(entry["bm"]).uci(), uci)


if __name__ == "__main__":
    unittest.main()
