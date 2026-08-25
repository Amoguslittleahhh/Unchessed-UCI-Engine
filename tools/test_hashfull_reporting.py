"""Guard the UCI `hashfull` reporting path.

Compiler-free checks that the wiring stays intact, since the Rust tests
cannot be executed in every environment here. `hashfull` is easy to break
silently: the metric depends on an encoding invariant in `pack` (`depth + 1`,
so a live entry is never all-zero), and on the field surviving from `TT`
through `InfoEvent` to the printed info line. Any one of those three links
breaking leaves the engine printing `hashfull 0` forever, which looks
plausible rather than broken.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TT_RS = ROOT / "unchessed-core" / "src" / "tt.rs"
SEARCH_RS = ROOT / "unchessed-core" / "src" / "search.rs"
UCI_RS = ROOT / "unchessed-core" / "src" / "uci.rs"


class TtTests(unittest.TestCase):
    def setUp(self):
        self.src = TT_RS.read_text()

    def test_hashfull_exists_and_is_public(self):
        self.assertIn("pub fn hashfull(&self) -> usize", self.src)

    def test_occupancy_test_matches_the_pack_encoding(self):
        """`data != 0` is only exact because pack stores depth + 1."""
        self.assertIn("depth as i32 + 1", self.src)
        self.assertIn("slot.data.load(Ordering::Relaxed) != 0", self.src)

    def test_result_is_permille(self):
        self.assertIn("used * 1000 / sample", self.src)

    def test_sampling_is_bounded(self):
        """Must not scan a multi-GB table during search."""
        self.assertIn("self.table.len().min(1000)", self.src)

    def test_division_by_zero_is_guarded(self):
        block = self.src[self.src.index("pub fn hashfull") :][:700]
        self.assertIn("if sample == 0", block)

    def test_rust_tests_present(self):
        for name in (
            "hashfull_is_zero_when_empty_and_grows_when_filled",
            "hashfull_counts_depth_zero_entries",
            "hashfull_reports_partial_occupancy",
            "hashfull_is_bounded",
        ):
            self.assertIn(name, self.src, f"{name} missing")


class WiringTests(unittest.TestCase):
    """The value must survive TT -> InfoEvent -> printed line."""

    def test_info_event_carries_the_field(self):
        src = SEARCH_RS.read_text()
        struct = src[src.index("pub struct InfoEvent") :][:600]
        self.assertIn("pub hashfull: usize", struct)

    def test_info_event_is_populated_from_the_tt(self):
        self.assertIn("hashfull: s.tt.hashfull()", SEARCH_RS.read_text())

    def test_uci_prints_the_field(self):
        src = UCI_RS.read_text()
        self.assertIn("hashfull {}", src)
        self.assertIn("ev.hashfull", src)

    def test_field_order_matches_the_format_string(self):
        """A misordered argument would print nps where hashfull belongs."""
        src = UCI_RS.read_text()
        fmt_index = src.index('"info depth {} multipv {}')
        block = src[fmt_index : fmt_index + 500]
        fmt_line = block[: block.index('"', 1) + 1]
        placeholders = re.findall(r"(\w+) \{\}", fmt_line)
        args = [
            a.strip()
            for a in block[block.index("\n") :].split("\n")
            if a.strip().rstrip(",") and a.strip().startswith("ev.")
        ]
        # `hashfull` sits between `nps` and `time` in both lists.
        self.assertIn("hashfull", placeholders)
        self.assertLess(placeholders.index("nps"), placeholders.index("hashfull"))
        self.assertLess(placeholders.index("hashfull"), placeholders.index("time"))
        self.assertIn("ev.hashfull,", [a.rstrip() for a in args])


if __name__ == "__main__":
    unittest.main()
