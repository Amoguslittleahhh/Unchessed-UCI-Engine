"""Verify the NNUE king-bucket table against Stockfish's HalfKAv2_hm.

The Rust tests in `unchessed-core/src/nnue.rs` assert the same properties, but
cannot run in every environment here (no cargo toolchain). This is the
compiler-free equivalent: it parses both tables straight out of the Rust
source and checks them, so the invariant is enforced even when the Rust suite
cannot be executed.

Why this table deserves a dedicated test: it decides which of the 32 buckets
every feature index lands in. One wrong entry silently routes a king square to
the wrong 704-feature block. The net still loads, still evaluates, and still
passes the existing colour-mirror tests -- it just quietly reads the wrong
weights for some king positions, costing strength with no visible symptom.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NNUE_RS = ROOT / "unchessed-core" / "src" / "nnue.rs"

N_BUCKETS = 32


def parse_table(name: str, source: str) -> list[int]:
    match = re.search(
        rf"const {re.escape(name)}: \[i8; 64\] = \[(.*?)\];", source, re.S
    )
    if not match:
        raise AssertionError(f"could not find {name} in nnue.rs")
    values = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
    if len(values) != 64:
        raise AssertionError(f"{name} has {len(values)} entries, expected 64")
    return values


class KingBucketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = NNUE_RS.read_text()
        cls.ours = parse_table("KING_BUCKETS", source)
        # The reference copy lives inside the Rust test that checks it.
        cls.stockfish = parse_table("STOCKFISH", source)

    def test_reference_is_horizontally_symmetric(self):
        """Sanity-check the reference before trusting it.

        This symmetry is the property that makes mirroring to one half
        lossless. If the transcribed reference were wrong, everything below
        would be comparing against nonsense.
        """
        for sq in range(64):
            mirrored = (sq // 8) * 8 + (7 - sq % 8)
            with self.subTest(square=sq):
                self.assertEqual(self.stockfish[sq], self.stockfish[mirrored])

    def test_mirrored_half_matches_stockfish(self):
        """The finding: our e-h half is byte-identical to Stockfish's."""
        for sq in range(64):
            if sq % 8 >= 4:
                with self.subTest(square=sq):
                    self.assertEqual(self.ours[sq], self.stockfish[sq])

    def test_a_to_d_files_are_unreachable(self):
        for sq in range(64):
            if sq % 8 < 4:
                with self.subTest(square=sq):
                    self.assertEqual(self.ours[sq], -1)

    def test_buckets_are_a_bijection(self):
        """No duplicate bucket and no out-of-range index."""
        seen = set()
        for sq in range(64):
            if sq % 8 < 4:
                continue
            bucket = self.ours[sq]
            self.assertGreaterEqual(bucket, 0)
            self.assertLess(bucket, N_BUCKETS)
            self.assertNotIn(bucket, seen, f"bucket {bucket} reused at {sq}")
            seen.add(bucket)
        self.assertEqual(len(seen), N_BUCKETS)
        self.assertEqual(seen, set(range(N_BUCKETS)))

    def test_feature_dimensions(self):
        self.assertEqual(N_BUCKETS * 11 * 64, 22528)


class RustTestPresenceTests(unittest.TestCase):
    """The Rust-side assertions must not be deleted."""

    def test_rust_tests_exist(self):
        source = NNUE_RS.read_text()
        for name in (
            "king_buckets_match_stockfish_half_ka_v2_hm",
            "king_buckets_cover_only_the_mirrored_half",
            "v3_feature_dimensions_are_consistent",
        ):
            self.assertIn(name, source, f"{name} missing from nnue.rs")


if __name__ == "__main__":
    unittest.main()
