"""Tests for the GAB ablation findings.

Two claims need locking down, because both are load-bearing for
`docs/gab-capacity-finding.md`:

1. The **capacity gap is real** — read from the actual exported checkpoint,
   not copied from the paper by hand. If someone retrains with a wider GAB,
   this must fail rather than keep asserting a stale finding.

2. The **ablation baseline agrees with the other committed artifact**. Both
   measure top-1 on the same checkpoint over the same 600 positions, so a
   disagreement means one of them is wrong. This exact mismatch already
   happened once (tied-best moves scored by string equality), so it is now
   pinned.

The checkpoint tests need only the package header, not torch.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "benchmarks" / "unarchitectured-v1" / "gab-ablation-2026-08-25.json"
ORDERING = ROOT / "benchmarks" / "unarchitectured-v1" / "ordering-risk-2026-08-24.json"
DOC = ROOT / "docs" / "gab-capacity-finding.md"
PACKAGE = ROOT / "artifacts" / "unarchitectured-v1-final.unarchv1"

# The paper's own configurations, arXiv:2605.19091 Appendix A.1.
PAPER_5M = {"d1": 32, "d2": 64, "d3": 64}


def read_section_shapes(path: Path) -> dict[str, tuple]:
    """Minimal UNARCHV1 header reader: section name -> shape.

    Deliberately independent of the reference implementation so the test does
    not inherit a bug from the code it checks.
    """
    header = struct.Struct("<8sHHIQQ16sIIQ")
    entry = struct.Struct("<HBBI8IQQfiII128s")
    data = path.read_bytes()
    magic, _v, _hs, section_count, *_ = header.unpack(data[:64])
    assert magic == b"UNARCHV1", magic

    shapes = {}
    offset = 0
    for _ in range(section_count):
        raw = data[64 + offset : 64 + offset + entry.size]
        fields = entry.unpack(raw)
        name_len, _dtype, ndim = fields[0], fields[1], fields[2]
        dims = fields[4:12][:ndim]
        name = fields[-1][:name_len].decode("ascii")
        shapes[name] = tuple(dims)
        offset += entry.size
    return shapes


@unittest.skipUnless(PACKAGE.is_file(), "checkpoint not present")
class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.shapes = read_section_shapes(PACKAGE)

    def test_gab_sections_exist(self):
        for name in ("gab.templates", "gab.token_projection", "gab.compress.weight"):
            self.assertIn(name, self.shapes)

    def test_capacity_matches_the_recorded_artifact(self):
        """The doc's table must come from the real file, not be hand-typed."""
        recorded = json.loads(ARTIFACT.read_text())["capacity"]["ours"]
        self.assertEqual(self.shapes["gab.token_projection"][0], recorded["d1"])
        self.assertEqual(self.shapes["gab.compress.weight"][0], recorded["d2"])
        self.assertEqual(self.shapes["gab.templates"][0], recorded["d3"])

    def test_gab_is_still_smaller_than_the_papers_smallest(self):
        """The finding itself. Should fail loudly if a retrain widens GAB."""
        self.assertLess(self.shapes["gab.token_projection"][0], PAPER_5M["d1"])
        self.assertLess(self.shapes["gab.compress.weight"][0], PAPER_5M["d2"])
        self.assertLess(self.shapes["gab.templates"][0], PAPER_5M["d3"])

    def test_templates_are_square_board_sized(self):
        self.assertEqual(self.shapes["gab.templates"][1:], (64, 64))


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(ARTIFACT.read_text())
        self.ablations = {a["variant"]: a for a in self.report["ablations"]}

    def test_all_three_variants_present(self):
        self.assertEqual(
            set(self.ablations), {"baseline", "gab_zeroed", "gab_shuffled"}
        )

    def test_every_variant_scored_the_whole_corpus(self):
        for name, a in self.ablations.items():
            with self.subTest(variant=name):
                self.assertEqual(a["positions"], 600)

    def test_removing_gab_hurts_materially(self):
        """The headline: GAB is load-bearing, not decorative."""
        base = self.ablations["baseline"]["top1_accuracy"]
        for variant in ("gab_zeroed", "gab_shuffled"):
            with self.subTest(variant=variant):
                self.assertLess(self.ablations[variant]["top1_accuracy"], base - 0.03)
                self.assertGreater(
                    self.ablations[variant]["mean_regret_cp"],
                    self.ablations["baseline"]["mean_regret_cp"],
                )

    def test_shuffling_is_as_damaging_as_zeroing(self):
        """Shows the signal is in the learned templates, not the tensor shape."""
        self.assertAlmostEqual(
            self.ablations["gab_zeroed"]["top1_accuracy"],
            self.ablations["gab_shuffled"]["top1_accuracy"],
            places=2,
        )


class ConsistencyTests(unittest.TestCase):
    """Both artifacts measure the same thing and must agree."""

    def test_baseline_matches_the_ordering_risk_artifact(self):
        gab = {a["variant"]: a for a in json.loads(ARTIFACT.read_text())["ablations"]}
        ordering = json.loads(ORDERING.read_text())["neural"]
        self.assertAlmostEqual(
            gab["baseline"]["top1_accuracy"],
            ordering["top1_accuracy"],
            places=6,
            msg="the two artifacts disagree about the same measurement",
        )
        self.assertAlmostEqual(
            gab["baseline"]["mean_regret_cp"],
            ordering["mean_first_move_regret_cp"],
            places=3,
        )


class DocTests(unittest.TestCase):
    def test_doc_numbers_come_from_the_artifact(self):
        text = DOC.read_text()
        ab = {a["variant"]: a for a in json.loads(ARTIFACT.read_text())["ablations"]}
        for value in (
            f"{ab['baseline']['top1_accuracy']:.4f}",
            f"{ab['gab_zeroed']['top1_accuracy']:.4f}",
            f"{ab['gab_zeroed']['mean_regret_cp']:.1f}",
        ):
            self.assertIn(value, text, f"{value} missing from the doc")

    def test_doc_records_the_stockfish_rejection(self):
        text = DOC.read_text()
        self.assertIn("f4bcd40", text)
        self.assertIn("Not portable", text)


if __name__ == "__main__":
    unittest.main()
