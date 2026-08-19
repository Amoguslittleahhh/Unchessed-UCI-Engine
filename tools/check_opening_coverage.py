#!/usr/bin/env python3
"""Audit named ECO and curated opening-book coverage."""

import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / "books" / "lichess-openings"


def main():
    rows = []
    for volume in "abcde":
        with (BOOK_DIR / f"{volume}.tsv").open(encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream, delimiter="\t"))

    expected = {f"{volume}{number:02d}" for volume in "ABCDE" for number in range(100)}
    codes = {row["eco"] for row in rows}
    assert len(rows) == 3_810, len(rows)
    assert codes == expected, f"missing={sorted(expected - codes)} extra={sorted(codes - expected)}"
    assert (BOOK_DIR / "CC0-1.0.txt").exists()
    assert "4b8622759e7ae6f93f011cc6c83a3823401ab45e" in (
        BOOK_DIR / "SOURCE.txt"
    ).read_text()

    source = (ROOT / "unchessed-core" / "src" / "book.rs").read_text()
    curated = re.findall(
        r'^\s*"((?:main|troll[123]);[^"\\]*)",?$', source, re.MULTILINE
    )
    tiers = Counter(line.split(";", 1)[0] for line in curated)
    assert tiers == {"main": 45, "troll1": 2, "troll2": 8, "troll3": 5}, tiers

    print(f"Historical named lines: {len(rows)}")
    print(f"ECO coverage: {len(codes)}/500 (100.0%)")
    print(f"Curated overlays: {sum(tiers.values())} {dict(tiers)}")
    print("Source/license metadata: PASS (CC0)")


if __name__ == "__main__":
    main()
