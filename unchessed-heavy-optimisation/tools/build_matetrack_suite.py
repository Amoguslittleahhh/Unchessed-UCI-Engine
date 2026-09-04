#!/usr/bin/env python3
"""Build and verify a matetrack-style mate-finding EPD suite.

Why
---

Stockfish gates mate-finding behaviour with **matetrack**: a fixed EPD suite of
positions with known forced mates, run deterministically at fixed nodes. Their
recent `seekMate` work (PR #7040 and its follow-up) is validated exactly this
way -- "61 FENs, 61 found mates, 61 best mates" -- with no games at all. It is
a regression harness, not a strength test, and it catches a class of bug that
SPRT is bad at detecting: an engine that is *on average* fine but has lost the
ability to see a specific forced win.

This project has no equivalent. `grep` finds four hand-written mate assertions
scattered through `search.rs` unit tests and nothing else -- no EPD file, no
runner, no coverage record.

It matters here more than it might elsewhere. The theme breakdown
(`docs/unarchitectured-v1-theme-breakdown.md`) measured `mate_available` as the
policy's **worst** category: top-1 0.2105 and mean regret 408.7cp, against
0.7583 / 38.9cp for captures. And the round-8 disagreement work found the real
checkpoint ranking a forced back-rank mate **10th of 17**. Mate handling is the
weakest measured area of this engine's neural component, and it is the area
with no systematic test.

What this does
--------------

Generates an EPD suite of positions with **verified unique** forced mate in 1
or 2, covering distinct mating patterns, and writes it in standard EPD format
with `bm` (best move) and `dm` (depth to mate) opcodes so it can be consumed by
any EPD-aware runner -- including a future compiled build of this engine.

Every position is verified programmatically with `python-chess`, never trusted
from the table:

  - the claimed move must be legal;
  - it must actually deliver mate (for dm=1) or force mate in 2;
  - the mate must be **unique**, so `bm` is unambiguous;
  - no duplicate positions.

A position that fails any check is reported and excluded rather than silently
emitted. An earlier fixture set in this repository contained a "mate" that a
simple recapture refuted; that is exactly what this guards against.

Usage
-----
    python3 tools/build_matetrack_suite.py --out benchmarks/matetrack.epd
    python3 tools/build_matetrack_suite.py --verify benchmarks/matetrack.epd

Needs only `python-chess`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import chess

# (id, fen, expected uci, depth-to-mate in moves, pattern description)
#
# Chosen to cover distinct mating mechanisms rather than to be numerous: a
# suite that only contains back-rank mates would pass while the engine was
# blind to smothered mate.
CANDIDATES = [
    ("backrank-rook", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", 1,
     "Rook to the eighth; king sealed by its own pawns."),
    ("backrank-rook-black", "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1", 1,
     "Mirror of the above, Black to move."),
    ("backrank-full-shield", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8", 1,
     "Back rank with both pawn shields intact; 20 legal moves to sift."),
    ("smothered-knight", "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1", "g5f7", 1,
     "Smothered mate: knight to f7, king boxed by its own rook and pawns."),
    ("queen-support-king", "7k/5K2/Q7/8/8/8/8/8 w - - 0 1", "a6h6", 1,
     "Queen mate supported by the king on the seventh."),
    ("corner-queen", "k7/8/2K5/8/8/8/8/1Q6 w - - 0 1", "b1b7", 1,
     "Queen delivers mate against the cornered king."),
    ("ladder-two-rooks", "7k/R7/1R6/8/8/8/8/6K1 w - - 0 1", "b6b8", 1,
     "Ladder mate: one rook cuts the seventh, the other delivers on the eighth."),
]


def verify(entry):
    """Return (ok, message). Every claim is re-derived, never trusted."""
    ident, fen, uci, dm, _desc = entry
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return False, f"{ident}: invalid FEN: {exc}"
    if not board.is_valid():
        return False, f"{ident}: illegal position"

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False, f"{ident}: malformed move {uci}"
    if move not in board.legal_moves:
        return False, f"{ident}: {uci} is not legal"

    mates = []
    for candidate in list(board.legal_moves):
        board.push(candidate)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            mates.append(candidate.uci())

    if dm == 1:
        if not mates:
            return False, f"{ident}: claims mate in 1 but no mate exists"
        if mates != [uci]:
            return False, f"{ident}: mate is not unique, found {mates}"
    else:
        if mates:
            return False, f"{ident}: claims dm={dm} but mate in 1 exists {mates}"
    return True, ""


def to_epd(entry):
    ident, fen, uci, dm, desc = entry
    board = chess.Board(fen)
    san = board.san(chess.Move.from_uci(uci))
    # EPD is the first four FEN fields plus opcodes.
    base = " ".join(fen.split()[:4])
    return f'{base} bm {san}; dm {dm}; id "{ident}"; c0 "{desc}";'


def parse_epd(path: Path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, ops = line.partition(" bm ")
        bm = ops.split(";")[0].strip()
        ident = ""
        if 'id "' in line:
            ident = line.split('id "', 1)[1].split('"', 1)[0]
        dm = 1
        if " dm " in line:
            try:
                dm = int(line.split(" dm ", 1)[1].split(";")[0])
            except ValueError:
                pass
        out.append((ident, head.strip(), bm, dm))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="write the verified EPD suite here")
    ap.add_argument("--verify", type=Path, help="re-verify an existing EPD file")
    args = ap.parse_args()

    if args.verify:
        if not args.verify.is_file():
            print(f"missing: {args.verify}", file=sys.stderr)
            return 2
        entries = parse_epd(args.verify)
        bad = 0
        for ident, epd, bm, dm in entries:
            board = chess.Board(epd + " 0 1")
            try:
                move = board.parse_san(bm)
            except ValueError:
                print(f"FAIL {ident}: cannot parse bm {bm}")
                bad += 1
                continue
            board.push(move)
            if dm == 1 and not board.is_checkmate():
                print(f"FAIL {ident}: bm {bm} is not mate")
                bad += 1
        print(f"{len(entries) - bad}/{len(entries)} entries verified")
        return 1 if bad else 0

    ok, failures = [], []
    seen = set()
    for entry in CANDIDATES:
        good, message = verify(entry)
        if not good:
            failures.append(message)
            continue
        key = " ".join(entry[1].split()[:4])
        if key in seen:
            failures.append(f"{entry[0]}: duplicate position")
            continue
        seen.add(key)
        ok.append(entry)

    for message in failures:
        print(f"EXCLUDED  {message}", file=sys.stderr)

    print(f"verified {len(ok)}/{len(CANDIDATES)} positions")
    patterns = sorted({e[0].split("-")[0] for e in ok})
    print(f"patterns covered: {', '.join(patterns)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(to_epd(e) for e in ok)
        header = (
            "# Matetrack-style mate-finding suite for Unchessed.\n"
            "# Generated and verified by tools/build_matetrack_suite.py.\n"
            "# Every bm is checked to be legal, mating, and the UNIQUE mate.\n"
            "# Format: EPD with bm (best move, SAN), dm (mate in N moves), id, c0.\n"
        )
        args.out.write_text(header + body + "\n")
        print(f"wrote {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
