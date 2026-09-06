#!/usr/bin/env python3
"""Measure how game-phase definitions split the positions we actually have.

arXiv:2401.16852 ("Checkmating One, by Using Many", Helfenstein et al.)
trains one CNN expert per game phase and gates on the *Lichess* phase
definition. Its section 3.3 shows that a simpler gate (move counter, or a
pure material criterion like the one used by the earlier dkappe/Scorpio
line of work the paper compares against) measurably degrades the result.

Our own corpus builder (`tools/build_unarchitectured_metal_calibration_corpus.py::
classify_phase`) buckets by material and ply, not by the Lichess rules.
Before anyone commits to phase-specialized training data, the two
definitions' disagreement on *our* positions is a free, offline question.

This tool implements all three definitions on the same FENs and reports:

  * the phase counts under each definition (Lichess full, Lichess
    material-only — i.e. the endgame/middlegame thresholds without the
    backrank-sparseness and mixedness terms — and the corpus builder's
    material+ply rule);
  * the cross-tabulation of the phase stored in the corpus against the
    Lichess definition;
  * the per-position phases of the matetrack suite, which should be
    uniformly endgame.

Lichess definition (arXiv:2401.16852 Appendix B, ported verbatim from
lichess-org/scalachess `core/src/main/scala/Divider.scala`, accessed
2026-08-26; the paper's own link predates the repo reorganization to the
`core/` subdirectory):

  endgame    iff majors_and_minors <= 6
  middlegame iff not endgame and (majors_and_minors <= 10
                 or backrank_sparse or mixedness > 150)
  opening    otherwise

  majors_and_minors: queens, rooks, bishops, knights (kings and pawns
  do not count).
  backrank_sparse: fewer than 4 of your own pieces on your own back rank
  (rank 1 for white, rank 8 for black), king included.
  mixedness: sum over the 36 overlapping 2x2 blocks of a position-
  dependent score of (white pieces in block, black pieces in block);
  the score table below is the exact one from Divider.scala.

The gate is stateless per position (a game can return to the opening);
that is how the paper uses it in search, and it is what a training-data
split needs.

Usage:
  python3 tools/measure_game_phase_definitions.py \
      --corpus artifacts/unarchitectured-metal-calibration-corpus.jsonl \
      --epd benchmarks/matetrack.epd \
      --out benchmarks/unarchitectured-metal/game-phase-definitions.json

No non-repo dependency beyond `chess` (already in tools/requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chess

PHASES = ("opening", "middlegame", "endgame")

LICHESS_ENDGAME_MAX_MM = 6
LICHESS_MIDDLEGAME_MAX_MM = 10
LICHESS_MIXEDNESS_THRESHOLD = 150
CORPUS_ENDGAME_MAX_TOTAL = 12
CORPUS_ENDGAME_MAX_NON_PAWN = 4
CORPUS_OPENING_MAX_FULLMOVES = 12


# ---------------------------------------------------------------------------
# Lichess phase definition
# ---------------------------------------------------------------------------

def _block_score(y: int, white: int, black: int) -> int:
    """Per-2x2-block mixedness score, verbatim from scalachess Divider.scala.

    `y` is the 1-based rank index of the block's lower row (1..7);
    `white`/`black` are the piece counts of each color inside the block.
    """
    if white == 0:
        if black == 1:
            return 1 + y
        if black == 2:
            return 2 + (6 - y) if y < 6 else 0
        if black in (3, 4):
            return 3 + (7 - y) if y < 7 else 0
        return 0
    if white == 1:
        if black == 0:
            return 1 + (8 - y)
        if black == 1:
            return 5 + abs(4 - y)
        if black == 2:
            return 4 + (7 - y)
        if black == 3:
            return 5 + (7 - y)
        return 0
    if white == 2:
        if black == 0:
            return 2 + (y - 2) if y > 2 else 0
        if black == 1:
            return 4 + (y - 1)
        if black == 2:
            return 7
        return 0
    if white == 3:
        if black == 0:
            return 3 + (y - 1) if y > 1 else 0
        if black == 1:
            return 5 + (y - 1)
        return 0
    if white == 4:
        # "group of 4 on the homerow = 0" — comment from Divider.scala
        if black == 0:
            return 3 + (y - 1) if y > 1 else 0
        return 0
    return 0


def lichess_mixedness(board: chess.Board) -> int:
    """Sum of per-block scores over the 36 overlapping 2x2 blocks."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    acc = 0
    for yb in range(7):  # 0-based rank of the block's lower row
        y = yb + 1  # 1-based rank index used by the score table
        for x in range(7):  # 0-based file of the block's left column
            region = (
                (1 << (8 * yb + x))
                | (1 << (8 * yb + x + 1))
                | (1 << (8 * (yb + 1) + x))
                | (1 << (8 * (yb + 1) + x + 1))
            )
            acc += _block_score(y, int(white & region).bit_count(), int(black & region).bit_count())
    return acc


def majors_and_minors(board: chess.Board) -> int:
    return int(board.occupied & ~(board.kings | board.pawns)).bit_count()


def backrank_sparse(board: chess.Board) -> bool:
    return (
        int(board.occupied_co[chess.WHITE] & chess.BB_RANK_1).bit_count() < 4
        or int(board.occupied_co[chess.BLACK] & chess.BB_RANK_8).bit_count() < 4
    )


def lichess_phase(board: chess.Board) -> str:
    mm = majors_and_minors(board)
    if mm <= LICHESS_ENDGAME_MAX_MM:
        return "endgame"
    if mm <= LICHESS_MIDDLEGAME_MAX_MM or backrank_sparse(board) or lichess_mixedness(board) > LICHESS_MIXEDNESS_THRESHOLD:
        return "middlegame"
    return "opening"


def lichess_material_only_phase(board: chess.Board) -> str:
    """Lichess thresholds without the backrank/mixedness terms.

    This is the shape of the 'simplified framework, employing
    material-based criteria' (dkappe/Scorpio) that the paper's section
    3.3 shows to be inferior to the full Lichess definition.
    """
    mm = majors_and_minors(board)
    if mm <= LICHESS_ENDGAME_MAX_MM:
        return "endgame"
    if mm <= LICHESS_MIDDLEGAME_MAX_MM:
        return "middlegame"
    return "opening"


# ---------------------------------------------------------------------------
# The corpus builder's current definition (mirrored verbatim)
# ---------------------------------------------------------------------------

def corpus_builder_phase(board: chess.Board) -> str:
    """Mirror of tools/build_unarchitectured_metal_calibration_corpus.py."""
    pieces = board.piece_map().values()
    non_pawn = sum(1 for p in pieces if p.piece_type not in (chess.PAWN, chess.KING))
    total = sum(1 for _ in pieces)
    if total <= CORPUS_ENDGAME_MAX_TOTAL or non_pawn <= CORPUS_ENDGAME_MAX_NON_PAWN:
        return "endgame"
    if board.fullmove_number <= CORPUS_OPENING_MAX_FULLMOVES:
        return "opening"
    return "middlegame"


DEFINITIONS = {
    "lichess": lichess_phase,
    "lichess_material_only": lichess_material_only_phase,
    "corpus_builder": corpus_builder_phase,
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_corpus_positions(path: Path) -> list[tuple[str, str | None]]:
    """(fen, stored phase tag) for each record with a 'fen' field."""
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or "fen" not in rec:
                continue  # e.g. the leading provenance manifest
            out.append((rec["fen"], rec.get("phase")))
    return out


def read_epd_positions(path: Path) -> list[tuple[str, str]]:
    """(fen, id) for each EPD record (non-comment lines).

    EPD options are space-separated after the FEN's six fields; this repo's
    matetrack file additionally decorates option values with ';' which is
    stripped here.
    """
    out = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        fen_parts = tokens[:4]  # placement, side-to-move, castling, ep
        i = 4
        # halfmove/fullmove clocks are optional in EPD; take leading int fields
        while i < len(tokens) and len(fen_parts) < 6 and tokens[i].lstrip("-").isdigit():
            fen_parts.append(tokens[i])
            i += 1
        while len(fen_parts) < 6:
            fen_parts.append("0" if len(fen_parts) == 4 else "1")
        fen = " ".join(fen_parts)
        label = fen
        for j in range(i, len(tokens) - 1):
            if tokens[j] == "id":
                label = tokens[j + 1].strip(";'\"")
                break
        out.append((fen, label[:80]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def measure(boards: list[chess.Board], stored_tags: list[str | None]) -> dict:
    counts = {name: Counter() for name in DEFINITIONS}
    for i, b in enumerate(boards):
        for name, fn in DEFINITIONS.items():
            counts[name][fn(b)] += 1
        # self-check: the corpus tag should equal the builder's rule
        if stored_tags[i] is not None and stored_tags[i] != corpus_builder_phase(b):
            raise AssertionError(
                f"stored phase tag {stored_tags[i]!r} disagrees with "
                f"corpus_builder_phase on fen {b.fen()!r}"
            )
    cross = Counter()
    for b, tag in zip(boards, stored_tags):
        if tag is None:
            continue
        cross[(tag, lichess_phase(b))] += 1
    total = len(boards)
    return {
        "by_definition": {
            name: {
                "counts": dict(c),
                "pct": {p: round(100.0 * c[p] / total, 2) for p in PHASES},
                "total": total,
            }
            for name, c in counts.items()
        },
        "cross_tab_stored_tag_vs_lichess": {
            f"{r} -> {c}": n for (r, c), n in sorted(cross.items())
        },
        "disagreements_stored_tag_vs_lichess": sum(
            n for (r, c), n in cross.items() if r != c
        ),
        "positions_tagged": sum(1 for t in stored_tags if t is not None),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        action="append",
        type=Path,
        default=[],
        help="JSONL corpus with 'fen' records (manifest lines skipped); repeatable",
    )
    parser.add_argument(
        "--epd",
        action="append",
        type=Path,
        default=[],
        help="EPD file, one position per line; repeatable",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON artifact here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if not args.corpus and not args.epd:
        print("nothing to do: pass at least one --corpus or --epd", file=sys.stderr)
        return 2

    boards: list[chess.Board] = []
    stored_tags: list[str | None] = []
    inputs: list[dict] = []

    for path in args.corpus:
        recs = read_corpus_positions(path)
        for fen, tag in recs:
            boards.append(chess.Board(fen))
            stored_tags.append(tag)
        inputs.append({"kind": "corpus", "path": str(path), "positions": len(recs)})

    per_position: list[dict] = []
    for path in args.epd:
        recs = read_epd_positions(path)
        for fen, label in recs:
            b = chess.Board(fen)
            boards.append(b)
            stored_tags.append(None)
            per_position.append(
                {
                    "source": str(path),
                    "label": label,
                    "fen": fen,
                    **{name: fn(b) for name, fn in DEFINITIONS.items()},
                }
            )
        inputs.append({"kind": "epd", "path": str(path), "positions": len(recs)})

    result = measure(boards, stored_tags)
    result["inputs"] = inputs
    result["per_position_epd"] = per_position
    result["definitions"] = {
        "lichess": (
            "endgame iff majors_and_minors <= 6; middlegame iff not endgame and "
            "(majors_and_minors <= 10 or backrank sparse (<4 own pieces on own back "
            "rank) or mixedness > 150); opening otherwise. Ported from scalachess "
            "Divider.scala per arXiv:2401.16852 Appendix B."
        ),
        "lichess_material_only": "same thresholds, without the backrank/mixedness terms",
        "corpus_builder": (
            "endgame iff total pieces <= 12 or non-pawn pieces <= 4; opening iff "
            "fullmove <= 12; middlegame otherwise (tools/build_unarchitectured_metal_"
            "calibration_corpus.py::classify_phase)"
        ),
    }

    li = result["by_definition"]["lichess"]["counts"]
    mo = result["by_definition"]["lichess_material_only"]["counts"]
    cb = result["by_definition"]["corpus_builder"]["counts"]
    n = len(boards)
    reclass_vs_material_only = sum(
        1
        for b in boards
        if lichess_phase(b) != lichess_material_only_phase(b)
    )
    result["conclusion"] = (
        f"Of {n} positions: Lichess gives {li['opening']}/{li['middlegame']}/{li['endgame']} "
        f"(opening/middlegame/endgame); the corpus builder's stored tag gives "
        f"{cb['opening']}/{cb['middlegame']}/{cb['endgame']}; the two disagree on "
        f"{result['disagreements_stored_tag_vs_lichess']} tagged positions. "
        f"Dropping the backrank/mixedness terms from the Lichess definition reclassifies "
        f"{reclass_vs_material_only} of {n} positions, i.e. those terms are not decorative "
        f"on real play. No phase-specialized model exists for this repo's checkpoint; "
        f"this measurement only sizes the data question a phase-specialized retrain would face."
    )

    print(f"total positions: {n}")
    print()
    for name in ("lichess", "lichess_material_only", "corpus_builder"):
        d = result["by_definition"][name]
        c = d["counts"]
        print(
            f"{name:24s} opening={c['opening']:4d} ({d['pct']['opening']:5.2f}%)  "
            f"middlegame={c['middlegame']:4d} ({d['pct']['middlegame']:5.2f}%)  "
            f"endgame={c['endgame']:4d} ({d['pct']['endgame']:5.2f}%)"
        )
    print()
    print(
        f"stored corpus tag vs Lichess: "
        f"{result['disagreements_stored_tag_vs_lichess']} disagreements "
        f"of {result['positions_tagged']} tagged"
    )
    if per_position:
        print()
        for rec in per_position:
            print(
                f"  {rec['label'][:40]:40s} lichess={rec['lichess']} "
                f"material_only={rec['lichess_material_only']} corpus={rec['corpus_builder']}"
            )
    print()
    print(result["conclusion"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
