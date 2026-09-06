#!/usr/bin/env python3
"""Sample a phase-stratified calibration corpus from over-the-board PGN archives.

Round 6 step 1 needs positions that are *provenance-disjoint* from whatever
trained Unarchitectured Metal. This repository ships no training-membership
manifest, so record-level disjointness cannot be proven here. What this script
can and does establish is **source disjointness**: it samples from
over-the-board tournament PGN archives (e.g. TWIC), whereas the student was
trained on Lichess online play. Those two populations do not share games.

That distinction is recorded in the output manifest so downstream reports state
the real guarantee rather than an unverifiable one.

Positions are stratified by game phase (opening / middlegame / endgame) and
deduplicated by FEN. Positions matching the in-repo smoke fixtures are excluded
so this corpus stays disjoint from the eight-position trial set and from both
frozen Python parity fixtures.

Usage:
  python3 tools/build_unarchitectured_metal_calibration_corpus.py \
      --pgn /path/to/twic900.pgn --pgn /path/to/twic901.pgn \
      --output artifacts/unarchitectured-metal-calibration-corpus.jsonl \
      --positions 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pgn", action="append", required=True, type=Path, help="source PGN file (repeatable)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positions", type=int, default=600)
    parser.add_argument("--min-elo", type=int, default=2300)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-games", type=int, default=20000)
    parser.add_argument(
        "--exclude-corpus",
        action="append",
        type=Path,
        default=[],
        help="existing corpus JSONL whose FENs must not reappear (repeatable)",
    )
    return parser


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    argument_parser().parse_args()

import chess
import chess.pgn

# Excluded so the calibration corpus stays disjoint from every in-repo fixture:
# the eight-position `TRIAL_FENS` smoke set and both frozen parity positions.
EXCLUDED_FENS = {
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4rrk1/pp1b1ppp/2n1p3/2qpP3/3N4/2P1B3/PPQ2PPP/R4RK1 w - - 0 16",
    "2r3k1/5ppp/p3p3/1p1q4/3P4/P2Q4/1P3PPP/2R3K1 w - - 0 24",
    "8/5pk1/6p1/3p4/3P1P2/5KP1/8/8 w - - 0 40",
    "4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",
    chess.STARTING_FEN,
}

PHASES = ("opening", "middlegame", "endgame")


def classify_phase(board: chess.Board) -> str:
    """Bucket a position by material and ply, mirroring the smoke set's spread."""
    non_pawn = sum(
        1
        for piece in board.piece_map().values()
        if piece.piece_type not in (chess.PAWN, chess.KING)
    )
    total = len(board.piece_map())
    if total <= 12 or non_pawn <= 4:
        return "endgame"
    if board.fullmove_number <= 12:
        return "opening"
    return "middlegame"


def game_identity(headers: chess.pgn.Headers) -> str:
    """Stable identity for a source game, for provenance auditing."""
    fields = [
        headers.get("Event", "?"),
        headers.get("Site", "?"),
        headers.get("Date", "?"),
        headers.get("Round", "?"),
        headers.get("White", "?"),
        headers.get("Black", "?"),
    ]
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:16]


def parse_elo(headers: chess.pgn.Headers, key: str) -> int:
    try:
        return int(headers.get(key, "0"))
    except ValueError:
        return 0


def sample_positions(
    pgn_paths: list[Path],
    target: int,
    min_elo: int,
    seed: int,
    max_games: int,
    excluded_fens: set[str] | None = None,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    per_phase = {phase: [] for phase in PHASES}
    quota = target // len(PHASES)
    seen_fens: set[str] = set(excluded_fens or ())
    externally_excluded = len(seen_fens)
    games_read = 0
    games_used = 0
    game_ids: set[str] = set()
    source_files: list[str] = []

    for pgn_path in pgn_paths:
        source_files.append(pgn_path.name)
        with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
            while games_read < max_games:
                try:
                    game = chess.pgn.read_game(handle)
                except Exception:  # malformed entry; skip it
                    continue
                if game is None:
                    break
                games_read += 1

                headers = game.headers
                if min(parse_elo(headers, "WhiteElo"), parse_elo(headers, "BlackElo")) < min_elo:
                    continue
                result = headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue

                board = game.board()
                if board.fen() != chess.STARTING_FEN:
                    continue  # skip odds/variant starts

                # Collect this game's candidate positions, then take a small
                # random subset so one long game cannot dominate the corpus.
                candidates: list[dict] = []
                for ply, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    if ply < 6:
                        continue
                    if board.is_game_over():
                        break
                    legal_count = board.legal_moves.count()
                    if legal_count < 2:
                        continue
                    fen = board.fen()
                    if fen in EXCLUDED_FENS or fen in seen_fens:
                        continue
                    candidates.append(
                        {
                            "fen": fen,
                            "phase": classify_phase(board),
                            "ply": ply + 1,
                            "legal_count": legal_count,
                            "result": result,
                            "game_id": game_identity(headers),
                            "source_event": headers.get("Event", "?"),
                            "source_date": headers.get("Date", "?"),
                            "white_elo": parse_elo(headers, "WhiteElo"),
                            "black_elo": parse_elo(headers, "BlackElo"),
                        }
                    )

                if not candidates:
                    continue
                rng.shuffle(candidates)
                took = 0
                for candidate in candidates:
                    phase = candidate["phase"]
                    if len(per_phase[phase]) >= quota:
                        continue
                    if candidate["fen"] in seen_fens:
                        continue
                    seen_fens.add(candidate["fen"])
                    per_phase[phase].append(candidate)
                    took += 1
                    if took >= 2:  # at most two positions per source game
                        break
                if took:
                    games_used += 1
                    game_ids.add(candidates[0]["game_id"])

                if all(len(per_phase[phase]) >= quota for phase in PHASES):
                    break
        if all(len(per_phase[phase]) >= quota for phase in PHASES):
            break

    positions: list[dict] = []
    for phase in PHASES:
        positions.extend(per_phase[phase])
    rng.shuffle(positions)

    manifest = {
        "source_kind": "over-the-board tournament PGN",
        "source_files": source_files,
        "provenance_claim": (
            "Source-disjoint from the Lichess online corpus used for training. "
            "No training-membership manifest exists in this repository, so "
            "record-level disjointness is asserted at the source-population "
            "level, not proven per position."
        ),
        "games_read": games_read,
        "games_used": games_used,
        "distinct_games": len(game_ids),
        "min_elo": min_elo,
        "seed": seed,
        "phase_counts": {phase: len(per_phase[phase]) for phase in PHASES},
        "excluded_fixture_fens": len(EXCLUDED_FENS),
        "externally_excluded_fens": externally_excluded,
        "positions": len(positions),
    }
    return positions, manifest


def main() -> int:
    args = argument_parser().parse_args()

    missing = [str(path) for path in args.pgn if not path.exists()]
    if missing:
        print(f"missing PGN inputs: {', '.join(missing)}", file=sys.stderr)
        return 2

    excluded_fens: set[str] = set()
    for corpus_path in args.exclude_corpus:
        if not corpus_path.exists():
            print(f"missing --exclude-corpus input: {corpus_path}", file=sys.stderr)
            return 2
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if "manifest" not in entry:
                    excluded_fens.add(entry["fen"])

    positions, manifest = sample_positions(
        args.pgn,
        args.positions,
        args.min_elo,
        args.seed,
        args.max_games,
        excluded_fens,
    )
    if not positions:
        print("no positions sampled", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"manifest": manifest}) + "\n")
        for position in positions:
            handle.write(json.dumps(position) + "\n")

    print(json.dumps(manifest, indent=2))
    print(f"wrote {len(positions)} positions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
