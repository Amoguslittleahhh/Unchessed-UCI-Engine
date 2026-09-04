#!/usr/bin/env python3
"""Training-data block management: fetch, band, validate, verify.

Why this tool exists
--------------------
This repo's training pipelines (NNUE, Unarchitectured v1) need game data
spanning all player levels. The sandbox cannot reach the usual data hosts
(lichess.org, archive.org, twic.sesse.net — see the egress audit in
docs/dev-environment.md), but it can reach GitHub over git, and the PGN
mirror repo `rozim/ChessData` (pinned by commit in the manifest) commits
the data as regular git blobs. So blocks are fetched by partial clone,
banded by rating at text level, and validated by sampled move-legal
parsing.

Design notes
------------
* **Banding is text-level and lossless.** PGN games are split with a small
  state machine (tag section / movetext / blank-line boundary), ratings
  are read from the TAG roster only, and each game's original text is
  copied verbatim into its band file. No re-serialization, no move
  application — this is what makes a 50 MB dirty dump process in seconds
  instead of minutes. Games with a missing/out-of-range rating or a
  non-standard result are quarantined (counted, written to a
  ``.quarantine`` file only with --keep-quarantine), never mixed into a
  rated band.
* **Validation is sampled.** python-chess applies every move and raises
  on illegal ones, which is exactly the "spam of illegal games" failure
  mode on dirty mirrors; so ``validate`` fully parses a deterministic
  seeded sample (default 200 games) and reports the legal rate instead
  of grinding through the whole file.
* **Provenance is pinned.** Every block records the source repo commit,
  the in-repo path, and the SHA-256 of the block file. ``verify`` re-checks
  the hashes; ``fetch`` re-derives a block from the source repo.

Bands (by the MEAN of WhiteElo and BlackElo, both required, both within
[100, 3500]):
    elo-0000-1400, elo-1400-1700, elo-1700-2000,
    elo-2000-2300, elo-2300-2600, elo-2600-3500

Usage:
  # split a source PGN into rating bands (fast, text-level)
  python3 tools/training_blocks.py split SOURCE.pgn --prefix out/band

  # sampled validation + band stats (JSON)
  python3 tools/training_blocks.py validate FILE.pgn [--sample 200] [--seed 7]

  # full move-legality validation, rewriting the file with legal games only
  # (run once per block before committing; silent, counted)
  python3 tools/training_blocks.py clean FILE.pgn [--dry-run]

  # re-check committed blocks against the manifest (hashes + counts)
  python3 tools/training_blocks.py verify [--manifest data/training/manifest.json]

  # re-fetch source files from the pinned repo (needs GitHub egress)
  python3 tools/training_blocks.py fetch --out /path/to/dir

Dependencies beyond stdlib: `chess` (tools/requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

BANDS = (
    (0, 1400, "elo-0000-1400"),
    (1400, 1700, "elo-1400-1700"),
    (1700, 2000, "elo-1700-2000"),
    (2000, 2300, "elo-2000-2300"),
    (2300, 2600, "elo-2300-2600"),
    (2600, 3500, "elo-2600-3500"),
)
RATING_MIN = 100
RATING_MAX = 3500
RESULTS = ("1-0", "0-1", "1/2-1/2")

TAG_RE = re.compile(r'^\[(\w+)\s+"((?:[^"\\]|\\.)*)"\]\s*$')
ELO_KEYS = ("WhiteElo", "BlackElo")

# Source of record for the committed blocks (pinned; see manifest).
DEFAULT_SOURCE = {
    "repo": "https://github.com/rozim/ChessData.git",
    "commit": "ed88abd2716da58ee55d42b662455c1c8ebe0776",
    "fetched": "2026-08-26",
    "note": (
        "PGN mirror; the upstream README warns 'There will be dups, dirty "
        "data, errors, GM draws etc -- the data will probably need to be "
        "post-processed, filtered, deduped'. This tool's banding is that "
        "first filter pass; blocks are sampled-validated, not fully "
        "move-legal."
    ),
}


# ---------------------------------------------------------------------------
# Text-level PGN splitting
# ---------------------------------------------------------------------------

def split_pgn_games(text: str) -> list[str]:
    """Split PGN text into per-game text blocks (verbatim, no reformat).

    State machine: TAG section lines start with '['; the first non-tag,
    non-blank line after tags begins the movetext; a blank line (or the
    start of a new tag section, or EOF) ends the current game. Comments
    containing blank lines are not anticipated — that is what the sampled
    python-chess validation in `validate` is for.
    """
    games: list[list[str]] = []
    cur: list[str] = []
    in_tags = False
    in_moves = False

    def flush() -> None:
        if cur:
            games.append("\n".join(cur).rstrip() + "\n")
            cur.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if in_moves:
            if stripped == "":
                # blank line ends the current game
                cur.append(line)
                flush()
                in_tags, in_moves = False, False
            elif stripped.startswith("["):
                # a new game starts mid-stream without a blank line:
                # flush FIRST (the tag line belongs to the next game only)
                flush()
                in_tags, in_moves = True, False
                cur.append(line)
            else:
                cur.append(line)
        else:
            if stripped.startswith("["):
                in_tags = True
                cur.append(line)
            elif stripped == "":
                if in_tags:
                    # the mandatory tag-section/movetext separator: keep it
                    cur.append(line)
                # blank lines outside any game: ignore
            else:
                if in_tags:
                    in_moves = True
                cur.append(line)
    flush()
    return games


def game_tags(game_text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line in game_text.splitlines():
        if not line.startswith("["):
            if line.strip() and not line.startswith("["):
                break
            continue
        m = TAG_RE.match(line)
        if m:
            tags[m.group(1)] = m.group(2)
    return tags


def band_for(white_elo: int, black_elo: int) -> str | None:
    if not (RATING_MIN <= white_elo <= RATING_MAX and RATING_MIN <= black_elo <= RATING_MAX):
        return None
    mean = (white_elo + black_elo) / 2
    for lo, hi, name in BANDS:
        if lo <= mean < hi or (hi == RATING_MAX and mean <= hi):
            return name
    return None


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_split(args: argparse.Namespace) -> int:
    src = Path(args.source)
    text = src.read_text(encoding="utf-8", errors="replace")
    games = split_pgn_games(text)
    prefix = Path(args.prefix)
    prefix.mkdir(parents=True, exist_ok=True)

    out: dict[str, list[str]] = {name: [] for _, _, name in BANDS}
    out["_quarantine"] = []
    counts = {"total_games": len(games)}
    for g in games:
        tags = game_tags(g)
        result = tags.get("Result", "")
        try:
            w = int(tags.get("WhiteElo", ""))
            b = int(tags.get("BlackElo", ""))
        except ValueError:
            w = b = 0
        name = band_for(w, b) if result in RESULTS else None
        if name is None:
            out["_quarantine"].append(g)
        else:
            out[name].append(g)

    written: dict[str, dict] = {}
    for name, body in out.items():
        if not body:
            continue
        fname = prefix / (name + ".pgn")
        fname.write_text("".join(body), encoding="utf-8")
        entry = {"games": len(body), "bytes": fname.stat().st_size,
                 "sha256": hashlib.sha256(fname.read_bytes()).hexdigest()}
        if name == "_quarantine":
            if not args.keep_quarantine:
                fname.unlink()
            entry["kept"] = args.keep_quarantine
        written[name] = entry
    counts["by_band"] = {k: v["games"] for k, v in written.items() if k != "_quarantine"}
    counts["quarantined"] = written.get("_quarantine", {}).get("games", 0)
    counts["source_bytes"] = src.stat().st_size
    print(json.dumps(counts, indent=2))
    return 0


def _parse_one_game(text: str) -> object:
    """Parse one game and decide its move-legality.

    python-chess 1.x does NOT raise on illegal SAN: it logs a warning on
    the ``chess.pgn`` logger and silently drops the bad token (the rest of
    the movetext may be truncated too). So the warning record itself is
    the illegality signal; we capture it, keep the console quiet, and also
    replay the parsed mainline as a second check.
    """
    import io
    import logging

    import chess.pgn

    events: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.append(record)

    logger = logging.getLogger("chess.pgn")
    old_level, old_propagate = logger.level, logger.propagate
    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # no stderr spam; we count the records
    try:
        g = chess.pgn.read_game(io.StringIO(text))
        if g is None:
            return "unparseable"
        if any(e.levelno >= logging.WARNING for e in events):
            return "illegal"
        board = g.board()
        for mv in g.mainline_moves():
            board.push(mv)
        return g
    except Exception:
        return "illegal"
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def cmd_validate(args: argparse.Namespace) -> int:
    import chess.pgn  # noqa: F401  (ensure availability early)

    path = Path(args.file)
    text = path.read_text(encoding="utf-8", errors="replace")
    games = split_pgn_games(text)
    rng = random.Random(args.seed)
    sample_idx = sorted(rng.sample(range(len(games)), min(args.sample, len(games))))

    legal = illegal = unparseable = 0
    band_counts: dict[str, int] = {}
    quarantined = 0
    for i, g in enumerate(games):
        tags = game_tags(g)
        try:
            w = int(tags.get("WhiteElo", ""))
            b = int(tags.get("BlackElo", ""))
        except ValueError:
            w = b = 0
        if tags.get("Result") in RESULTS and band_for(w, b):
            band_counts[band_for(w, b)] = band_counts.get(band_for(w, b), 0) + 1
        else:
            quarantined += 1
        if i in set(sample_idx):
            r = _parse_one_game(g)
            if r == "illegal":
                illegal += 1
            elif r is None or r == "unparseable":
                unparseable += 1
            else:
                legal += 1

    n = len(sample_idx)
    report = {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_games": len(games),
        "band_counts": band_counts,
        "quarantined": quarantined,
        "sample_size": n,
        "sample_seed": args.seed,
        "sample_legal": legal,
        "sample_illegal": illegal,
        "sample_unparseable": unparseable,
        "sample_legal_rate": round(legal / n, 4) if n else None,
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Full move-legality validation; rewrite the file with legal games only.

    This is the slow path (every game is fully parsed), used once per
    block before committing so a committed block is 100% move-legal, not
    just sample-legal. Silent: parser warnings are captured and counted,
    not printed.
    """
    import chess.pgn  # noqa: F401  (fail fast if the dependency is missing)

    path = Path(args.file)
    text = path.read_text(encoding="utf-8", errors="replace")
    games = split_pgn_games(text)
    kept: list[str] = []
    illegal = unparseable = 0
    for g in games:
        r = _parse_one_game(g)
        if r == "illegal":
            illegal += 1
        elif r is None or r == "unparseable":
            unparseable += 1
        else:
            kept.append(g)
    if not args.dry_run:
        path.write_text("".join(kept), encoding="utf-8")
    report = {
        "file": str(path),
        "games_total": len(games),
        "games_kept": len(kept),
        "illegal": illegal,
        "unparseable": unparseable,
        "legal_rate_full": round(len(kept) / len(games), 4) if games else None,
        "written": not args.dry_run,
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT() / "data/training/manifest.json"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    base = REPO_ROOT()  # block paths are relative to the repo root
    failures = 0
    for block in manifest["blocks"]:
        p = base / block["path"]
        if not p.is_file():
            print(f"MISSING {block['path']}")
            failures += 1
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        size = p.stat().st_size
        ok = sha == block["sha256"] and size == block["bytes"]
        print(f"{'OK     ' if ok else 'DRIFT  '} {block['path']} "
              f"({size} bytes, {block['games']} games)")
        if not ok:
            failures += 1
    print(f"{len(manifest['blocks'])} blocks checked, {failures} drift(s)")
    return 1 if failures else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    if shutil.which("git") is None:
        print("git not found on PATH", file=sys.stderr)
        return 2
    src = DEFAULT_SOURCE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clone_dir = out / "_chessdata"
    if not clone_dir.exists():
        print(f"partial clone of {src['repo']} @ {src['commit'][:12]} "
              f"(needs GitHub egress)...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
             src["repo"], str(clone_dir)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(clone_dir), "checkout", "-q", src["commit"]],
        check=True,
    )
    print(f"ready: {clone_dir} @ {src['commit']}", file=sys.stderr)
    return 0


def REPO_ROOT() -> Path:
    return Path(__file__).resolve().parent.parent


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="band a source PGN by mean player rating (text-level, lossless)")
    sp.add_argument("source", type=Path)
    sp.add_argument("--prefix", required=True, type=Path, help="output directory for band files")
    sp.add_argument("--keep-quarantine", action="store_true",
                    help="also write the _quarantine.pgn file (excluded from rated bands)")
    sp.set_defaults(fn=cmd_split)

    vp = sub.add_parser("validate", help="header stats + seeded sampled move-legality parse")
    vp.add_argument("file", type=Path)
    vp.add_argument("--sample", type=int, default=200)
    vp.add_argument("--seed", type=int, default=7)
    vp.set_defaults(fn=cmd_validate)

    cp = sub.add_parser("clean", help="FULL move-legality validation; rewrite file keeping legal games only")
    cp.add_argument("file", type=Path)
    cp.add_argument("--dry-run", action="store_true", help="report without rewriting")
    cp.set_defaults(fn=cmd_clean)

    vp2 = sub.add_parser("verify", help="re-check committed blocks against the manifest")
    vp2.add_argument("--manifest", type=Path, default=None)
    vp2.set_defaults(fn=cmd_verify)

    fp = sub.add_parser("fetch", help="partial-clone the pinned source repo (needs GitHub egress)")
    fp.add_argument("--out", required=True, type=Path)
    fp.set_defaults(fn=cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
