#!/usr/bin/env python3
"""Curate and maintain the chess-history archive in data/archive/.

The archive is the finest breadth of human play collectable from reachable
sources: games from 1834 (De la Bourdonnais-McDonnell) to 2022, across
world championships, candidates, national championships, the Polgar/Kosteniuk
women's archives, ICCF correspondence chess, GM-annotated classics, and a
2600+-rated megadatabase slice. Sources (pinned commits) and the exact
file-by-file layout live in `tools/archive_layout.json`; this tool keeps the
build reproducible from a fresh clone.

Subcommands:

  fetch --stage DIR
      Partial-clone the pinned source repos into DIR (git egress required;
      only the files listed in the layout are downloaded).

  build --stage DIR
      Stage -> data/archive/: games are copied VERBATIM (byte-identical per
      game); files whose games span several eras (`split_by_era` note) are
      partitioned by the Date year of each game; every destination file then
      gets the full move-legality clean (per-game parse, dropped tokens =>
      game dropped, silent and counted — the same rule as
      training_blocks.py clean); cross-file duplicates are scanned; and
      manifest.json is written with sha256, game counts, year spans and the
      per-file legality drops.

  verify
      Re-check every block against manifest.json (sha256 + game count).

The legality clean is per-GAME, never per-token: python-chess 1.11 does not
raise on illegal SAN — it logs on the `chess.pgn` logger and silently drops
the token, so a game that logs even one warning is dropped whole (its board
state is desynced from then on). See tools/training_blocks.py.

Dependencies beyond stdlib: `chess` (tools/requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_blocks import _parse_one_game, split_pgn_games  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = REPO_ROOT / "data" / "archive"
MANIFEST = ARCHIVE_ROOT / "manifest.json"
LAYOUT = Path(__file__).resolve().parent / "archive_layout.json"

# Pinned sources: key -> (repo, commit). Commits are the tree hashes the
# layout table was generated against.
SOURCES = {
    "chessdata": ("rozim/ChessData",
                  "ed88abd2716da58ee55d42b662455c1c8ebe0776"),
    "scoutfish": ("mcostalba/scoutfish",
                  "00cec1339f97114a32c30080dbad5e3a500634f2"),
    "annotated": ("hegde10122/CHESS_ANNOTATED_GAMES",
                  "550bdd161514b48abd5008ee4a7daa0db7718f66"),
    "pepper": ("saikrishna-1996/deep_pepper_chess",
               "b05bfe2e6defad7a85d6099ad5d69e1b46888eb5"),
}

# Era buckets for split_by_era files (inclusive year bounds).
ERA_BUCKETS = [
    ("classics-1834-1899", 0, 1899),
    ("1900-1945", 1900, 1945),
    ("1946-1970", 1946, 1970),
    ("1971-1999", 1971, 1999),
    ("2000-plus", 2000, 9999),
]


def load_layout() -> dict:
    return json.loads(LAYOUT.read_text(encoding="utf-8"))


def era_for_year(year: int | None) -> str | None:
    if year is None:
        return None
    for name, lo, hi in ERA_BUCKETS:
        if lo <= year <= hi:
            return name
    return None


def game_year(game_text: str) -> int | None:
    tags = {}
    for line in game_text.splitlines():
        m = re.match(r'^\[(\w+) "([^"]*)"\]', line)
        if m:
            tags[m.group(1)] = m.group(2)
    d = tags.get("Date", "")
    if re.match(r"^\d{4}", d):
        return int(d[:4])
    return None


def clean_file(path: Path) -> tuple[int, int]:
    """Full move-legality clean (in place). Returns (kept, dropped)."""
    games = split_pgn_games(path.read_text(encoding="utf-8", errors="replace"))
    kept_text: list[str] = []
    kept = dropped = 0
    for g in games:
        r = _parse_one_game(g)
        if r is None or r == "unparseable" or r == "illegal":
            dropped += 1
        else:
            kept += 1
            kept_text.append(g)
    if dropped:
        # each split chunk already ends with exactly one newline
        path.write_text("".join(kept_text), encoding="utf-8")
    return kept, dropped


def build(staged: dict[str, Path], force: bool = False) -> dict:
    """stage: layout key -> local path of the fetched source file."""
    layout = load_layout()
    if ARCHIVE_ROOT.exists():
        stray = [p for p in ARCHIVE_ROOT.rglob("*") if p.is_file()]
        if stray and not force:
            raise SystemExit(
                f"data/archive already holds {len(stray)} files; "
                "re-run with --force to rebuild it")
    blocks: list[dict] = []
    dup_keys: Counter[str] = Counter()

    for key, entry in layout.items():
        src_key, repo_path = key.split(":", 1)
        src = staged.get(key)
        if src is None or not src.exists():
            raise SystemExit(f"missing staged file for {key}: {src}")
        src_games = split_pgn_games(src.read_text(encoding="utf-8",
                                                  errors="replace"))
        split_era = "split_by_era" in entry.get("note", "")
        if not entry.get("file") or (not split_era and not entry.get("dir")):
            raise SystemExit(f"layout entry {key!r} missing file/dir: {entry}")
        if split_era:
            by_era: dict[str, list[str]] = {}
            years: Counter[int] = Counter()
            for g in src_games:
                y = game_year(g)
                if y is not None:
                    years[y] += 1
            majority = years.most_common(1)[0][0] if years else None
            for g in src_games:
                y = game_year(g)
                era = era_for_year(y) or era_for_year(majority)
                if era:
                    by_era.setdefault(era, []).append(g)
            for era in sorted(by_era):
                dest = ARCHIVE_ROOT / era / f"{entry['file'].replace('.pgn', '')}-{era}.pgn"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("".join(by_era[era]), encoding="utf-8")
                kept, dropped = clean_file(dest)
                blocks.append(_block_record(dest, entry, era, src_key,
                                            repo_path, kept, dropped,
                                            len(by_era[era])))
        else:
            dest = ARCHIVE_ROOT / entry["dir"] / entry["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text(encoding="utf-8", errors="replace"),
                            encoding="utf-8")
            kept, dropped = clean_file(dest)
            blocks.append(_block_record(dest, entry, entry["dir"], src_key,
                                        repo_path, kept, dropped,
                                        len(src_games)))
        for gtext in split_pgn_games(
                (REPO_ROOT / blocks[-1]["path"]).read_text(
                    encoding="utf-8", errors="replace")):
            dup_keys[_game_key(gtext)] += 1

    dups = sum(c - 1 for c in dup_keys.values() if c > 1)
    manifest = {
        "generated": "2026-08-26",
        "purpose": ("Finest breadth of human chess from 1834 to 2022 across "
                    "world championships, candidates, national events, "
                    "women's chess, correspondence and annotated classics. "
                    "Complements the rating-banded data/training/ set."),
        "source_repos": {k: {"repo": v[0], "commit": v[1]}
                         for k, v in SOURCES.items()},
        "method": ("Games copied verbatim from the pinned sources; "
                   "split_by_era files partitioned by per-game Date year; "
                   "every file then move-legality-cleaned (per-game parse, "
                   "games with dropped SAN tokens dropped and counted). "
                   "No dedup — cross-file duplicates are counted and "
                   "documented."),
        "blocks": blocks,
        "totals": {
            "files": len(blocks),
            "bytes": sum(b["bytes"] for b in blocks),
            "games": sum(b["games"] for b in blocks),
            "illegal_games_dropped": sum(b["illegal_games_dropped"] for b in blocks),
            "cross_file_duplicate_games": dups,
        },
        "layout_table": str(LAYOUT.relative_to(REPO_ROOT)),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _game_key(g: str) -> str:
    tags = {}
    for line in g.splitlines():
        m = re.match(r'^\[(\w+) "([^"]*)"\]', line)
        if m:
            tags[m.group(1)] = m.group(2)
    moves = re.sub(r"^\s*\d+\.(\.\.)?\s*", "",
                   g.split("\n\n", 1)[-1] if "\n\n" in g else g)
    raw = "|".join([tags.get("White", ""), tags.get("Black", ""),
                    tags.get("Date", ""), tags.get("Result", ""),
                    " ".join(moves.split())])
    return hashlib.sha1(raw.encode()).hexdigest()


def _block_record(dest: Path, entry: dict, era: str, src_key: str,
                  repo_path: str, kept: int, dropped: int,
                  before: int) -> dict:
    years: list[int] = []
    rated = 0
    for g in split_pgn_games(dest.read_text(encoding="utf-8",
                                              errors="replace")):
        y = game_year(g)
        if y is not None:
            years.append(y)
        for line in g.splitlines():
            if line.startswith("[WhiteElo ") and line[10:-1] not in ("", "0"):
                rated += 1
                break
    return {
        "path": str(dest.relative_to(REPO_ROOT)),
        "theme": era,
        "source": {"key": src_key, "repo": SOURCES[src_key.split(":")[0]][0],
                   "commit": SOURCES[src_key.split(":")[0]][1],
                   "path": repo_path},
        "bytes": dest.stat().st_size,
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "games": kept,
        "games_total_before_clean": before,
        "illegal_games_dropped": dropped,
        "rated_games": rated,
        "years": [min(years), max(years)] if years else None,
        "note": entry.get("note", ""),
    }


def cmd_fetch(args: argparse.Namespace) -> int:
    layout = load_layout()
    stage = Path(args.stage)
    for key in layout:
        src, repo_path = key.split(":", 1)
        repo, commit = SOURCES[src]
        dst = stage / src
        if not (dst / ".git").exists():
            if src in ("scoutfish", "annotated", "pepper"):
                subprocess.run(["git", "clone", "-q", "--depth", "1",
                                f"https://github.com/{repo}", str(dst)],
                               check=True)
                head = subprocess.run(
                    ["git", "-C", str(dst), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True).stdout.strip()
                if head != commit:
                    raise SystemExit(
                        f"{src}: HEAD {head} != pinned {commit}; "
                        "update tools/archive_layout.json + SOURCES")
            else:
                subprocess.run(["git", "clone", "-q", "--filter=blob:none",
                                "--no-checkout", f"https://github.com/{repo}",
                                str(dst)], check=True)
                subprocess.run(["git", "-C", str(dst), "fetch", "-q", "origin",
                                commit], check=True)
        have = dst / repo_path
        if not have.exists():
            if (dst / ".git").exists():
                ref = commit if src == "chessdata" else "HEAD"
                subprocess.run(["git", "-C", str(dst), "checkout", "-q", ref,
                                "--", repo_path], check=True)
    print(f"staged {len(layout)} source files under {stage}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    layout = load_layout()
    staged = {}
    for key in layout:
        src, repo_path = key.split(":", 1)
        staged[key] = Path(args.stage) / src / repo_path
    manifest = build(staged, force=args.force)
    t = manifest["totals"]
    print(json.dumps(t, indent=2))
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    if not MANIFEST.exists():
        print("no manifest.json — run build first", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = 0
    for b in manifest["blocks"]:
        p = REPO_ROOT / b["path"]
        ok = (p.exists()
              and p.stat().st_size == b["bytes"]
              and hashlib.sha256(p.read_bytes()).hexdigest() == b["sha256"]
              and len(split_pgn_games(p.read_text(encoding="utf-8",
                                                 errors="replace"))) == b["games"])
        print(("OK      " if ok else "DRIFT   ") + b["path"])
        failures += 0 if ok else 1
    print(f"{len(manifest['blocks'])} blocks checked, {failures} drift(s)")
    return 1 if failures else 0


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="stage the pinned source files")
    f.add_argument("--stage", required=True)
    b = sub.add_parser("build", help="stage -> data/archive/ + manifest")
    b.add_argument("--stage", required=True)
    b.add_argument("--force", action="store_true",
                   help="rebuild even if data/archive/ is not empty")
    v = sub.add_parser("verify", help="re-check blocks against the manifest")
    f.set_defaults(fn=cmd_fetch)
    b.set_defaults(fn=cmd_build)
    v.set_defaults(fn=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
