#!/usr/bin/env python3
"""Collect real-human games into 100-elo bands covering 100-3200.

Complements data/training/ (six coarse bands from a three-chunk sample)
by scanning the FULL lichess 2022-10-05 mega-clean set (63 chunks,
~3.9 GB, all rated lichess games) at TEXT level — fast and silent, no
per-game parser spam — and banding each game by the MEAN of
WhiteElo/BlackElo into 100-wide bands:

    100, 200, ..., 3100, 3200        (32 requested bands)
    3300                              (overflow: mean 3300-3499)

A game is banded only if both ratings are present, numeric, and within
[100, 3500) (the same guard as tools/training_blocks.py band_for);
the 3200 band therefore covers mean 3200-3299 and the 3300 band covers
3300-3499. Games are copied VERBATIM (byte-identical per game) into one
file per band; `--cap N` keeps at most N games per band (counting
continues, so the manifest can report true availability).

Subcommands:

  count --source DIR
      Scan the PGN files under DIR (recursively, names containing .pgn)
      and report the available game count per band, no output files.
      One progress line per input file; no per-game output.

  build --source DIR --out DIR [--cap N]
      As count, but also writes <out>/elo-XXXX.pgn band files (capped)
      and <out>/manifest.json with per-band totals.

Bands built this way still need the move-legality clean before they are
trusted (tools/training_blocks.py clean per file — same rule as
data/training/). This tool is text-level only, by design: the
user-facing rule is that bulk processing of large dirty PGN dumps must
be fast and silent; malformed/illegal games are counted by the clean
step, never printed per game.

Usage:
  python3 tools/build_elo_bands.py count  --source /tmp/cd/Release/2022-10-05
  python3 tools/build_elo_bands.py build  --source /tmp/cd/Release/2022-10-05 \
      --out /tmp/elo-bands --cap 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_blocks import split_pgn_games  # noqa: E402

# 100-wide bands: lower bound -> file suffix. 3200 covers 3200-3299,
# 3300 is the overflow band for 3300-3499 (mean of two <3500 ratings).
BANDS = tuple(range(100, 3400, 100))  # 100, 200, ..., 3300
MIN_ELO, MAX_ELO = 100, 3500


def band_for(white: int, black: int) -> int | None:
    """Mean-elo band lower bound, or None if either rating is out of range."""
    if not (MIN_ELO <= white < MAX_ELO and MIN_ELO <= black < MAX_ELO):
        return None
    mean = (white + black) / 2
    lo = int(mean // 100) * 100
    lo = max(lo, 100)
    lo = min(lo, 3300)
    return lo


def _elo_tags(game: str) -> tuple[str, str]:
    tags = {}
    for line in game.splitlines():
        if line.startswith("[WhiteElo "):
            tags["w"] = line[11:-2]
        elif line.startswith("[BlackElo "):
            tags["b"] = line[11:-2]
            break
    return tags.get("w", ""), tags.get("b", "")


def cmd_count(args: argparse.Namespace) -> int:
    per_band = {b: 0 for b in BANDS}
    print(f"scanning {args.source}")
    for pgn in sorted(p for p in Path(args.source).rglob("*") if ".pgn" in p.name):
        t0 = time.time()
        n = 0
        for game in split_pgn_games(pgn.read_text(encoding="utf-8",
                                                errors="replace")):
            n += 1
            w, b = _elo_tags(game)
            try:
                band = band_for(int(w), int(b))
            except ValueError:
                continue
            if band is not None:
                per_band[band] += 1
        print(f"  {pgn.name}: {n} games in {time.time() - t0:.1f}s "
              f"(running banded total {sum(per_band.values())})", flush=True)
    report = {
        "source": str(args.source),
        "per_band": {str(b): n for b, n in per_band.items()},
        "banded_total": sum(per_band.values()),
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_band = {b: 0 for b in BANDS}      # true available count
    kept = {b: 0 for b in BANDS}
    handles = {b: (out / f"elo-{b:04d}.pgn").open("w", encoding="utf-8")
               for b in BANDS}
    try:
        for pgn in sorted(p for p in Path(args.source).rglob("*")
                          if ".pgn" in p.name):
            t0 = time.time()
            n = 0
            for game in split_pgn_games(pgn.read_text(encoding="utf-8",
                                                    errors="replace")):
                n += 1
                w, b = _elo_tags(game)
                try:
                    band = band_for(int(w), int(b))
                except ValueError:
                    continue
                if band is None:
                    continue
                per_band[band] += 1
                if args.cap is None or kept[band] < args.cap:
                    handles[band].write(game)
                    kept[band] += 1
            print(f"  {pgn.name}: {n} games in {time.time() - t0:.1f}s "
                  f"(kept total {sum(kept.values())})", flush=True)
    finally:
        for h in handles.values():
            h.close()
    manifest = {
        "source": str(args.source),
        "method": ("text-level banding by mean(WhiteElo, BlackElo) into "
                   "100-wide bands; games copied verbatim; both ratings "
                   "required in [100, 3500)"),
        "cap": args.cap,
        "per_band_available": {str(b): per_band[b] for b in BANDS},
        "per_band_kept": {str(b): kept[b] for b in BANDS},
        "kept_total": sum(kept.values()),
        "available_total": sum(per_band.values()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"kept_total": manifest["kept_total"],
                      "available_total": manifest["available_total"]},
                     indent=2))
    return 0


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("count", "build"):
        s = sub.add_parser(name)
        s.add_argument("--source", required=True,
                       help="directory containing the PGN files")
        if name == "build":
            s.add_argument("--out", required=True)
            s.add_argument("--cap", type=int, default=None,
                           help="max games kept per band (counting continues)")
    sub.choices["count"].set_defaults(fn=cmd_count)
    sub.choices["build"].set_defaults(fn=cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
