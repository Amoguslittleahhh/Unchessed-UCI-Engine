#!/usr/bin/env python3
"""Build Maia-style level-conditioned move labels from the training blocks.

`docs/research-notes-maia-levels-reverse-engineering.md` reverse-engineered
the Maia opponent ladder. Its training rule for a level L is a **both-
players** window: a game contributes to level L only if *both* players'
Elo fall in [L, L+100) (plus: standard results, bullet games removed,
moves under time pressure removed, first/last moves trimmed). This tool
applies that rule over the committed `data/training/` blocks and emits the
move-level label rows a level-conditioned retrain trains on:

    one row per (position, move) with `active_elo` AND `opponent_elo`

(the dual-elo labels are exactly what the Maia-2 discrete-bucket and
Maia-3 anchor-embedding conditioning designs consume).

What is and isn't in the rows:

* In: game key, FEN before the move, move (UCI + SAN), ply, side,
  active_elo, opponent_elo, level_window, result, date, event, source
  block, and a `clock: null` / `low_time: null` placeholder.
* Out (deferred, documented): the Maia CSV's CP columns (`cp`, `cp_loss`,
  `winrate`, `is_blunder`). Those need a Stockfish pass or Lichess's
  `%eval`/`%clk` tags; the mirror's mega-clean set stripped the clock tags
  and the blocks are multi-source, so the clock columns are null today and
  the low_time filter is not applied. `--trim-first/--trim-last` (Maia-2
  defaults 10/10) is applied, since it needs no clock data.

Memory and size: the full output for all committed blocks is ~8x10^5 rows
(~250 MB JSONL), too big to commit. The tool STREAMS either way. The tool STREAMS: every row is hashed as it is produced, and when
`--sample` is set the rows are spooled to a temp file with per-window byte
offsets, so peak memory stays a couple hundred MB regardless of corpus
size. Commit the *profile* (counts + sha256 + exact parameters, via
--profile-out) and a deterministic `--sample`; both are reproducible
bit-for-bit from the pinned blocks.

Usage:
  python3 tools/build_level_conditioned_moves.py --profile-out /tmp/profile.json
  python3 tools/build_level_conditioned_moves.py --sample 200 --out sample.jsonl
  python3 tools/build_level_conditioned_moves.py --report-only
Dependencies beyond stdlib: `chess` (tools/requirements-dev.txt), and
`tools/training_blocks.py` for the shared game-boundary splitter.

Parsing note: games are parsed ONE GAME AT A TIME from the text produced by
`training_blocks.split_pgn_games` — the same path the blocks were validated
with. Whole-file `chess.pgn.read_game(handle)` stream parsing mis-assigns a
few game boundaries in these dumps (measured: 44 of 695 games in
lichess-2022-10-05/elo-0000-1400.pgn log dropped SAN tokens under stream
parsing, 0 under per-game parsing), which desyncs board positions for the
rest of the affected game. Per-game parsing is also how
`training_blocks.py clean` defined the committed 100%-legal guarantee, so
labeling on the same path keeps the FENs consistent with that guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_blocks import split_pgn_games  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLOCK_GLOB = "data/training/**/*.pgn"
RESULTS = ("1-0", "0-1", "1/2-1/2")
DEFAULT_WINDOWS = tuple(range(600, 3100, 100))  # [L, L+100) windows
SKIP_KEYS = ("unparseable_or_desync", "unrated", "other_result",
             "bullet", "no_window", "too_short")


def read_games(path: Path):
    """Yield (game, ok) per game, parsed one game at a time (see module doc)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    events: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.append(record)

    logger = logging.getLogger("chess.pgn")
    old_level, old_propagate = logger.level, logger.propagate
    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        for game_text in split_pgn_games(text):
            n_before = len(events)
            try:
                game = chess.pgn.read_game(io.StringIO(game_text))
            except Exception:
                game = None
            if game is None or len(events) > n_before:
                yield None, False  # unparseable or dropped tokens -> desynced
                continue
            yield game, True
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def game_key(game) -> str:
    h = game.headers
    moves = " ".join(m.uci() for m in game.mainline_moves())
    return hashlib.sha256(
        "|".join([h.get("White", ""), h.get("Black", ""), h.get("Date", ""),
                  h.get("Event", ""), h.get("Result", ""), moves]).encode()
    ).hexdigest()[:16]


def elo_window(white_elo: int, black_elo: int, windows) -> int | None:
    """Maia's both-players rule: the unique window containing both elos."""
    for L in windows:
        if L <= white_elo < L + 100 and L <= black_elo < L + 100:
            return L
    return None


def is_bullet(game) -> bool:
    event = game.headers.get("Event", "")
    tc = game.headers.get("TimeControl", "")
    return "bullet" in event.lower() or "bullet" in tc.lower()


def iter_rows(blocks, windows, trim_first: int, trim_last: int):
    """Stream (row, tag) pairs; O(1) memory.

    Yields ``(source, "game")`` once per game, ``(row, None)`` for label
    rows and ``(None, reason)`` for skipped games.
    """
    for block in blocks:
        try:
            rel = str(block.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(block)
        for game, ok in read_games(block):
            yield rel, "game"  # per-game marker for counting
            if not ok:
                yield None, "unparseable_or_desync"
                continue
            h = game.headers
            try:
                white_elo = int(h.get("WhiteElo", ""))
                black_elo = int(h.get("BlackElo", ""))
            except ValueError:
                yield None, "unrated"
                continue
            result = h.get("Result", "")
            if result not in RESULTS:
                yield None, "other_result"
                continue
            if is_bullet(game):
                yield None, "bullet"
                continue
            window = elo_window(white_elo, black_elo, windows)
            if window is None:
                yield None, "no_window"
                continue
            moves = list(game.mainline_moves())
            lo = min(trim_first, len(moves))
            hi = max(lo, len(moves) - trim_last)
            if hi <= lo:
                yield None, "too_short"
                continue
            key = game_key(game)
            board = chess.Board()  # fresh: game.board() would be the FINAL position
            for ply in range(len(moves)):
                move = moves[ply]
                side = "white" if board.turn == chess.WHITE else "black"
                if lo <= ply < hi:
                    row = {
                        "game_key": key,
                        "fen": board.fen(),
                        "move_uci": move.uci(),
                        "move_san": board.san(move),
                        "move_ply": ply + 1,
                        "side": side,
                        "active_elo": white_elo if side == "white" else black_elo,
                        "opponent_elo": black_elo if side == "white" else white_elo,
                        "level_window": window,
                        "result": result,
                        "date": h.get("Date", ""),
                        "event": h.get("Event", ""),
                        "source": rel,
                        "clock": None,   # deferred: mirror stripped %clk
                        "low_time": None,
                    }
                    yield row, None
                board.push(move)


def even_stride_indices(total: int, keep: int) -> set[int]:
    """Deterministic even-stride selection: exactly min(keep,total) indices."""
    if total <= keep:
        return set(range(total))
    return {i * total // keep for i in range(keep)}


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--blocks", action="append", type=Path, default=None,
                   help="PGN files (repeatable); default: all committed blocks")
    p.add_argument("--windows", type=int, nargs="*", default=None,
                   help="level windows (lower bounds, 100-wide); default 600..3000")
    p.add_argument("--trim-first", type=int, default=10)
    p.add_argument("--trim-last", type=int, default=10)
    p.add_argument("--out", type=Path, default=None,
                   help="write rows (JSONL); with --sample, only the sample")
    p.add_argument("--sample", type=int, default=None,
                   help="keep at most N rows per window (deterministic even stride)")
    p.add_argument("--profile-out", type=Path, default=None,
                   help="write the profile JSON (counts + params + sha256 of the rows)")
    p.add_argument("--report-only", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    windows = tuple(args.windows) if args.windows else DEFAULT_WINDOWS
    if args.blocks:
        blocks = args.blocks
    else:
        blocks = sorted(REPO_ROOT.glob(DEFAULT_BLOCK_GLOB))
    if not blocks:
        print("no block files found", file=sys.stderr)
        return 2

    need_spool = args.sample is not None and bool(args.out or args.profile_out)
    skipped: dict[str, int] = {k: 0 for k in SKIP_KEYS}
    per_window: dict[str, int] = {str(L): 0 for L in windows}
    per_source: dict[str, int] = {}
    games_seen = 0
    rows_total = 0
    rows_sha = hashlib.sha256()
    sample_sha = hashlib.sha256()
    preview: list[str] = []
    window_offsets: dict[str, list[int]] = {str(L): [] for L in windows}

    spool = tempfile.TemporaryFile(mode="w+b") if need_spool else None
    out_fh = None
    try:
        if args.out and args.sample is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            out_fh = args.out.open("w", encoding="utf-8")
        for row, skip_reason in iter_rows(blocks, windows, args.trim_first,
                                          args.trim_last):
            if skip_reason == "game":
                games_seen += 1
                per_source[row] = per_source.get(row, 0) + 1
                continue
            if skip_reason:
                skipped[skip_reason] += 1
                continue
            rows_total += 1
            w = str(row["level_window"])
            per_window[w] += 1
            line = json.dumps(row, separators=(",", ":"))
            payload = (line + "\n").encode("utf-8")
            rows_sha.update(payload)
            if len(preview) < 3:
                preview.append(line)
            if out_fh is not None:
                out_fh.write(line + "\n")
            if spool is not None:
                window_offsets[w].append(spool.tell())
                spool.write(payload)
    finally:
        if out_fh is not None:
            out_fh.close()

    sample_rows = 0
    if args.sample is not None:
        if spool is not None:
            # extract the even-stride sample by direct byte-offset seek
            selected = {w: even_stride_indices(n, args.sample)
                        for w, n in per_window.items() if n}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with spool, args.out.open("w", encoding="utf-8") as out_fh:
                for w in sorted(selected, key=int):
                    offs = window_offsets[w]
                    for i in sorted(selected[w]):
                        spool.seek(offs[i])
                        line = spool.readline().rstrip(b"\n")
                        out_fh.write(line.decode("utf-8") + "\n")
                        sample_sha.update(line + b"\n")
                        sample_rows += 1
        else:
            # no spool requested (no --out/--profile-out): nothing to write,
            # sample stats are uncomputable without a second pass; say so.
            sample_rows = None

    profile = {
        "tool": "tools/build_level_conditioned_moves.py",
        "blocks": [
            (str(b.relative_to(REPO_ROOT))
             if b.is_absolute() and str(b).startswith(str(REPO_ROOT)) else str(b))
            for b in blocks
        ],
        "windows": list(windows),
        "trim_first": args.trim_first,
        "trim_last": args.trim_last,
        "games_seen": games_seen,
        "rows": rows_total,
        "skipped_games": skipped,
        "per_window": {w: n for w, n in sorted(
            per_window.items(), key=lambda kv: int(kv[0])) if n},
        "per_source": per_source,
        "sha256_of_rows": rows_sha.hexdigest(),
    }
    if args.sample is not None:
        profile["sample_rows"] = sample_rows
        if sample_rows is not None:
            profile["sha256_of_sample"] = sample_sha.hexdigest()

    if args.profile_out:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        args.profile_out.write_text(json.dumps(profile, indent=2) + "\n",
                                    encoding="utf-8")

    if args.report_only:
        for line in preview:
            print(line)
        if rows_total > len(preview):
            print(f"... ({rows_total} rows total; use --out/--sample to write)")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
