#!/usr/bin/env python3
"""CPU stage of the move-prediction pretrain pipeline: PGN -> v5 shards.

This is the CPU-side work of the two-stage pretrain
(docs/move-prediction-pretrain-plan.md): it turns labeled PGN files —
the cloud generator's output (WhiteElo/BlackElo + WhiteEngine/
BlackEngine + EloQuality headers), data/training-elo (real rated
humans), or data/selfplay — into **Unchessed data record v5** shards
that the GPU stage (tools/pretrain_v1_a100.py) memory-maps directly.

v5 = the frozen v4 wire record (UNCHD4R0 layout, 1088 bytes) with its
48 reserved bytes redefined as `elo_oppo:u16 + pretrain_quality:u1 +
pad:45`, a new magic UNCHD5R0, version 5, and a new schema SHA-256.
The mover's own elo stays in the existing `rating` field (it was
always the mover's rating), so v5 carries the DUAL-elo pair the
level-conditioned objective needs. v4 tools reject v5 files (magic)
and v5 readers reject v4 (magic) — no accidental mixing.

Per-record semantics for pretrain rows:
  rating         mover elo (u16, 100-3200)
  reserved.0     opponent elo (u16)
  reserved.2     pretrain quality: 0 calibrated (maia3), 1 native
                 (stockfish), 2 approximate (lc0/rubichess ladders),
                 3 human (real rated game)
  target_action  the played move (16-bit v4 action encoding:
                 from|to<<6|promo<<12|kind<<14)
  legal_actions  the position's legal moves, same encoding
  policy_kind    0 (human) for maia3 + human rows, 1 (guide) for
                 stockfish + lc0 + rubichess
  wdl            GAME-OUTCOME PROXY (side-to-move's result in the
                 game: 2 win / 1 draw / 0 loss). Stage-1 trains policy
                 CE only and does not use it; do not treat it as a
                 per-position value label.
  history        the previous up-to-8 plies of the game, same 16-bit
                 encoding (kind bit set for promotions)
  teacher_*      sentinel/zero — no teacher signals in pretrain data

Guards:
  * the played move must be in the legal set (hard fail otherwise),
  * train/val split is by GAME (never by row) — the same game's moves
    can only ever appear on one side,
  * elo/quality domains are validated per row,
  * headers without WhiteElo/BlackElo are skipped (counted, reported).

Usage:
  python3 tools/pretrain_v5_data.py build \
      --pgn /data/mixed-5m/pgn/shard-*.pgn \
      --pgn data/training-elo/elo-*.pgn \
      --out /data/pretrain-v5 --val-games 20000
  python3 tools/pretrain_v5_data.py validate --dir /data/pretrain-v5

Build design (fast on many-core boxes, 180 or 360 vCPU):
  pass 1  text-only scan (no chess parsing) counts, per file, the games
          with valid dual-elo headers (and passes the quality filter),
          in parallel over files;
  pass 2  one worker per PGN file replays its games (full legality),
          streaming records straight into the final shard files —
          no temporary files, so disk usage equals the dataset size.
  Output shards are per source file:
      train/shard-<file:03d>-<seq:03d>.v5   val/shard-...v5
  in exact game order, so the output bytes are identical regardless
  of --workers (the split is decided per game by the global cutoff).
  The GPU stage consumes them via glob: train/shard-*.v5.
  Sizes: 1M mixed games ≈ 65M records ≈ 71 GB; 5M ≈ 356 GB.

Dependencies: python-chess (tools/requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn

# ----------------------------------------------------------------------
# v5 wire ABI (v4 layout + redefined 48-byte reserved area)
# ----------------------------------------------------------------------

MAGIC = b"UNCHD5R0"
VERSION = 5
HEADER_BYTES = 64
RECORD_BYTES = 1088
ENDIAN_MARKER = 0x01020304
MANDATORY_FLAGS = 0x00FF
MAX_LEGAL_ACTIONS = 218
ACTION_SENTINEL = 0xFFFF

POLICY_HUMAN = 0
POLICY_GUIDE = 1

QUALITY_CALIBRATED = 0  # maia3 (native conditioning)
QUALITY_NATIVE = 1      # stockfish (native UCI_Elo)
QUALITY_APPROXIMATE = 2  # lc0 / rubichess (ladders, uncalibrated value)
QUALITY_HUMAN = 3
QUALITY_NAMES = ("calibrated", "native", "approximate", "human")

# engine header value -> quality
ENGINE_QUALITY = {
    "maia3": QUALITY_CALIBRATED,
    "stockfish": QUALITY_NATIVE,
    "lc0": QUALITY_APPROXIMATE,
    "rubichess": QUALITY_APPROXIMATE,
}
# engine -> policy_kind (0 human, 1 guide)
ENGINE_POLICY_KIND = {
    "maia3": POLICY_HUMAN,
    "human": POLICY_HUMAN,
    "stockfish": POLICY_GUIDE,
    "lc0": POLICY_GUIDE,
    "rubichess": POLICY_GUIDE,
}

SCHEMA_DESCRIPTOR = (
    "Unchessed pretrain data record v5;little-endian;"
    "header=magic:8,version:u16,header_bytes:u16,record_bytes:u16,"
    "flags:u16,endian:u32,records:u64,schema_sha256:32,crc32:u32;"
    "record=v3_semantics:160,legal_count:u16,target_action:u16,"
    "teacher_best_action:u16,policy_kind:u8,legal_flags:u8,"
    "legal_actions:218xu16,legal_regrets:218xi16,"
    "reserved=elo_oppo:u16,pretrain_quality:u8,pad:45"
)
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_DESCRIPTOR.encode("ascii")).digest()
HEADER_PREFIX = struct.Struct("<8sHHHHIQ32s")
HEADER = struct.Struct("<8sHHHHIQ32sI")

# The v3 base record (160 bytes) is the frozen struct from
# unarchitectured_metal_base_data: 12 bitboards, move, promotion, wdl,
# rating, castling, ep_file, halfmove, time_class, flags, history_len,
# history(8), game_hash, player_hash, teacher_score, best_move,
# best_score, move_score, ply, remaining_ms, increment_ms, reserved.
# v5 appends the redefined 48-byte reserved area:
TAIL_V5 = struct.Struct("<HHHBB218H218hHB45s")  # 8+436+436+48 = 928
assert TAIL_V5.size == 928


def make_header(record_count: int) -> bytes:
    prefix = HEADER_PREFIX.pack(
        MAGIC, VERSION, HEADER_BYTES, RECORD_BYTES, MANDATORY_FLAGS,
        ENDIAN_MARKER, record_count, SCHEMA_SHA256,
    )
    return prefix + struct.pack("<I", zlib.crc32(prefix) & 0xFFFFFFFF)


def parse_header(payload: bytes) -> int:
    if len(payload) != HEADER_BYTES:
        raise ValueError(f"header has {len(payload)} bytes, expected "
                         f"{HEADER_BYTES}")
    (magic, version, header_bytes, record_bytes, flags, endian, count,
     digest, crc) = HEADER.unpack(payload)
    if magic != MAGIC:
        raise ValueError(f"bad data magic {magic!r} (expected {MAGIC!r})")
    if (version, header_bytes, record_bytes) != (VERSION, HEADER_BYTES,
                                                 RECORD_BYTES):
        raise ValueError("unsupported v5 wire version/width")
    if flags != MANDATORY_FLAGS or endian != ENDIAN_MARKER:
        raise ValueError("bad flags/endian")
    if digest != SCHEMA_SHA256:
        raise ValueError("v5 schema SHA-256 mismatch")
    if crc != zlib.crc32(payload[: HEADER_PREFIX.size]) & 0xFFFFFFFF:
        raise ValueError("header CRC32 mismatch")
    return count


# ----------------------------------------------------------------------
# board -> record fields (STM-normalized, matching unchessed-datagen)
# ----------------------------------------------------------------------

PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
               chess.QUEEN, chess.KING)
PROMO_TO_BIT = {None: 0, chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2,
                chess.QUEEN: 3}


def v5_action(mv: chess.Move) -> int:
    """16-bit action encoding in the STM-normalized view:
    from|to<<6|promo01<<12|1<<14."""
    frm = chess.square_mirror(mv.from_square)
    to = chess.square_mirror(mv.to_square)
    promo = PROMO_TO_BIT[mv.promotion]
    if mv.promotion is None:
        return frm | (to << 6)
    return frm | (to << 6) | (promo << 12) | (1 << 14)


def v5_action_of_board_move(board: chess.Board, mv: chess.Move) -> int:
    """Same encoding without mirroring (board already normalized)."""
    frm, to = mv.from_square, mv.to_square
    promo = PROMO_TO_BIT[mv.promotion]
    if mv.promotion is None:
        return frm | (to << 6)
    return frm | (to << 6) | (promo << 12) | (1 << 14)


def bitboards_stm(board: chess.Board) -> tuple[int, ...]:
    if board.turn == chess.BLACK:
        board = board.mirror()
    planes = [int(board.pieces(pt, chess.WHITE)) for pt in PIECE_ORDER]
    planes += [int(board.pieces(pt, chess.BLACK)) for pt in PIECE_ORDER]
    return tuple(planes)


def castling_byte(board: chess.Board) -> int:
    if board.turn == chess.BLACK:
        board = board.mirror()
    c = 0
    if board.has_castling_rights(chess.H1):
        c |= 1
    if board.has_castling_rights(chess.A1):
        c |= 2
    if board.has_castling_rights(chess.H8):
        c |= 4
    if board.has_castling_rights(chess.A8):
        c |= 8
    return c


def pack_record(bb: tuple, move: int, promotion: int, wdl: int, rating: int,
                castling: int, ep_file: int, halfmove: int, time_class: int,
                flags: int, history_len: int, history: tuple, game_hash: int,
                player_hash: int, teacher_score: int, best_move: int,
                best_score: int, move_score: int, ply: int,
                remaining_ms: int, increment_ms: int, base_reserved: int,
                legal_count: int, target_action: int,
                teacher_best_action: int, policy_kind: int,
                legal_flags: int, legal_actions: tuple, legal_regrets: tuple,
                elo_oppo: int, pretrain_quality: int) -> bytes:
    from unarchitectured_metal_base_data import RECORD as V3_STRUCT
    assert len(history) == 8
    base_bytes = V3_STRUCT.pack(
        *bb, move, promotion, wdl, rating, castling, ep_file, halfmove,
        time_class, flags, history_len, *history, game_hash, player_hash,
        teacher_score, best_move, best_score, move_score, ply,
        remaining_ms, increment_ms, base_reserved,
    )
    tail = TAIL_V5.pack(
        legal_count, target_action, teacher_best_action, policy_kind,
        legal_flags, *legal_actions, *legal_regrets,
        elo_oppo, pretrain_quality, b"\x00" * 45,
    )
    out = base_bytes + tail
    if len(out) != RECORD_BYTES:
        raise ValueError(f"v5 record is {len(out)} bytes, expected "
                         f"{RECORD_BYTES}")
    return out


def unpack_record(payload: bytes) -> dict:
    from unarchitectured_metal_base_data import RECORD as V3_STRUCT
    base = V3_STRUCT.unpack(payload[: V3_STRUCT.size])
    tail = TAIL_V5.unpack(payload[V3_STRUCT.size:])
    return {
        "bb": tuple(base[0:12]),
        "move": base[12], "promotion": base[13], "wdl": base[14],
        "rating": base[15], "castling": base[16], "ep_file": base[17],
        "halfmove": base[18], "time_class": base[19], "flags": base[20],
        "history_len": base[21], "history": tuple(base[22:30]),
        "game_hash": base[30], "player_hash": base[31],
        "teacher_score": base[32], "best_move": base[33],
        "best_score": base[34], "move_score": base[35], "ply": base[36],
        "legal_count": tail[0], "target_action": tail[1],
        "teacher_best_action": tail[2], "policy_kind": tail[3],
        "legal_flags": tail[4],
        "legal_actions": tuple(tail[5:5 + MAX_LEGAL_ACTIONS]),
        "legal_regrets": tuple(tail[5 + MAX_LEGAL_ACTIONS:
                                    5 + 2 * MAX_LEGAL_ACTIONS]),
        "elo_oppo": tail[5 + 2 * MAX_LEGAL_ACTIONS],
        "pretrain_quality": tail[6 + 2 * MAX_LEGAL_ACTIONS],
    }


# ----------------------------------------------------------------------
# game -> rows
# ----------------------------------------------------------------------

def parse_elo(header_value) -> int | None:
    try:
        v = int(str(header_value))
    except (TypeError, ValueError):
        return None
    return v if 0 <= v <= 65535 else None


def side_quality_from_tags(tags, white: bool,
                           default_engine: str) -> int:
    """Quality for one side from a game's header tags. Header quality
    wins when present and valid; else derive from the engine header;
    else default_engine. Used by BOTH the pass-1 text scanner and the
    pass-2 replay so the split/filter decisions always agree."""
    engine = str(tags.get(
        "WhiteEngine" if white else "BlackEngine",
        "")).strip().lower() or default_engine
    q = str(tags.get(
        "WhiteEloQuality" if white else "BlackEloQuality",
        "")).strip().lower()
    return QUALITY_NAMES.index(q) if q in QUALITY_NAMES \
        else ENGINE_QUALITY.get(engine, QUALITY_HUMAN)


def game_kept(tags, keep, default_engine: str) -> bool:
    """True when the game's headers give it a valid dual-elo pair and
    (when a quality filter is active) both sides pass it — all-rows
    semantics: quality is per side and constant across that side's
    plies, so both-sides is equivalent to all-rows."""
    if parse_elo(tags.get("WhiteElo")) is None or \
            parse_elo(tags.get("BlackElo")) is None:
        return False
    if keep is None:
        return True
    return (side_quality_from_tags(tags, True, default_engine) in keep
            and side_quality_from_tags(tags, False, default_engine)
            in keep)


def game_rows(source_name: str, game_ordinal: int,
              g: "chess.pgn.Game", default_engine: str = "human") -> list[dict]:
    """Replay one PGN game into per-move pretrain rows.

    default_engine applies when a side has no WhiteEngine/BlackEngine
    header (e.g. the committed data/selfplay PGN is maia3 but predates
    the engine headers; real-human corpora keep the default "human").
    """
    tags = g.headers
    w = parse_elo(tags.get("WhiteElo"))
    b = parse_elo(tags.get("BlackElo"))
    if w is None or b is None:
        return None  # caller counts as skipped
    we = str(tags.get("WhiteEngine", "")).strip().lower() \
        or default_engine
    be = str(tags.get("BlackEngine", "")).strip().lower() \
        or default_engine
    wq = side_quality_from_tags(tags, True, default_engine)
    bq = side_quality_from_tags(tags, False, default_engine)
    result = tags.get("Result", "*")
    # Content-based, not filename-based: source_name/Round/ordinal meant two
    # copies of the identical game under different filenames (or a
    # different Round tag) got different identities, which is exactly what
    # let a duplicate game cross the train/validation split undetected --
    # actual cross-split leakage prevention now happens earlier via
    # scan_file's dedup pass, but this per-row field should still describe
    # the game's real content for anyone using it downstream, not an
    # artifact of which file it happened to be read from.
    move_seq = " ".join(m.uci() for m in g.mainline_moves())
    game_hash = int.from_bytes(hashlib.sha256(move_seq.encode()).digest()[:8], "little")

    board = chess.Board()
    moves = []
    rows = []
    for mv in g.mainline_moves():
        if not mv in board.legal_moves:
            break  # truncated/illegal tail: stop the game cleanly
        white_turn = board.turn == chess.WHITE
        rating = w if white_turn else b
        oppo = b if white_turn else w
        quality = wq if white_turn else bq
        # STM-normalized encoding: v5_action() mirrors original-view
        # moves into the side-to-move view; bitboards_stm() mirrors the
        # board. Both use the same convention as unchessed-datagen.
        bb = bitboards_stm(board)
        legal = [v5_action(m) for m in board.legal_moves]
        target = v5_action(mv)
        # hard guard: the played move must encode into its own legal set
        if target not in legal:
            raise ValueError(
                f"{source_name} game {game_ordinal} ply {board.ply()}: "
                f"target {target:#x} absent from legal set — encoding bug")
        # ep FILE index 0-7 (0xFF when absent) — the v4 embedding
        # domain, not a square
        ep_file = (chess.square_file(board.ep_square)
                   if board.ep_square is not None else 0xFF)
        halfmove = min(board.halfmove_clock, 255)
        flags = 0
        if mv.promotion is not None:
            flags |= 4  # FLAG_PROMOTION
        if board.is_castling(mv):
            flags |= 1  # FLAG_CASTLE
        if board.is_en_passant(mv):
            flags |= 2  # FLAG_EN_PASSANT
        if moves:
            flags |= 16  # FLAG_HISTORY
        # history: previous up-to-8 plies, same encoding as the target
        hist = [v5_action(prev) for prev in moves[-8:]]
        hist_len = len(hist)
        # game-outcome proxy wdl from the mover's perspective
        if result == "1-0":
            wdl = 2 if white_turn else 0
        elif result == "0-1":
            wdl = 0 if white_turn else 2
        else:
            wdl = 1
        player_hash = game_hash ^ (0x5A5A5A5A5A5A5A5A if white_turn
                                   else 0xA5A5A5A5A5A5A5A5)
        rows.append({
            "bb": bb, "target": target, "legal": legal, "wdl": wdl,
            "rating": rating, "elo_oppo": oppo, "quality": quality,
            "castling": castling_byte(board), "ep_file": ep_file,
            "halfmove": halfmove, "time_class": 0, "flags": flags,
            "history": tuple(hist), "history_len": hist_len,
            "game_hash": game_hash, "player_hash": player_hash,
            "ply": board.ply() + 1,
            "policy_kind": ENGINE_POLICY_KIND.get(
                we if white_turn else be, POLICY_HUMAN),
        })
        moves.append(mv)
        board.push(mv)
    return rows


# ----------------------------------------------------------------------
# build / validate commands
# ----------------------------------------------------------------------

_TAG_RE = None


def _tag_re():
    global _TAG_RE
    if _TAG_RE is None:
        import re
        _TAG_RE = re.compile(r'^\s*\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
    return _TAG_RE


def scan_file(path: Path, keep, default_engine: str, seen_hashes: set) -> tuple:
    """Text-only pass-1 scan: count games with a valid dual-elo header
    pair (and passing the quality filter when active). No chess
    parsing — fast enough to run over tens of GB, and uses the exact
    same header logic (game_kept) as the pass-2 replay so the two
    passes always agree.

    Also flags duplicate games by content: a game whose raw movetext
    (not the source filename/Round tag, which was the actual bug --
    identical games copied into differently-named PGNs used to get
    different identities and could land on opposite sides of the
    train/validation split) was already seen -- in this file or an
    earlier one -- is excluded from the returned count and its
    raw-kept-ordinal (0-indexed among every game_kept-passing game in
    THIS file, duplicates included) is returned so pass 2 can skip the
    exact same occurrence. `seen_hashes` is shared and mutated across
    the whole corpus; callers must scan files in a fixed order (first
    occurrence wins) for this to be deterministic.
    """
    n_kept = 0
    raw_kept_idx = 0
    dup_ordinals = set()
    tags: dict = {}
    movetext_lines: list = []
    in_game = False
    in_movetext = False
    tag_re = _tag_re()

    def finalize():
        nonlocal n_kept, raw_kept_idx
        if in_game and game_kept(tags, keep, default_engine):
            content_hash = hashlib.sha256(
                "".join(movetext_lines).strip().encode()
            ).digest()
            if content_hash in seen_hashes:
                dup_ordinals.add(raw_kept_idx)
            else:
                seen_hashes.add(content_hash)
                n_kept += 1
            raw_kept_idx += 1

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = tag_re.match(line)
            if m:
                if in_movetext:
                    finalize()  # a new tag block finalizes the previous game
                    in_game = True
                    in_movetext = False
                    tags = {}
                    movetext_lines = []
                tags[m.group(1)] = m.group(2)
                in_game = True
            elif line.strip():
                in_movetext = True
                movetext_lines.append(line)
    finalize()
    return n_kept, frozenset(dup_ordinals)


def _iter_pgn_games(path: Path):
    stream = path.open("r", encoding="utf-8")
    try:
        ordinal = 0
        while True:
            g = chess.pgn.read_game(stream)
            if g is None:
                break
            ordinal += 1
            yield ordinal, g
    finally:
        stream.close()


class _SideWriter:
    """Streams records into per-file shard files for one side
    (train/val). The final shard name embeds the source-file index so
    shards never collide across files and the output is deterministic
    regardless of worker count. The header is patched in at close
    time; the manifest sha256 is the FULL-FILE hash (header +
    records) so `validate --dir` accepts the set it just built."""

    def __init__(self, side_dir: Path, file_idx: int, shard_records: int):
        self.side_dir = side_dir
        self.file_idx = file_idx
        self.shard_records = shard_records
        self.shards = []  # (name, count, sha256)
        self.total = 0
        self._fh = None
        self._path = None
        self._count = 0

    def _open_next(self):
        seq = len(self.shards)
        self._name = f"shard-{self.file_idx:03d}-{seq:03d}.v5"
        self._path = self.side_dir / self._name
        self._fh = self._path.open("wb")
        self._fh.write(b"\x00" * HEADER_BYTES)  # placeholder
        self._count = 0

    def write(self, record: bytes):
        if self._fh is None:
            self._open_next()
        self._fh.write(record)
        self._count += 1
        if self._count >= self.shard_records:
            self._close_current()

    def _close_current(self):
        assert self._fh is not None
        self._fh.seek(0)
        self._fh.write(make_header(self._count))
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        h = hashlib.sha256()
        with self._path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.shards.append((self._name, self._count, h.hexdigest()))
        self.total += self._count
        self._fh = None

    def close(self):
        if self._fh is not None:
            self._close_current()


def _build_file(job: tuple) -> dict:
    (file_idx, path, global_start, cutoff, keep, out_dir,
     default_engine, shard_records, expected_kept, dup_ordinals) = job
    import time

    t0 = time.monotonic()
    train = _SideWriter(Path(out_dir) / "train", file_idx, shard_records)
    val = _SideWriter(Path(out_dir) / "val", file_idx, shard_records)
    kept_idx = 0  # unique games only -- matches pass 1's now-deduplicated count
    raw_kept_idx = 0  # every game_kept-passing game, duplicates included --
    # this is the indexing space scan_file's dup_ordinals was computed in
    rows = 0
    dist = defaultdict(lambda: defaultdict(int))
    for _ordinal, g in _iter_pgn_games(Path(path)):
        if not game_kept(g.headers, keep, default_engine):
            continue
        if raw_kept_idx in dup_ordinals:
            # same content already written once (this file or an earlier
            # one): drop the repeat rather than let it inflate one split or
            # cross into the other.
            raw_kept_idx += 1
            continue
        raw_kept_idx += 1
        writer = val if (global_start + kept_idx) >= cutoff else train
        recs = game_rows(Path(path).name, kept_idx, g, default_engine)
        if recs:
            for row in recs:
                writer.write(make_v5_bytes(row))
                dist[row["quality"]][row["rating"] // 100 * 100] += 1
                rows += 1
        kept_idx += 1
    train.close()
    val.close()
    if kept_idx != expected_kept:
        raise RuntimeError(
            f"file {file_idx} ({path}): replay kept {kept_idx} games, "
            f"pass-1 scan counted {expected_kept} — the two passes "
            f"disagree (text scan vs parse); refusing to build")
    return {
        "file_idx": file_idx, "path": str(path),
        "kept_games": kept_idx, "rows": rows,
        "train_shards": train.shards, "val_shards": val.shards,
        "train_total": train.total, "val_total": val.total,
        "dist": {q: dict(bands) for q, bands in dist.items()},
        "elapsed_s": round(time.monotonic() - t0, 1),
    }


def cmd_build(args: argparse.Namespace) -> int:
    import multiprocessing as mp
    import time

    out = Path(args.out)
    for sub in ("train", "val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # deterministic regeneration: clear any prior shards
    for sub in ("train", "val"):
        for stale in (out / sub).glob("shard-*.v5"):
            stale.unlink()

    keep = None
    if args.quality_filter:
        keep_names = {q.strip() for q in args.quality_filter.split(",")}
        unknown = keep_names - set(QUALITY_NAMES)
        if unknown:
            raise SystemExit(f"unknown quality name(s) {sorted(unknown)}; "
                             f"choose from {list(QUALITY_NAMES)}")
        keep = {QUALITY_NAMES.index(q) for q in keep_names}

    files = [Path(p) for p in args.pgn]
    workers = args.workers or max(1, min((os.cpu_count() or 4) - 2, 128))

    print(f"pass 1: text scan of {len(files)} file(s) for the "
          f"game-disjoint split (also flagging duplicate games by "
          f"content, not filename)...", flush=True)
    t0 = time.monotonic()
    seen_hashes: set = set()
    # Sequential and in a fixed (input) order is load-bearing here: dedup is
    # first-occurrence-wins, so scanning file i must see everything file
    # i-1 already claimed before deciding what's new in file i.
    scan_results = [
        scan_file(p, keep, args.default_engine, seen_hashes) for p in files
    ]
    per_file_kept = [n for n, _dups in scan_results]
    per_file_dups = [dups for _n, dups in scan_results]
    n_dropped_dups = sum(len(d) for d in per_file_dups)
    n_kept_total = sum(per_file_kept)
    print(f"pass 1 done in {time.monotonic() - t0:.1f}s: "
          f"{n_kept_total:,} kept games"
          + (f" ({n_dropped_dups:,} duplicate games dropped)"
             if n_dropped_dups else ""),
          flush=True)
    if n_kept_total == 0:
        raise SystemExit("no usable games (valid dual-elo headers"
                         + (" + quality filter" if keep else "") + ")")

    val_n = min(args.val_games, n_kept_total // 2)
    if val_n <= 0:
        raise SystemExit("not enough games for a validation split")
    # last val_n kept games (input order) -> validation; deterministic
    cutoff = n_kept_total - val_n

    # per-file global kept-index ranges (input order)
    jobs = []
    start = 0
    for i, p in enumerate(files):
        jobs.append((i, str(p), start, cutoff, keep, str(out),
                     args.default_engine, args.shard_records,
                     per_file_kept[i], per_file_dups[i]))
        start += per_file_kept[i]
    # skip files with nothing to do
    jobs = [j for j in jobs if j[8] > 0]

    print(f"pass 2: replaying {len(jobs)} file(s) with {workers} "
          f"worker(s); cutoff after game {cutoff} "
          f"({val_n} val games)...", flush=True)
    t0 = time.monotonic()
    reports = []
    if workers == 1 or len(jobs) == 1:
        for job in jobs:
            rep = _build_file(job)
            reports.append(rep)
            print(f"  file {rep['file_idx']:03d}: {rep['kept_games']:,} "
                  f"games, {rep['rows']:,} rows "
                  f"({rep['train_total']:,} train / "
                  f"{rep['val_total']:,} val) in {rep['elapsed_s']}s",
                  flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for rep in pool.imap_unordered(_build_file, jobs):
                reports.append(rep)
                print(f"  file {rep['file_idx']:03d}: "
                      f"{rep['kept_games']:,} games, {rep['rows']:,} rows "
                      f"({rep['train_total']:,} train / "
                      f"{rep['val_total']:,} val) in {rep['elapsed_s']}s",
                      flush=True)
    reports.sort(key=lambda r: r["file_idx"])
    print(f"pass 2 done in {time.monotonic() - t0:.1f}s", flush=True)

    train_count = sum(r["train_total"] for r in reports)
    val_count = sum(r["val_total"] for r in reports)
    dist = defaultdict(lambda: defaultdict(int))
    for r in reports:
        for q, bands in r["dist"].items():
            for band, v in bands.items():
                dist[q][band] += v
    def _sha256_streamed(path: Path, block_bytes: int = 1 << 20) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(block_bytes), b""):
                digest.update(chunk)
        return digest.hexdigest()

    report = {
        "tool": "tools/pretrain_v5_data.py",
        "magic": MAGIC.decode("ascii"),
        # A path string alone can't independently prove which bytes actually
        # produced this dataset -- reusing a path with changed content (or
        # relying on a filename alone to identify a source) left no way to
        # detect it. Streamed, not read into memory at once, since these can
        # be tens-of-GB PGN files.
        "sources": [
            {"path": str(p), "sha256": _sha256_streamed(p)} for p in files
        ],
        "games": {
            "kept_total": n_kept_total,
            "train": n_kept_total - val_n,
            "val": val_n,
        },
        "rows": {"train": train_count, "val": val_count},
        "split": ("game-disjoint: last N kept games (input order) -> "
                  "validation; per-file shards named "
                  "shard-<file>-<seq>.v5"),
        "quality_histogram": {QUALITY_NAMES[k]: {
            f"{band}-{band+99}": v for band, v in sorted(bands.items())
        } for k, bands in sorted(dist.items())},
        "workers": workers,
    }
    manifest = {
        **report,
        "shards": [
            {"side": "train", "path": str(out / "train" / name),
             "records": c, "sha256": h}
            for r in reports for name, c, h in r["train_shards"]
        ] + [
            {"side": "val", "path": str(out / "val" / name),
             "records": c, "sha256": h}
            for r in reports for name, c, h in r["val_shards"]
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:2000])
    print(f"wrote {train_count:,} train + {val_count:,} val records "
          f"({len(manifest['shards'])} shards)")
    return 0


def make_v5_bytes(row: dict) -> bytes:
    legal_padded = list(row["legal"][:MAX_LEGAL_ACTIONS]) + \
        [ACTION_SENTINEL] * (MAX_LEGAL_ACTIONS - len(row["legal"][:MAX_LEGAL_ACTIONS]))
    if len(row["legal"]) > MAX_LEGAL_ACTIONS:
        raise ValueError("position has more than 218 legal moves")
    hist = list(row["history"]) + [0] * (8 - len(row["history"]))
    return pack_record(
        bb=row["bb"], move=row["target"],
        promotion=1 if row["target"] & (1 << 14) else 0,
        wdl=row["wdl"], rating=row["rating"], castling=row["castling"],
        ep_file=row["ep_file"], halfmove=row["halfmove"],
        time_class=row["time_class"], flags=row["flags"],
        history_len=row["history_len"], history=tuple(hist),
        game_hash=row["game_hash"], player_hash=row["player_hash"],
        teacher_score=0, best_move=0, best_score=0, move_score=0,
        ply=min(row["ply"], 65535), remaining_ms=0, increment_ms=0,
        base_reserved=0, legal_count=len(row["legal"]),
        target_action=row["target"], teacher_best_action=ACTION_SENTINEL,
        policy_kind=row["policy_kind"], legal_flags=0,
        legal_actions=tuple(legal_padded), legal_regrets=(0,) * MAX_LEGAL_ACTIONS,
        elo_oppo=row["elo_oppo"], pretrain_quality=row["quality"],
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_validate(args: argparse.Namespace) -> int:
    import numpy as np

    out = Path(args.dir)
    manifest = json.loads((out / "manifest.json").read_text())
    errors = []
    total = 0
    for shard in manifest["shards"]:
        path = Path(shard["path"])
        if not path.exists():
            errors.append(f"missing shard {path}")
            continue
        with path.open("rb") as handle:
            count = parse_header(handle.read(HEADER_BYTES))
        size = path.stat().st_size
        if count != shard["records"]:
            errors.append(f"{path}: manifest says {shard['records']}, "
                          f"header says {count}")
        if size != HEADER_BYTES + count * RECORD_BYTES:
            errors.append(f"{path}: physical size {size} != header + "
                          f"{count}*{RECORD_BYTES}")
        if sha256_file(path) != shard["sha256"]:
            errors.append(f"{path}: sha256 mismatch")
        # structural spot-checks over a deterministic sample
        rng = np.random.default_rng(20260828)
        n = min(args.sample, count)
        mmapped = np.memmap(str(path), dtype=np.uint8, mode="r",
                            offset=HEADER_BYTES,
                            shape=(count, RECORD_BYTES))
        for i in np.sort(rng.integers(0, count, n, endpoint=False)):
            rec = unpack_record(bytes(mmapped[int(i)]))
            total += 1
            legal = list(rec["legal_actions"][:rec["legal_count"]])
            if rec["target_action"] not in legal:
                errors.append(f"{path} record {i}: target not in legal "
                              f"set (encoding bug)")
            if not (0 <= rec["rating"] <= 65535):
                errors.append(f"{path} record {i}: bad rating")
            if rec["ep_file"] not in (0xFF,) and not (
                    0 <= rec["ep_file"] <= 7):
                errors.append(f"{path} record {i}: ep_file {rec['ep_file']}"
                              f" outside 0-7/0xFF")
            if not (0 <= rec["castling"] <= 15):
                errors.append(f"{path} record {i}: bad castling byte")
            if not (0 <= rec["wdl"] <= 2):
                errors.append(f"{path} record {i}: bad wdl")
            if rec["elo_oppo"] > 65535:
                errors.append(f"{path} record {i}: bad elo_oppo")
            if rec["pretrain_quality"] > 3:
                errors.append(f"{path} record {i}: bad quality")
            if rec["teacher_best_action"] != ACTION_SENTINEL:
                errors.append(f"{path} record {i}: pretrain row must have "
                              f"sentinel teacher")
            if rec["policy_kind"] not in (0, 1):
                errors.append(f"{path} record {i}: bad policy_kind")
    report = {"records_spot_checked": total, "errors": errors[:50],
              "error_count": len(errors)}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:2000])
    return 0 if not errors else 1


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="PGN -> v5 shards (CPU stage)")
    b.add_argument("--pgn", action="append", required=True,
                   help="PGN file(s); repeatable/globbed")
    b.add_argument("--out", required=True)
    b.add_argument("--val-games", type=int, default=20000,
                   help="number of games held out for validation "
                        "(game-disjoint; default 20000)")
    b.add_argument("--shard-records", type=int, default=1_000_000,
                   help="records per shard file (default 1000000)")
    b.add_argument("--workers", type=int, default=None,
                   help="pass-2 replay workers, one per PGN file "
                        "(default min(cpus-2, 128))")
    b.add_argument("--quality-filter", default=None,
                   help="comma list of qualities to keep "
                        "(calibrated,native,approximate,human); used to "
                        "build the stage-2 trusted-only subset")
    b.add_argument("--default-engine", default="human",
                   choices=("human", "maia3", "stockfish", "lc0",
                            "rubichess"),
                   help="engine/quality for sides without an engine "
                        "header (default human; use maia3 for the "
                        "committed data/selfplay PGN)")
    v = sub.add_parser("validate", help="re-validate a built directory")
    v.add_argument("--dir", required=True)
    v.add_argument("--sample", type=int, default=2000,
                   help="structural spot-checks per shard")
    b.set_defaults(fn=cmd_build)
    v.set_defaults(fn=cmd_validate)
    return p


if __name__ == "__main__":
    ns = argument_parser().parse_args()
    sys.exit(ns.fn(ns))
