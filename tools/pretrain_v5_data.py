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
      --out /data/pretrain-v5 --val-games 20000 --shard-records 250000
  python3 tools/pretrain_v5_data.py validate --dir /data/pretrain-v5

Sizes: ~1088 bytes per record before zip; 13k rows ~ 14 MB, 327M rows
~ 356 GB raw (the A100 box streams via mmap; on the CPU box consider
--shard-records tuning if the volume is tight).

Dependencies: python-chess (tools/requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
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
# unarchitectured_v1_base_data: 12 bitboards, move, promotion, wdl,
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
    from unarchitectured_v1_base_data import RECORD as V3_STRUCT
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
    from unarchitectured_v1_base_data import RECORD as V3_STRUCT
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


def game_rows(source_name: str, game_ordinal: int,
              g: "chess.pgn.Game", default_engine: str = "human") -> list[dict]:
    """Replay one PGN game into per-move pretrain rows.

    default_engine applies when a side has no WhiteEngine/BlackEngine
    header (e.g. the committed data/selfplay PGN is maia3 but predates
    the engine headers; real-human corpora keep the default "human").
    """
    w = parse_elo(g.headers.get("WhiteElo"))
    b = parse_elo(g.headers.get("BlackElo"))
    if w is None or b is None:
        return None  # caller counts as skipped
    we = str(g.headers.get("WhiteEngine", "")).strip().lower()         or default_engine
    be = str(g.headers.get("BlackEngine", "")).strip().lower()         or default_engine
    wq = str(g.headers.get("WhiteEloQuality", "")).strip().lower()
    bq = str(g.headers.get("BlackEloQuality", "")).strip().lower()
    # header quality wins when present and valid; else derive from engine
    wq = QUALITY_NAMES.index(wq) if wq in QUALITY_NAMES \
        else ENGINE_QUALITY.get(we, QUALITY_HUMAN)
    bq = QUALITY_NAMES.index(bq) if bq in QUALITY_NAMES \
        else ENGINE_QUALITY.get(be, QUALITY_HUMAN)
    result = g.headers.get("Result", "*")
    game_key = f"{source_name}:{int(g.headers.get('Round', 0)) if str(g.headers.get('Round', '')).isdigit() else game_ordinal}"
    game_hash = int.from_bytes(hashlib.sha256(game_key.encode()).digest()[:8], "little")

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

def iter_pgn_games(path: Path):
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


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    for sub in ("train", "val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # pass 1: collect per-game row counts for the game-disjoint split
    games = []  # (source, ordinal, header_elo_present, n_rows)
    skipped_no_elo = 0
    for pg in args.pgn:
        path = Path(pg)
        for ordinal, g in iter_pgn_games(path):
            if parse_elo(g.headers.get("WhiteElo")) is None or \
                    parse_elo(g.headers.get("BlackElo")) is None:
                skipped_no_elo += 1
                continue
            rows = game_rows(path.name, ordinal, g, args.default_engine)
            if rows is None:
                skipped_no_elo += 1
                continue
            games.append((str(path), ordinal, rows))
    total_rows = sum(len(r) for _, _, r in games)
    print(f"games with dual-elo headers: {len(games)} "
          f"({total_rows:,} rows); skipped {skipped_no_elo} "
          f"games without elos", flush=True)
    if not games:
        raise SystemExit("no usable games")

    if args.quality_filter:
        keep_names = {q.strip() for q in args.quality_filter.split(",")}
        unknown = keep_names - set(QUALITY_NAMES)
        if unknown:
            raise SystemExit(f"unknown quality name(s) {sorted(unknown)}; "
                             f"choose from {list(QUALITY_NAMES)}")
        keep = {QUALITY_NAMES.index(q) for q in keep_names}
        before = len(games)
        # ALL-rows semantics: a game with even one approximate row is
        # excluded from the trusted subset (stage 2 is trusted-only)
        games = [g for g in games if all(
            row["quality"] in keep for row in g[2])]
        total_rows = sum(len(r) for _, _, r in games)
        print(f"quality filter {sorted(keep_names)}: {before} -> "
              f"{len(games)} games ({total_rows:,} rows)", flush=True)
        if not games:
            raise SystemExit("quality filter removed all games")
    val_n = min(args.val_games, len(games) // 2)
    if val_n <= 0:
        raise SystemExit("not enough games for a validation split")
    # last val_n games (in input order) -> validation; deterministic
    val_games = games[len(games) - val_n:]
    train_games = games[:len(games) - val_n]
    train_ids = {id(r) for _, _, r in train_games}
    val_ids = {id(r) for _, _, r in val_games}
    assert train_ids.isdisjoint(val_ids)

    def write_side(name, side_games):
        rows_iter = (row for _, _, rows in side_games for row in rows)
        paths = []
        total = 0
        shard_index = 0
        while True:
            count = 0
            temporary = None
            with tempfile.NamedTemporaryFile(dir=out / name,
                                             prefix="shard-",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(make_header(0))
                for row in rows_iter:
                    handle.write(make_v5_bytes(row))
                    count += 1
                    if count >= args.shard_records:
                        break
                if count:
                    handle.seek(0)
                    handle.write(make_header(count))
                    handle.flush()
                    os.fsync(handle.fileno())
            if count == 0:
                temporary.unlink()
                return total, paths
            path = out / name / f"shard-{shard_index:05d}.v5"
            os.replace(temporary, path)
            total += count
            paths.append((count, path))
            shard_index += 1
            if count < args.shard_records:
                return total, paths

    train_count, train_paths = write_side("train", train_games)
    val_count, val_paths = write_side("val", val_games)

    # distribution report
    dist = defaultdict(lambda: defaultdict(int))
    for _, _, rows in games:
        for row in rows:
            dist[row["quality"]][row["rating"] // 100 * 100] += 1
    report = {
        "tool": "tools/pretrain_v5_data.py",
        "magic": MAGIC.decode("ascii"),
        "sources": [str(p) for p in args.pgn],
        "games": {"train": len(train_games), "val": len(val_games)},
        "rows": {"train": train_count, "val": val_count},
        "split": "game-disjoint: last N games (input order) -> validation",
        "quality_histogram": {QUALITY_NAMES[k]: {
            f"{band}-{band+99}": v for band, v in sorted(bands.items())
        } for k, bands in sorted(dist.items())},
        "skipped_games_without_elos": skipped_no_elo,
    }
    manifest = {
        **report,
        "shards": [
            {"side": "train", "path": str(p), "records": c,
             "sha256": sha256_file(p)} for c, p in train_paths
        ] + [
            {"side": "val", "path": str(p), "records": c,
             "sha256": sha256_file(p)} for c, p in val_paths
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:2000])
    print(f"wrote {train_count:,} train + {val_count:,} val records")
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
    b.add_argument("--shard-records", type=int, default=500_000,
                   help="records per shard file (default 500000)")
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
