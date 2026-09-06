#!/usr/bin/env python3
"""Read, write, inspect, and leakage-audit Unarchitectured Metal base records.

The frozen wire format is version 3 for compatibility with already-generated
shards; that wire version is not an architecture/product version. The format
uses a 64-byte header, a canonical schema SHA-256, and 160-byte records with
promotion identity, WDL, recent moves, privacy-preserving game/player hashes,
time class, and optional common-budget teacher regret labels.

This module has no NumPy/PyTorch dependency, so data shards can be validated on
CPU ingestion hosts before an A100 job starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Iterable, Iterator, Sequence

MAGIC = b"UNCHD3R0"
VERSION = 3
HEADER_BYTES = 64
RECORD_BYTES = 160
ENDIAN_MARKER = 0x01020304
MANDATORY_FLAGS = 0x003F
HISTORY_PLIES = 8
UNKNOWN_EP = 0xFF
UNKNOWN_CLOCK = 0xFFFFFFFF

# The string, not Python implementation details, defines field order and width.
# Its predecessor-era label is frozen wire identity and must not be rewritten.
SCHEMA_DESCRIPTOR = """Unchessed Hydra Aegis data record v3;little-endian;header=magic:8,version:u16,header_bytes:u16,record_bytes:u16,flags:u16,endian:u32,records:u64,schema_sha256:32,crc32:u32;record=bitboards:12xu64,move:u16,promotion:u8,wdl:u8,rating:u16,castling:u8,ep_file:u8,halfmove:u8,time_class:u8,flags:u8,history_len:u8,history:8xu16,game_hash:u64,player_hash:u64,teacher_score:i16,best_move:u16,best_score:i16,move_score:i16,ply:u16,remaining_ms:u32,increment_ms:u32,reserved:u16"""
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_DESCRIPTOR.encode("ascii")).digest()

HEADER_PREFIX = struct.Struct("<8sHHHHIQ32s")
HEADER = struct.Struct("<8sHHHHIQ32sI")
RECORD = struct.Struct("<12QHBBHBBBBBB8HQQhHhhHIIH")
assert HEADER.size == HEADER_BYTES
assert RECORD.size == RECORD_BYTES

FLAG_CASTLE = 1 << 0
FLAG_EN_PASSANT = 1 << 1
FLAG_PROMOTION = 1 << 2
FLAG_TEACHER = 1 << 3
FLAG_HISTORY = 1 << 4
FLAG_CLOCK = 1 << 5

TIME_BULLET = 0
TIME_BLITZ = 1
TIME_RAPID = 2
TIME_CLASSICAL = 3
TIME_UNKNOWN = 4
TIME_CLASS_COUNT = 5


class FormatError(ValueError):
    """Raised when a shard violates the frozen v3 ABI."""


@dataclasses.dataclass(frozen=True)
class UnarchitecturedV1BaseRecord:
    bitboards: tuple[int, ...]
    move: int
    promotion: int
    wdl: int
    rating: int
    castling: int
    ep_file: int
    halfmove: int
    time_class: int
    flags: int
    history_len: int
    history: tuple[int, ...]
    game_hash: int
    player_hash: int
    teacher_score: int = 0
    best_move: int = 0
    best_score: int = 0
    move_score: int = 0
    ply: int = 0
    remaining_ms: int = UNKNOWN_CLOCK
    increment_ms: int = UNKNOWN_CLOCK
    reserved: int = 0

    def validate(self) -> None:
        if len(self.bitboards) != 12:
            raise FormatError("record must contain exactly 12 bitboards")
        if any(value < 0 or value > 0xFFFFFFFFFFFFFFFF for value in self.bitboards):
            raise FormatError("bitboard outside u64 range")
        if not 0 <= self.move < 4096:
            raise FormatError("move must contain a 6-bit source and destination")
        if not 0 <= self.promotion <= 4:
            raise FormatError("promotion must be 0 none, 1 knight, 2 bishop, 3 rook, or 4 queen")
        is_promotion = bool(self.flags & FLAG_PROMOTION)
        if is_promotion != (self.promotion != 0):
            raise FormatError("promotion flag and promotion identity disagree")
        if self.wdl not in (0, 1, 2):
            raise FormatError("WDL must be 0 loss, 1 draw, or 2 win")
        if not 0 <= self.rating <= 65535:
            raise FormatError("rating outside u16 range")
        if not 0 <= self.castling < 16:
            raise FormatError("castling rights use only four bits")
        if self.ep_file not in (*range(8), UNKNOWN_EP):
            raise FormatError("en-passant file must be 0..7 or 0xff")
        if not 0 <= self.halfmove <= 255:
            raise FormatError("halfmove clock outside u8 range")
        if not 0 <= self.time_class < TIME_CLASS_COUNT:
            raise FormatError("unknown time-class code")
        if self.flags & ~0x3F:
            raise FormatError("record has unknown flag bits")
        if len(self.history) != HISTORY_PLIES or not 0 <= self.history_len <= HISTORY_PLIES:
            raise FormatError("history must have eight slots and history_len <= 8")
        if any(not 0 <= move < 65536 for move in self.history):
            raise FormatError("history move outside u16 range")
        if any(self.history[self.history_len :]):
            raise FormatError("unused history slots must be zero")
        if bool(self.flags & FLAG_HISTORY) != (self.history_len != 0):
            raise FormatError("history flag and history_len disagree")
        if not self.game_hash or not self.player_hash:
            raise FormatError("game/player hashes must be non-zero pseudonyms")
        for name in ("teacher_score", "best_score", "move_score"):
            value = getattr(self, name)
            if not -32768 <= value <= 32767:
                raise FormatError(f"{name} outside i16 range")
        if not 0 <= self.best_move < 65536:
            raise FormatError("best_move outside u16 range")
        if not 0 <= self.ply <= 65535:
            raise FormatError("ply outside u16 range")
        if not 0 <= self.remaining_ms <= 0xFFFFFFFF or not 0 <= self.increment_ms <= 0xFFFFFFFF:
            raise FormatError("clock field outside u32 range")
        if self.reserved != 0:
            raise FormatError("reserved record field must be zero")
        has_teacher = bool(self.flags & FLAG_TEACHER)
        if not has_teacher and any((self.teacher_score, self.best_move, self.best_score, self.move_score)):
            raise FormatError("teacher fields require FLAG_TEACHER")
        has_clock = bool(self.flags & FLAG_CLOCK)
        clocks_known = self.remaining_ms != UNKNOWN_CLOCK or self.increment_ms != UNKNOWN_CLOCK
        if has_clock != clocks_known:
            raise FormatError("clock flag and clock fields disagree")

    @property
    def regret_cp(self) -> int | None:
        if not self.flags & FLAG_TEACHER:
            return None
        return max(0, self.best_score - self.move_score)

    def pack(self) -> bytes:
        self.validate()
        return RECORD.pack(
            *self.bitboards,
            self.move,
            self.promotion,
            self.wdl,
            self.rating,
            self.castling,
            self.ep_file,
            self.halfmove,
            self.time_class,
            self.flags,
            self.history_len,
            *self.history,
            self.game_hash,
            self.player_hash,
            self.teacher_score,
            self.best_move,
            self.best_score,
            self.move_score,
            self.ply,
            self.remaining_ms,
            self.increment_ms,
            self.reserved,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "UnarchitecturedV1BaseRecord":
        if len(payload) != RECORD_BYTES:
            raise FormatError(f"record has {len(payload)} bytes, expected {RECORD_BYTES}")
        values = RECORD.unpack(payload)
        record = cls(
            bitboards=tuple(values[0:12]),
            move=values[12],
            promotion=values[13],
            wdl=values[14],
            rating=values[15],
            castling=values[16],
            ep_file=values[17],
            halfmove=values[18],
            time_class=values[19],
            flags=values[20],
            history_len=values[21],
            history=tuple(values[22:30]),
            game_hash=values[30],
            player_hash=values[31],
            teacher_score=values[32],
            best_move=values[33],
            best_score=values[34],
            move_score=values[35],
            ply=values[36],
            remaining_ms=values[37],
            increment_ms=values[38],
            reserved=values[39],
        )
        record.validate()
        return record


def make_header(record_count: int) -> bytes:
    if not 0 <= record_count <= 0xFFFFFFFFFFFFFFFF:
        raise FormatError("record count outside u64 range")
    prefix = HEADER_PREFIX.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        RECORD_BYTES,
        MANDATORY_FLAGS,
        ENDIAN_MARKER,
        record_count,
        SCHEMA_SHA256,
    )
    return prefix + struct.pack("<I", zlib.crc32(prefix) & 0xFFFFFFFF)


def parse_header(payload: bytes) -> int:
    if len(payload) != HEADER_BYTES:
        raise FormatError(f"header has {len(payload)} bytes, expected {HEADER_BYTES}")
    magic, version, header_bytes, record_bytes, flags, endian, count, digest, crc = HEADER.unpack(payload)
    if magic != MAGIC:
        raise FormatError(f"bad data magic {magic!r}")
    if version != VERSION or header_bytes != HEADER_BYTES or record_bytes != RECORD_BYTES:
        raise FormatError("unsupported v3 version/header/record width")
    if flags != MANDATORY_FLAGS:
        raise FormatError(f"unsupported mandatory semantic flags 0x{flags:04x}")
    if endian != ENDIAN_MARKER:
        raise FormatError("wrong endian marker")
    if digest != SCHEMA_SHA256:
        raise FormatError("record schema SHA-256 mismatch")
    expected_crc = zlib.crc32(payload[: HEADER_PREFIX.size]) & 0xFFFFFFFF
    if crc != expected_crc:
        raise FormatError("header CRC32 mismatch")
    return count


def write_shard(path: str | Path, records: Iterable[UnarchitecturedV1BaseRecord]) -> int:
    """Atomically write a shard and return its record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(make_header(0))
            count = 0
            for record in records:
                handle.write(record.pack())
                count += 1
            handle.seek(0)
            handle.write(make_header(count))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return count
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def shard_record_count(path: str | Path) -> int:
    path = Path(path)
    size = path.stat().st_size
    if size < HEADER_BYTES or (size - HEADER_BYTES) % RECORD_BYTES:
        raise FormatError(f"{path}: file length is not header + N*{RECORD_BYTES}")
    with path.open("rb") as handle:
        count = parse_header(handle.read(HEADER_BYTES))
    physical = (size - HEADER_BYTES) // RECORD_BYTES
    if count != physical:
        raise FormatError(f"{path}: header says {count} records, file contains {physical}")
    return count


def iter_shard(path: str | Path) -> Iterator[UnarchitecturedV1BaseRecord]:
    path = Path(path)
    count = shard_record_count(path)
    with path.open("rb") as handle:
        handle.seek(HEADER_BYTES)
        for index in range(count):
            payload = handle.read(RECORD_BYTES)
            try:
                yield UnarchitecturedV1BaseRecord.unpack(payload)
            except FormatError as error:
                raise FormatError(f"{path}: record {index}: {error}") from error
        if handle.read(1):
            raise FormatError(f"{path}: trailing bytes after declared records")


def validate_shard(path: str | Path) -> dict:
    path = Path(path)
    game_hashes: set[int] = set()
    player_hashes: set[int] = set()
    promotions = teacher = history = clocks = 0
    count = 0
    for record in iter_shard(path):
        count += 1
        game_hashes.add(record.game_hash)
        player_hashes.add(record.player_hash)
        promotions += bool(record.flags & FLAG_PROMOTION)
        teacher += bool(record.flags & FLAG_TEACHER)
        history += bool(record.flags & FLAG_HISTORY)
        clocks += bool(record.flags & FLAG_CLOCK)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "records": count,
        "games": len(game_hashes),
        "players": len(player_hashes),
        "promotions": promotions,
        "teacher_labels": teacher,
        "history_records": history,
        "clock_records": clocks,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_sha256": SCHEMA_SHA256.hex(),
    }


def audit_disjoint_splits(train_paths: Sequence[str | Path], validation_paths: Sequence[str | Path]) -> dict:
    def identities(paths):
        games: set[int] = set()
        players: set[int] = set()
        records = 0
        for path in paths:
            for record in iter_shard(path):
                records += 1
                games.add(record.game_hash)
                players.add(record.player_hash)
        return records, games, players

    train_records, train_games, train_players = identities(train_paths)
    val_records, val_games, val_players = identities(validation_paths)
    game_overlap = train_games & val_games
    player_overlap = train_players & val_players
    report = {
        "train_records": train_records,
        "validation_records": val_records,
        "train_games": len(train_games),
        "validation_games": len(val_games),
        "train_players": len(train_players),
        "validation_players": len(val_players),
        "overlapping_games": len(game_overlap),
        "overlapping_players": len(player_overlap),
        "disjoint": not game_overlap and not player_overlap,
    }
    if not report["disjoint"]:
        raise FormatError(
            f"split leakage: {len(game_overlap)} game hashes and {len(player_overlap)} player hashes overlap"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="validate one or more v3 shards")
    inspect_parser.add_argument("shards", nargs="+")
    inspect_parser.add_argument("--json", type=Path)
    audit_parser = subparsers.add_parser("audit-split", help="require game/player-disjoint splits")
    audit_parser.add_argument("--train", nargs="+", required=True)
    audit_parser.add_argument("--validation", nargs="+", required=True)
    audit_parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        result: object = [validate_shard(path) for path in args.shards]
    else:
        result = audit_disjoint_splits(args.train, args.validation)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
