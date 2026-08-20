#!/usr/bin/env python3
"""Hydra Aegis v4 legal-set data ABI and leakage validator.

V4 extends each validated v3 semantic record with all legal promotion-aware
actions and an optional common-budget regret for every action.  This makes the
policy loss and runtime head genuinely legal-only: no dense 4,096/20,480-class
surrogate and no collapsed underpromotions.

The module remains NumPy/PyTorch-free at import. `numpy_dtype()` imports NumPy
only when an A100 trainer asks for a memory-map layout.
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

try:
    from aegis_v3_data import (
        AegisV3Record,
        FLAG_TEACHER,
        FormatError,
        RECORD as V3_RECORD,
    )
except ModuleNotFoundError:  # imported as tools.aegis_v4_data in unit tests
    from tools.aegis_v3_data import (
        AegisV3Record,
        FLAG_TEACHER,
        FormatError,
        RECORD as V3_RECORD,
    )

MAGIC = b"UNCHD4R0"
VERSION = 4
HEADER_BYTES = 64
RECORD_BYTES = 1088
ENDIAN_MARKER = 0x01020304
MANDATORY_FLAGS = 0x00FF
MAX_LEGAL_ACTIONS = 218
ACTION_VOCABULARY = 64 * 64 * 5
ACTION_SENTINEL = 0xFFFF
REGRET_SENTINEL = 0x7FFF
POLICY_HUMAN = 0
POLICY_GUIDE = 1
LEGAL_FLAG_REGRETS = 1 << 0

SCHEMA_DESCRIPTOR = """Unchessed Hydra Aegis data record v4;little-endian;header=magic:8,version:u16,header_bytes:u16,record_bytes:u16,flags:u16,endian:u32,records:u64,schema_sha256:32,crc32:u32;record=v3_semantics:160,legal_count:u16,target_action:u16,teacher_best_action:u16,policy_kind:u8,legal_flags:u8,legal_actions:218xu16,legal_regrets:218xi16,reserved:48"""
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_DESCRIPTOR.encode("ascii")).digest()
HEADER_PREFIX = struct.Struct("<8sHHHHIQ32s")
HEADER = struct.Struct("<8sHHHHIQ32sI")
TAIL = struct.Struct("<HHHBB218H218h48s")
assert HEADER.size == HEADER_BYTES
assert V3_RECORD.size + TAIL.size == RECORD_BYTES


def encode_action(move: int, promotion: int) -> int:
    """Encode normalized 12-bit from/to plus 0/N/B/R/Q promotion class."""
    if not 0 <= move < 4096 or not 0 <= promotion <= 4:
        raise FormatError("cannot encode action outside move/promotion vocabulary")
    return move | (promotion << 12)


def mirror_action(action: int) -> int:
    if not 0 <= action < ACTION_VOCABULARY:
        raise FormatError("action outside v4 vocabulary")
    source = (action & 63) ^ 7
    target = ((action >> 6) & 63) ^ 7
    return source | (target << 6) | (action & 0xF000)


@dataclasses.dataclass(frozen=True)
class AegisV4Record:
    base: AegisV3Record
    legal_count: int
    target_action: int
    teacher_best_action: int
    policy_kind: int
    legal_flags: int
    legal_actions: tuple[int, ...]
    legal_regrets: tuple[int, ...]
    reserved: bytes = bytes(48)

    def validate(self) -> None:
        self.base.validate()
        if not 1 <= self.legal_count <= MAX_LEGAL_ACTIONS:
            raise FormatError("legal_count must be in 1..218")
        if len(self.legal_actions) != MAX_LEGAL_ACTIONS:
            raise FormatError("legal_actions must have exactly 218 slots")
        if len(self.legal_regrets) != MAX_LEGAL_ACTIONS:
            raise FormatError("legal_regrets must have exactly 218 slots")
        active = self.legal_actions[: self.legal_count]
        if any(not 0 <= action < ACTION_VOCABULARY for action in active):
            raise FormatError("active legal action outside 20,480-class vocabulary")
        if any(a >= b for a, b in zip(active, active[1:])):
            raise FormatError("active legal actions must be strictly increasing")
        if any(action != ACTION_SENTINEL for action in self.legal_actions[self.legal_count :]):
            raise FormatError("unused legal action slots must contain 0xffff")
        expected_target = encode_action(self.base.move, self.base.promotion)
        if self.target_action != expected_target or self.target_action not in active:
            raise FormatError("selected action is inconsistent with base move or absent from legal set")
        if self.policy_kind not in (POLICY_HUMAN, POLICY_GUIDE):
            raise FormatError("policy_kind must be human or guide")
        if self.legal_flags & ~LEGAL_FLAG_REGRETS:
            raise FormatError("unknown legal-set flag bits")
        has_regrets = bool(self.legal_flags & LEGAL_FLAG_REGRETS)
        has_teacher = bool(self.base.flags & FLAG_TEACHER)
        if has_regrets != has_teacher:
            raise FormatError("per-action regrets and base teacher flag must agree")
        active_regrets = self.legal_regrets[: self.legal_count]
        if has_regrets:
            if self.teacher_best_action not in active:
                raise FormatError("teacher best action is absent from legal set")
            if (self.teacher_best_action & 0x0FFF) != (self.base.best_move & 0x0FFF):
                raise FormatError("exact teacher action and base best move disagree")
            if any(not 0 <= regret < REGRET_SENTINEL for regret in active_regrets):
                raise FormatError("active regrets must be non-negative finite i16 values")
            best_index = active.index(self.teacher_best_action)
            if active_regrets[best_index] != 0 or min(active_regrets) != 0:
                raise FormatError("teacher best action must have zero regret")
        else:
            if self.teacher_best_action != ACTION_SENTINEL:
                raise FormatError("teacher action must be 0xffff without regret labels")
            if any(regret != REGRET_SENTINEL for regret in active_regrets):
                raise FormatError("unlabelled active regrets must use 0x7fff")
        if any(regret != REGRET_SENTINEL for regret in self.legal_regrets[self.legal_count :]):
            raise FormatError("unused regret slots must contain 0x7fff")
        if self.reserved != bytes(48):
            raise FormatError("reserved v4 tail bytes must be zero")

    def pack(self) -> bytes:
        self.validate()
        return self.base.pack() + TAIL.pack(
            self.legal_count,
            self.target_action,
            self.teacher_best_action,
            self.policy_kind,
            self.legal_flags,
            *self.legal_actions,
            *self.legal_regrets,
            self.reserved,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "AegisV4Record":
        if len(payload) != RECORD_BYTES:
            raise FormatError(f"record has {len(payload)} bytes, expected {RECORD_BYTES}")
        base = AegisV3Record.unpack(payload[: V3_RECORD.size])
        values = TAIL.unpack(payload[V3_RECORD.size :])
        record = cls(
            base=base,
            legal_count=values[0],
            target_action=values[1],
            teacher_best_action=values[2],
            policy_kind=values[3],
            legal_flags=values[4],
            legal_actions=tuple(values[5 : 5 + MAX_LEGAL_ACTIONS]),
            legal_regrets=tuple(values[5 + MAX_LEGAL_ACTIONS : 5 + 2 * MAX_LEGAL_ACTIONS]),
            reserved=values[-1],
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
    if (version, header_bytes, record_bytes) != (VERSION, HEADER_BYTES, RECORD_BYTES):
        raise FormatError("unsupported v4 version/header/record width")
    if flags != MANDATORY_FLAGS:
        raise FormatError(f"unsupported mandatory semantic flags 0x{flags:04x}")
    if endian != ENDIAN_MARKER:
        raise FormatError("wrong endian marker")
    if digest != SCHEMA_SHA256:
        raise FormatError("record schema SHA-256 mismatch")
    if crc != zlib.crc32(payload[: HEADER_PREFIX.size]) & 0xFFFFFFFF:
        raise FormatError("header CRC32 mismatch")
    return count


def write_shard(path: str | Path, records: Iterable[AegisV4Record]) -> int:
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


def iter_shard(path: str | Path) -> Iterator[AegisV4Record]:
    path = Path(path)
    count = shard_record_count(path)
    with path.open("rb") as handle:
        handle.seek(HEADER_BYTES)
        for index in range(count):
            try:
                yield AegisV4Record.unpack(handle.read(RECORD_BYTES))
            except FormatError as error:
                raise FormatError(f"{path}: record {index}: {error}") from error


def validate_shard(path: str | Path) -> dict:
    path = Path(path)
    games: set[int] = set()
    players: set[int] = set()
    records = labelled = legal_sum = legal_max = 0
    for record in iter_shard(path):
        records += 1
        games.add(record.base.game_hash)
        players.add(record.base.player_hash)
        labelled += bool(record.legal_flags & LEGAL_FLAG_REGRETS)
        legal_sum += record.legal_count
        legal_max = max(legal_max, record.legal_count)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "records": records,
        "games": len(games),
        "players": len(players),
        "regret_labelled": labelled,
        "mean_legal_actions": legal_sum / max(1, records),
        "maximum_legal_actions": legal_max,
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
                games.add(record.base.game_hash)
                players.add(record.base.player_hash)
        return records, games, players

    train_records, train_games, train_players = identities(train_paths)
    val_records, val_games, val_players = identities(validation_paths)
    game_overlap = train_games & val_games
    player_overlap = train_players & val_players
    report = {
        "train_records": train_records,
        "validation_records": val_records,
        "overlapping_games": len(game_overlap),
        "overlapping_players": len(player_overlap),
        "disjoint": not game_overlap and not player_overlap,
    }
    if not report["disjoint"]:
        raise FormatError(
            f"split leakage: {len(game_overlap)} game hashes and {len(player_overlap)} player hashes overlap"
        )
    return report


def numpy_dtype():
    import numpy as np

    dtype = np.dtype(
        [
            ("bb", "<u8", 12),
            ("move", "<u2"),
            ("promotion", "u1"),
            ("wdl", "u1"),
            ("rating", "<u2"),
            ("castling", "u1"),
            ("ep_file", "u1"),
            ("halfmove", "u1"),
            ("time_class", "u1"),
            ("flags", "u1"),
            ("history_len", "u1"),
            ("history", "<u2", 8),
            ("game_hash", "<u8"),
            ("player_hash", "<u8"),
            ("teacher_score", "<i2"),
            ("best_move", "<u2"),
            ("best_score", "<i2"),
            ("move_score", "<i2"),
            ("ply", "<u2"),
            ("remaining_ms", "<u4"),
            ("increment_ms", "<u4"),
            ("base_reserved", "<u2"),
            ("legal_count", "<u2"),
            ("target_action", "<u2"),
            ("teacher_best_action", "<u2"),
            ("policy_kind", "u1"),
            ("legal_flags", "u1"),
            ("legal_actions", "<u2", MAX_LEGAL_ACTIONS),
            ("legal_regrets", "<i2", MAX_LEGAL_ACTIONS),
            ("reserved", "V48"),
        ]
    )
    if dtype.itemsize != RECORD_BYTES:
        raise AssertionError(f"NumPy ABI is {dtype.itemsize}, expected {RECORD_BYTES}")
    return dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("shards", nargs="+")
    inspect_parser.add_argument("--json", type=Path)
    audit_parser = subparsers.add_parser("audit-split")
    audit_parser.add_argument("--train", nargs="+", required=True)
    audit_parser.add_argument("--validation", nargs="+", required=True)
    audit_parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result: object
    if args.command == "inspect":
        result = [validate_shard(path) for path in args.shards]
    else:
        result = audit_disjoint_splits(args.train, args.validation)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
