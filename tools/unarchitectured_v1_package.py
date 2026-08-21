#!/usr/bin/env python3
"""Binary UNARCHV1 tensor-package reader/writer shared by export tools."""

from __future__ import annotations

import dataclasses
import json
import os
import struct
import tempfile
import uuid
import zlib
from pathlib import Path

MAGIC = b"UNARCHV1"
VERSION = 1
HEADER_SIZE = 64
ENTRY_SIZE = 200
ALIGNMENT = 64
HEADER = struct.Struct("<8sHHIQQ16sIIQ")
ENTRY = struct.Struct("<HBBI8IQQfiII128s")
assert HEADER.size == HEADER_SIZE
assert ENTRY.size == ENTRY_SIZE

DTYPE_BYTES = 0
DTYPE_I8 = 1
DTYPE_I16 = 2
DTYPE_I32 = 3
DTYPE_F32 = 4
DTYPE_NAMES = {
    DTYPE_BYTES: "bytes",
    DTYPE_I8: "int8",
    DTYPE_I16: "int16",
    DTYPE_I32: "int32",
    DTYPE_F32: "float32",
}
FLAG_QUANTIZED = 1 << 0
FLAG_METADATA = 1 << 1


class PackageError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Section:
    name: str
    dtype: int
    shape: tuple[int, ...]
    data: bytes
    scale: float = 1.0
    zero_point: int = 0
    flags: int = 0

    def validate(self):
        encoded = self.name.encode("utf-8")
        if not encoded or len(encoded) > 128:
            raise PackageError("section name must occupy 1..128 UTF-8 bytes")
        if self.dtype not in DTYPE_NAMES:
            raise PackageError(f"unsupported dtype code {self.dtype}")
        if len(self.shape) > 8 or any(not 0 <= value <= 0xFFFFFFFF for value in self.shape):
            raise PackageError("section shape must contain at most eight u32 dimensions")
        if not self.data and self.name != "__metadata__":
            raise PackageError("non-metadata sections cannot be empty")


@dataclasses.dataclass(frozen=True)
class Package:
    model_uuid: bytes
    sections: tuple[Section, ...]
    metadata: dict

    def section(self, name):
        return next((section for section in self.sections if section.name == name), None)


def align(value, alignment=ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def metadata_section(metadata):
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return Section(
        name="__metadata__",
        dtype=DTYPE_BYTES,
        shape=(len(payload),),
        data=payload,
        flags=FLAG_METADATA,
    )


def build_package(sections, metadata, model_uuid=None):
    model_uuid = model_uuid or uuid.uuid4().bytes
    if len(model_uuid) != 16:
        raise PackageError("model UUID must be exactly 16 bytes")
    all_sections = [metadata_section(metadata), *sections]
    names = set()
    payload = bytearray()
    entries = []
    for section in all_sections:
        section.validate()
        if section.name in names:
            raise PackageError(f"duplicate section name {section.name!r}")
        names.add(section.name)
        offset = align(len(payload))
        payload.extend(b"\0" * (offset - len(payload)))
        payload.extend(section.data)
        shape = (*section.shape, *(0 for _ in range(8 - len(section.shape))))
        name = section.name.encode("utf-8")
        entries.append(
            ENTRY.pack(
                len(name),
                section.dtype,
                len(section.shape),
                section.flags,
                *shape,
                offset,
                len(section.data),
                float(section.scale),
                int(section.zero_point),
                zlib.crc32(section.data) & 0xFFFFFFFF,
                0,
                name.ljust(128, b"\0"),
            )
        )
    table = b"".join(entries)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_SIZE,
        len(entries),
        len(table),
        len(payload),
        model_uuid,
        zlib.crc32(table) & 0xFFFFFFFF,
        zlib.crc32(payload) & 0xFFFFFFFF,
        0,
    )
    return header + table + payload


def parse_package(blob):
    if len(blob) < HEADER_SIZE:
        raise PackageError("truncated UNARCHV1 header")
    (
        magic,
        version,
        header_size,
        section_count,
        table_bytes,
        payload_bytes,
        model_uuid,
        table_crc,
        payload_crc,
        reserved,
    ) = HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION or header_size != HEADER_SIZE or reserved != 0:
        raise PackageError("unsupported UNARCHV1 header")
    if table_bytes != section_count * ENTRY_SIZE:
        raise PackageError("section table size/count mismatch")
    expected = HEADER_SIZE + table_bytes + payload_bytes
    if len(blob) != expected:
        raise PackageError(f"package length {len(blob)} != declared {expected}")
    table = blob[HEADER_SIZE : HEADER_SIZE + table_bytes]
    payload = blob[HEADER_SIZE + table_bytes :]
    if zlib.crc32(table) & 0xFFFFFFFF != table_crc:
        raise PackageError("section table CRC32 mismatch")
    if zlib.crc32(payload) & 0xFFFFFFFF != payload_crc:
        raise PackageError("payload CRC32 mismatch")
    sections = []
    names = set()
    for index in range(section_count):
        values = ENTRY.unpack_from(table, index * ENTRY_SIZE)
        name_length, dtype, ndim, flags = values[:4]
        shape_values = values[4:12]
        offset, length, scale, zero_point, crc, entry_reserved, name_field = values[12:]
        if not 1 <= name_length <= 128 or ndim > 8 or entry_reserved != 0:
            raise PackageError(f"invalid section entry {index}")
        try:
            name = name_field[:name_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise PackageError(f"section {index} has invalid UTF-8 name") from error
        if name in names:
            raise PackageError(f"duplicate section name {name!r}")
        names.add(name)
        if dtype not in DTYPE_NAMES or offset % ALIGNMENT or offset + length > len(payload):
            raise PackageError(f"section {name!r} has invalid dtype/bounds/alignment")
        data = bytes(payload[offset : offset + length])
        if zlib.crc32(data) & 0xFFFFFFFF != crc:
            raise PackageError(f"section {name!r} CRC32 mismatch")
        sections.append(
            Section(name, dtype, tuple(shape_values[:ndim]), data, scale, zero_point, flags)
        )
    metadata_entry = next(
        (section for section in sections if section.name == "__metadata__"), None
    )
    if metadata_entry is None or not metadata_entry.flags & FLAG_METADATA:
        raise PackageError("missing metadata section")
    try:
        metadata = json.loads(metadata_entry.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageError("metadata section is not valid JSON") from error
    return Package(model_uuid, tuple(sections), metadata)


def read_package(path):
    return parse_package(Path(path).read_bytes())


def atomic_write(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
