#!/usr/bin/env python3
"""Safely replace score labels in fixed-record NNUE shards.

The shard ABI is 104 bytes per record: twelve little-endian u64 piece planes,
an STM-perspective little-endian i16 score at offsets 96--97, one WDL byte,
and five zero pad bytes. This transformer does not reconstruct full chess
state, run a search, or claim to create high-node labels.

Legacy commands remain available:
  nnue_relabel_existing.py compare <old.bin> <new_scores.i16>
  nnue_relabel_existing.py apply <old.bin> <new_scores.i16> <out.bin>

``apply`` streams bounded chunks, refuses destination/input aliases, refuses
existing destinations without ``--force``, writes a same-directory temporary
file, fsyncs it, verifies it in a fresh pass, then atomically replaces the
destination. ``verify`` independently proves that only the score bytes changed.
The optional v1 ``manifest`` command provides a source/sidecar hash binding; it
does not pretend that this 104-byte format retains full-position state.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

REC_SIZE = 104
BB_BYTES = 12 * 8
SCORE_OFF = BB_BYTES  # 96
WDL_OFF = 98
PAD_OFF = 99
DEFAULT_CHUNK_RECORDS = 8192
DEFAULT_SAFETY_MARGIN_BYTES = 64 * 1024 * 1024
MANIFEST_SCHEMA = "unchessed.nnue-label-package"
MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 1024 * 1024


class RelabelError(Exception):
    """Expected validation or operational failure with a safe user message."""


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return result


def file_size(path: str | Path, label: str) -> int:
    try:
        result = os.stat(path)
    except OSError as exc:
        raise RelabelError(f"{label} is not readable: {path}: {exc.strerror}") from exc
    if not os.path.isfile(path):
        raise RelabelError(f"{label} is not a regular file: {path}")
    return result.st_size


def shard_layout(path: str | Path) -> tuple[int, int]:
    size = file_size(path, "source shard")
    if size % REC_SIZE:
        raise RelabelError(f"{path} size {size} is not a multiple of {REC_SIZE}")
    if size == 0:
        raise RelabelError(f"{path} is empty")
    return size, size // REC_SIZE


def score_layout(path: str | Path, expected_records: int) -> tuple[int, int]:
    size = file_size(path, "score sidecar")
    if size % 2:
        raise RelabelError(f"{path} is not packed i16 (odd size {size})")
    records = size // 2
    if records != expected_records:
        raise RelabelError(
            f"{path} has {records} scores, shard has {expected_records} records"
        )
    return size, records


def _is_exactly_one_bit(value: int) -> bool:
    return value != 0 and (value & (value - 1)) == 0


def validate_record(
    record: bytes | bytearray, path: str | Path, index: int, structural: bool
) -> None:
    """Validate documented byte-level invariants, not complete chess legality."""
    wdl = record[WDL_OFF]
    if wdl > 2:
        raise RelabelError(f"{path}: record {index}: WDL byte {wdl} is outside 0..2")
    if record[PAD_OFF:] != b"\x00" * 5:
        raise RelabelError(f"{path}: record {index}: nonzero padding")
    if not structural:
        return
    planes = struct.unpack_from("<12Q", record, 0)
    if not _is_exactly_one_bit(planes[5]) or not _is_exactly_one_bit(planes[11]):
        raise RelabelError(
            f"{path}: record {index}: expected exactly one king in planes 5 and 11"
        )
    occupied = 0
    for plane_index, plane in enumerate(planes):
        if occupied & plane:
            raise RelabelError(
                f"{path}: record {index}: piece-plane overlap at plane {plane_index}"
            )
        occupied |= plane


def _read_exact(handle, size: int, description: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise RelabelError(f"short read while reading {description}")
    return data


def sha256_file(path: str | Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            data = handle.read(block_bytes)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def scan_source(
    shard: str | Path, records: int, chunk_records: int, structural: bool
) -> dict[str, object]:
    """Stream source validation plus whole-file and immutable-payload digests."""
    source_digest = hashlib.sha256()
    immutable_digest = hashlib.sha256()
    with open(shard, "rb") as handle:
        for start in range(0, records, chunk_records):
            count = min(chunk_records, records - start)
            chunk = _read_exact(handle, count * REC_SIZE, f"source shard {shard}")
            source_digest.update(chunk)
            for offset in range(0, len(chunk), REC_SIZE):
                record = chunk[offset : offset + REC_SIZE]
                validate_record(record, shard, start + offset // REC_SIZE, structural)
                immutable_digest.update(record[:SCORE_OFF])
                immutable_digest.update(record[WDL_OFF:])
    return {
        "bytes": records * REC_SIZE,
        "records": records,
        "sha256": source_digest.hexdigest(),
        "immutable_sha256": immutable_digest.hexdigest(),
    }


class RunningStats:
    """One-pass bivariate Welford statistics; no score/record history is held."""

    def __init__(self) -> None:
        self.n = 0
        self.old_mean = 0.0
        self.new_mean = 0.0
        self.old_m2 = 0.0
        self.new_m2 = 0.0
        self.covariance_sum = 0.0
        self.abs_sum = 0
        self.square_sum = 0
        self.changed = 0
        self.sign_flips = 0

    def update(self, old: int, new: int) -> None:
        self.n += 1
        count = self.n
        old_delta = float(old) - self.old_mean
        self.old_mean += old_delta / count
        new_delta = float(new) - self.new_mean
        self.new_mean += new_delta / count
        self.old_m2 += old_delta * (float(old) - self.old_mean)
        self.new_m2 += new_delta * (float(new) - self.new_mean)
        self.covariance_sum += old_delta * (float(new) - self.new_mean)
        delta = new - old
        self.abs_sum += abs(delta)
        self.square_sum += delta * delta
        if old != new:
            self.changed += 1
        if (old < 0 < new) or (new < 0 < old):
            self.sign_flips += 1

    def report(self) -> dict[str, object]:
        if self.n == 0:
            raise RelabelError("no records were processed")
        denominator = math.sqrt(self.old_m2 * self.new_m2)
        return {
            "n": self.n,
            "mae_cp": self.abs_sum / self.n,
            "rms_cp": math.sqrt(self.square_sum / self.n),
            "changed": self.changed,
            "frac_changed": self.changed / self.n,
            "sign_flips": self.sign_flips,
            "pearson": None if denominator == 0.0 else self.covariance_sum / denominator,
            "old_mean": self.old_mean,
            "new_mean": self.new_mean,
        }


def stream_compare(
    shard: str | Path,
    scores: str | Path,
    records: int,
    chunk_records: int,
    structural: bool,
    writer=None,
) -> dict[str, object]:
    """Stream source and sidecar in matching chunks; optionally transform to writer."""
    stats = RunningStats()
    with open(shard, "rb") as source_handle, open(scores, "rb") as score_handle:
        for start in range(0, records, chunk_records):
            count = min(chunk_records, records - start)
            source_chunk = _read_exact(
                source_handle, count * REC_SIZE, f"source shard {shard}"
            )
            score_chunk = _read_exact(score_handle, count * 2, f"score sidecar {scores}")
            output_chunk = bytearray(source_chunk) if writer is not None else None
            for local_index in range(count):
                record_offset = local_index * REC_SIZE
                score_offset = local_index * 2
                record = source_chunk[record_offset : record_offset + REC_SIZE]
                validate_record(record, shard, start + local_index, structural)
                old_score = struct.unpack_from("<h", record, SCORE_OFF)[0]
                new_score = struct.unpack_from("<h", score_chunk, score_offset)[0]
                stats.update(old_score, new_score)
                if output_chunk is not None:
                    struct.pack_into("<h", output_chunk, record_offset + SCORE_OFF, new_score)
            if output_chunk is not None:
                writer.write(output_chunk)
    return stats.report()


def load_manifest(path: str | Path) -> dict[str, object]:
    size = file_size(path, "manifest")
    if size > MAX_MANIFEST_BYTES:
        raise RelabelError(f"manifest {path} is larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelabelError(f"invalid manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RelabelError(f"invalid manifest {path}: root must be an object")
    return data


def _object(value: object, field: str, manifest_path: str | Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RelabelError(f"invalid manifest {manifest_path}: {field} must be an object")
    return value


def _value(mapping: dict[str, object], field: str, manifest_path: str | Path):
    if field not in mapping:
        raise RelabelError(f"invalid manifest {manifest_path}: missing {field}")
    return mapping[field]


def validate_manifest(
    manifest_path: str | Path,
    shard: str | Path,
    scores: str | Path,
    shard_bytes: int,
    records: int,
    score_bytes: int,
    chunk_records: int,
    structural: bool,
) -> dict[str, object]:
    """Fail closed unless the v1 manifest binds these exact input bytes."""
    manifest = load_manifest(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != MANIFEST_VERSION:
        raise RelabelError(
            f"invalid manifest {manifest_path}: expected {MANIFEST_SCHEMA} v{MANIFEST_VERSION}"
        )
    if manifest.get("record_size") != REC_SIZE:
        raise RelabelError(f"invalid manifest {manifest_path}: record_size must be {REC_SIZE}")
    if manifest.get("score_format") != {
        "encoding": "little-endian-i16", "pov": "stm", "unit": "cp"
    }:
        raise RelabelError(f"invalid manifest {manifest_path}: unsupported score_format")
    source = _object(manifest.get("source"), "source", manifest_path)
    score_info = _object(manifest.get("scores"), "scores", manifest_path)
    expected_source = {"name": Path(shard).name, "bytes": shard_bytes, "records": records}
    expected_scores = {"name": Path(scores).name, "bytes": score_bytes, "records": records}
    for field, expected in expected_source.items():
        if _value(source, field, manifest_path) != expected:
            raise RelabelError(f"manifest {manifest_path}: source {field} does not match input")
    for field, expected in expected_scores.items():
        if _value(score_info, field, manifest_path) != expected:
            raise RelabelError(f"manifest {manifest_path}: scores {field} does not match input")
    scanned_source = scan_source(shard, records, chunk_records, structural)
    score_sha256 = sha256_file(scores)
    for field in ("sha256", "immutable_sha256"):
        if _value(source, field, manifest_path) != scanned_source[field]:
            raise RelabelError(f"manifest {manifest_path}: source {field} does not match input")
    if _value(score_info, "sha256", manifest_path) != score_sha256:
        raise RelabelError(f"manifest {manifest_path}: scores sha256 does not match input")
    return manifest


def _check_manifest_after_verify(
    manifest: dict[str, object] | None, output: str | Path, result: dict[str, object]
) -> None:
    if manifest is None:
        return
    source = _object(manifest.get("source"), "source", "provided manifest")
    scores = _object(manifest.get("scores"), "scores", "provided manifest")
    for field, actual in (
        ("sha256", result["source_sha256"]),
        ("immutable_sha256", result["source_immutable_sha256"]),
    ):
        if _value(source, field, "provided manifest") != actual:
            raise RelabelError(f"manifest source {field} changed before verification completed")
    if _value(scores, "sha256", "provided manifest") != result["scores_sha256"]:
        raise RelabelError("manifest score sidecar changed before verification completed")
    if "output" not in manifest:
        return
    expected = _object(manifest["output"], "output", "provided manifest")
    for field, actual in (
        ("bytes", result["bytes"]),
        ("records", result["records"]),
        ("sha256", result["output_sha256"]),
    ):
        if _value(expected, field, "provided manifest") != actual:
            raise RelabelError(f"manifest output {field} does not match {output}")
    if "name" in expected and expected["name"] != Path(output).name:
        raise RelabelError(f"manifest output name does not match {output}")


def verify_stream(
    shard: str | Path,
    scores: str | Path,
    output: str | Path,
    records: int,
    shard_bytes: int,
    chunk_records: int,
    structural: bool,
    manifest: dict[str, object] | None,
) -> dict[str, object]:
    """Fresh-pass byte comparison proving score-only output modification."""
    output_size = file_size(output, "output shard")
    if output_size != shard_bytes:
        raise RelabelError(
            f"{output} has {output_size} bytes; expected {shard_bytes} bytes from source"
        )
    source_digest = hashlib.sha256()
    immutable_digest = hashlib.sha256()
    score_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    with (
        open(shard, "rb") as source_handle,
        open(scores, "rb") as score_handle,
        open(output, "rb") as output_handle,
    ):
        for start in range(0, records, chunk_records):
            count = min(chunk_records, records - start)
            source_chunk = _read_exact(source_handle, count * REC_SIZE, f"source shard {shard}")
            score_chunk = _read_exact(score_handle, count * 2, f"score sidecar {scores}")
            output_chunk = _read_exact(output_handle, count * REC_SIZE, f"output shard {output}")
            source_digest.update(source_chunk)
            score_digest.update(score_chunk)
            output_digest.update(output_chunk)
            for local_index in range(count):
                offset = local_index * REC_SIZE
                score_offset = local_index * 2
                index = start + local_index
                source_record = source_chunk[offset : offset + REC_SIZE]
                output_record = output_chunk[offset : offset + REC_SIZE]
                validate_record(source_record, shard, index, structural)
                validate_record(output_record, output, index, structural)
                immutable_digest.update(source_record[:SCORE_OFF])
                immutable_digest.update(source_record[WDL_OFF:])
                if (
                    source_record[:SCORE_OFF] != output_record[:SCORE_OFF]
                    or source_record[WDL_OFF:] != output_record[WDL_OFF:]
                ):
                    raise RelabelError(
                        f"{output}: record {index}: immutable fields differ from source"
                    )
                expected_score = struct.unpack_from("<h", score_chunk, score_offset)[0]
                actual_score = struct.unpack_from("<h", output_record, SCORE_OFF)[0]
                if actual_score != expected_score:
                    raise RelabelError(
                        f"{output}: record {index}: score {actual_score} does not match "
                        f"sidecar value {expected_score}"
                    )
    result = {
        "records": records,
        "bytes": shard_bytes,
        "source_sha256": source_digest.hexdigest(),
        "source_immutable_sha256": immutable_digest.hexdigest(),
        "scores_sha256": score_digest.hexdigest(),
        "output_sha256": output_digest.hexdigest(),
    }
    _check_manifest_after_verify(manifest, output, result)
    return result


def _path_aliases(left: str | Path, right: str | Path) -> bool:
    """Detect lexical, resolved/symlink, and existing hard-link aliases."""
    left_text = os.fspath(left)
    right_text = os.fspath(right)
    try:
        if os.path.samefile(left_text, right_text):
            return True
    except OSError:
        pass
    try:
        return Path(left_text).resolve(strict=False) == Path(right_text).resolve(strict=False)
    except OSError:
        return os.path.abspath(left_text) == os.path.abspath(right_text)


def assert_not_input_alias(output: str | Path, inputs: list[tuple[str, str | Path]]) -> None:
    for label, input_path in inputs:
        if _path_aliases(output, input_path):
            raise RelabelError(
                f"destination {output} aliases declared input {label} {input_path}"
            )


def assert_safe_destination(
    output: str | Path, inputs: list[tuple[str, str | Path]], force: bool
) -> Path:
    destination = Path(output)
    if not destination.parent.is_dir():
        raise RelabelError(f"destination directory does not exist: {destination.parent}")
    if os.path.lexists(destination) and destination.is_dir():
        raise RelabelError(f"destination is a directory: {destination}")
    assert_not_input_alias(destination, inputs)
    if os.path.lexists(destination) and not force:
        raise RelabelError(
            f"destination already exists: {destination} (use --force after checking it)"
        )
    return destination


def check_destination_space(destination: Path, output_bytes: int, safety_margin_bytes: int) -> None:
    available = shutil.disk_usage(destination.parent).free
    needed = output_bytes + safety_margin_bytes
    if available < needed:
        raise RelabelError(
            f"destination filesystem has {available} free bytes; need {needed} "
            f"({output_bytes} output + {safety_margin_bytes} safety margin)"
        )


def fsync_directory(directory: Path) -> None:
    """Best-effort durability for the directory entry created by a rename.

    Windows has no equivalent of opening a directory for fsync (NTFS
    durability for a rename is handled differently at the OS level, and
    the attempt fails with a permission error rather than succeeding or
    cleanly reporting "unsupported"). The per-file fsync elsewhere in this
    module already guarantees the file *contents* are durable before the
    atomic rename; this call is an additional POSIX-specific guarantee for
    the directory metadata, so skip it (not silently -- see the printed
    note) rather than fail the whole publish on a platform that cannot do
    it at all.
    """
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise RelabelError(f"cannot fsync destination directory {directory}: {exc.strerror}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise RelabelError(f"cannot fsync destination directory {directory}: {exc.strerror}") from exc
    finally:
        os.close(descriptor)


def atomic_apply(
    shard: str | Path,
    scores: str | Path,
    destination: Path,
    records: int,
    shard_bytes: int,
    chunk_records: int,
    structural: bool,
    manifest: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Transform, fsync, independently verify, and only then atomically publish."""
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temp_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            statistics = stream_compare(
                shard, scores, records, chunk_records, structural, writer=handle
            )
            handle.flush()
            os.fsync(handle.fileno())
        verification = verify_stream(
            shard, scores, temporary, records, shard_bytes, chunk_records, structural, manifest
        )
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
        return statistics, verification
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(destination: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temp_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with open(temporary, "r", encoding="utf-8") as handle:
            json.load(handle)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def print_result(human: str, result: dict[str, object], json_only: bool) -> None:
    if not json_only:
        print(human)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def stats_human(stats: dict[str, object]) -> str:
    pearson = stats["pearson"]
    pearson_text = "nan" if pearson is None else f"{float(pearson):.4f}"
    display_stats = dict(stats)
    display_stats["pearson"] = pearson_text
    return (
        "n={n} mae={mae_cp:.2f}cp rms={rms_cp:.2f}cp "
        "changed={changed}/{n} ({frac_changed:.4f}) sign_flips={sign_flips} "
        "pearson={pearson} old_mean={old_mean:.1f} new_mean={new_mean:.1f}"
    ).format(**display_stats)


def input_details(args) -> tuple[int, int, int, dict[str, object] | None]:
    shard_bytes, records = shard_layout(args.shard)
    score_bytes, _ = score_layout(args.scores, records)
    manifest = None
    if args.manifest:
        manifest = validate_manifest(
            args.manifest,
            args.shard,
            args.scores,
            shard_bytes,
            records,
            score_bytes,
            args.chunk_records,
            args.structural_check == "required",
        )
    return shard_bytes, records, score_bytes, manifest


def declared_inputs(args) -> list[tuple[str, str | Path]]:
    result: list[tuple[str, str | Path]] = [
        ("source shard", args.shard),
        ("score sidecar", args.scores),
    ]
    if args.manifest:
        result.append(("manifest", args.manifest))
    return result


def cmd_compare(args) -> int:
    _shard_bytes, records, _score_bytes, manifest = input_details(args)
    statistics = stream_compare(
        args.shard, args.scores, records, args.chunk_records, args.structural_check == "required"
    )
    print_result(
        stats_human(statistics),
        {"command": "compare", "manifest": args.manifest, "statistics": statistics},
        args.json,
    )
    return 0


def cmd_apply(args) -> int:
    destination = assert_safe_destination(args.out, declared_inputs(args), args.force)
    shard_bytes, records, _score_bytes, manifest = input_details(args)
    check_destination_space(destination, shard_bytes, args.safety_margin_bytes)
    statistics, verification = atomic_apply(
        args.shard,
        args.scores,
        destination,
        records,
        shard_bytes,
        args.chunk_records,
        args.structural_check == "required",
        manifest,
    )
    result = {
        "command": "apply",
        "manifest": args.manifest,
        "output": str(destination),
        "statistics": statistics,
        "verification": verification,
    }
    print_result(
        "wrote {out} ({bytes} bytes) {stats}".format(
            out=destination, bytes=verification["bytes"], stats=stats_human(statistics)
        ),
        result,
        args.json,
    )
    return 0


def cmd_verify(args) -> int:
    # Verify writes nothing, but rejecting an input alias avoids falsely presenting
    # a source or sidecar as an independently verified output.
    assert_not_input_alias(args.out, declared_inputs(args))
    shard_bytes, records, _score_bytes, manifest = input_details(args)
    verification = verify_stream(
        args.shard,
        args.scores,
        args.out,
        records,
        shard_bytes,
        args.chunk_records,
        args.structural_check == "required",
        manifest,
    )
    print_result(
        "verified {out} n={records} bytes={bytes} output_sha256={output_sha256}".format(
            out=args.out, **verification
        ),
        {"command": "verify", "manifest": args.manifest, "output": str(args.out), "verification": verification},
        args.json,
    )
    return 0


def cmd_manifest(args) -> int:
    destination = assert_safe_destination(
        args.out, [("source shard", args.shard), ("score sidecar", args.scores)], args.force
    )
    shard_bytes, records = shard_layout(args.shard)
    score_bytes, _ = score_layout(args.scores, records)
    check_destination_space(destination, 1, args.safety_margin_bytes)
    source = scan_source(
        args.shard, records, args.chunk_records, args.structural_check == "required"
    )
    scores = {
        "name": Path(args.scores).name,
        "bytes": score_bytes,
        "records": records,
        "sha256": sha256_file(args.scores),
    }
    provenance: dict[str, object] = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "operator": os.environ.get("USER", "unknown"),
        "note": "Source/sidecar binding only; full position state and a search teacher are not represented.",
    }
    if args.provenance_json:
        try:
            declared = json.loads(args.provenance_json)
        except json.JSONDecodeError as exc:
            raise RelabelError(f"--provenance-json is not valid JSON: {exc}") from exc
        if not isinstance(declared, dict):
            raise RelabelError("--provenance-json must be a JSON object")
        provenance["declared"] = declared
    payload = {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "record_size": REC_SIZE,
        "score_format": {"encoding": "little-endian-i16", "pov": "stm", "unit": "cp"},
        "source": {"name": Path(args.shard).name, **source},
        "scores": scores,
        "provenance": provenance,
    }
    atomic_write_json(destination, payload)
    print_result(
        f"wrote manifest {destination} for {records} records",
        {
            "command": "manifest",
            "manifest": str(destination),
            "records": records,
            "source_sha256": source["sha256"],
            "scores_sha256": scores["sha256"],
        },
        args.json,
    )
    return 0


def add_common_options(parser: argparse.ArgumentParser, include_manifest: bool = True) -> None:
    parser.add_argument(
        "--chunk-records",
        type=positive_int,
        default=DEFAULT_CHUNK_RECORDS,
        help=f"maximum records held per stream chunk (default: {DEFAULT_CHUNK_RECORDS})",
    )
    parser.add_argument(
        "--structural-check",
        choices=("required", "off"),
        default="required",
        help="require king-count/plane-overlap checks, or retain legacy WDL/padding-only checks",
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            help="v1 source/score binding manifest to validate before processing",
        )
    parser.add_argument("--json", action="store_true", help="emit JSON only (otherwise summary plus JSONL)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    compare = sub.add_parser("compare", help="stream old-vs-new score statistics without writing")
    compare.add_argument("shard")
    compare.add_argument("scores")
    add_common_options(compare)
    apply = sub.add_parser("apply", help="atomically write and fresh-pass verify a relabeled shard")
    apply.add_argument("shard")
    apply.add_argument("scores")
    apply.add_argument("out")
    apply.add_argument("--force", action="store_true", help="replace a non-input existing destination")
    apply.add_argument(
        "--safety-margin-bytes",
        type=nonnegative_int,
        default=DEFAULT_SAFETY_MARGIN_BYTES,
        help=f"free-space reserve beyond one output (default: {DEFAULT_SAFETY_MARGIN_BYTES})",
    )
    add_common_options(apply)
    verify = sub.add_parser("verify", help="fresh-pass verify an existing relabeled shard")
    verify.add_argument("shard")
    verify.add_argument("scores")
    verify.add_argument("out")
    add_common_options(verify)
    manifest = sub.add_parser("manifest", help="create a v1 source/score binding manifest")
    manifest.add_argument("shard")
    manifest.add_argument("scores")
    manifest.add_argument("out")
    manifest.add_argument("--force", action="store_true", help="replace a non-input existing destination")
    manifest.add_argument(
        "--safety-margin-bytes",
        type=nonnegative_int,
        default=DEFAULT_SAFETY_MARGIN_BYTES,
        help=f"free-space reserve for atomic manifest publication (default: {DEFAULT_SAFETY_MARGIN_BYTES})",
    )
    manifest.add_argument(
        "--provenance-json",
        help="optional declared provenance JSON object; no teacher/state semantics are inferred",
    )
    add_common_options(manifest, include_manifest=False)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "compare":
            return cmd_compare(args)
        if args.cmd == "apply":
            return cmd_apply(args)
        if args.cmd == "verify":
            return cmd_verify(args)
        return cmd_manifest(args)
    except RelabelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(json.dumps({"command": args.cmd, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
