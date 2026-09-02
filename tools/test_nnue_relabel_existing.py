#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TOOL = TOOLS / "nnue_relabel_existing.py"
REC_SIZE = 104


def pack_record(score: int, wdl: int = 1, *, white_king: int = 4, black_king: int = 60) -> bytes:
    rec = bytearray(REC_SIZE)
    struct.pack_into("<Q", rec, 5 * 8, 1 << white_king)
    struct.pack_into("<Q", rec, 11 * 8, 1 << black_king)
    struct.pack_into("<h", rec, 96, score)
    rec[98] = wdl
    return bytes(rec)


def make_shard(path: Path, n: int = 8) -> list[int]:
    scores = [i * 10 - 30 for i in range(n)]
    with open(path, "wb") as handle:
        for score in scores:
            handle.write(pack_record(score))
    return scores


def write_scores(path: Path, scores: list[int]) -> None:
    path.write_bytes(struct.pack("<" + "h" * len(scores), *scores))


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def result_json(proc: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(proc.stdout.splitlines()[-1])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporary_files(directory: Path, destination: Path) -> list[Path]:
    return list(directory.glob(f".{destination.name}.*.tmp"))


def test_help_standalone():
    out = run(["--help"])
    assert out.returncode == 0, out.stderr
    assert "sidecar" in out.stdout or "score" in out.stdout


def test_apply_replaces_scores_keeps_all_immutable_bytes_and_verifies():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        old = make_shard(shard)
        scores = tmp / "new.i16"
        new_scores = [score + 25 for score in old]
        write_scores(scores, new_scores)
        output = tmp / "out.bin"
        proc = run(["apply", str(shard), str(scores), str(output), "--chunk-records", "2"])
        assert proc.returncode == 0, proc.stderr + proc.stdout
        payload = result_json(proc)
        assert payload["statistics"]["n"] == len(old)
        assert payload["verification"]["records"] == len(old)
        raw = output.read_bytes()
        assert len(raw) == REC_SIZE * len(old)
        original = shard.read_bytes()
        for index, wanted in enumerate(new_scores):
            before = original[index * REC_SIZE : (index + 1) * REC_SIZE]
            after = raw[index * REC_SIZE : (index + 1) * REC_SIZE]
            assert struct.unpack_from("<h", after, 96)[0] == wanted
            assert before[:96] == after[:96]
            assert before[98:] == after[98:]
        verified = run(["verify", str(shard), str(scores), str(output), "--chunk-records", "3", "--json"])
        assert verified.returncode == 0, verified.stderr + verified.stdout
        assert result_json(verified)["verification"]["output_sha256"] == sha256(output)


def test_refuses_existing_destination_without_force_and_preserves_it():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        old = make_shard(shard)
        scores = tmp / "new.i16"
        write_scores(scores, [score + 1 for score in old])
        output = tmp / "out.bin"
        output.write_bytes(b"retain existing destination")
        before = output.read_bytes()
        proc = run(["apply", str(shard), str(scores), str(output)])
        assert proc.returncode != 0
        assert "already exists" in proc.stderr
        assert output.read_bytes() == before
        forced = run(["apply", str(shard), str(scores), str(output), "--force"])
        assert forced.returncode == 0, forced.stderr + forced.stdout


def test_aliases_source_sidecar_hardlink_symlink_and_manifest_are_rejected_unchanged():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        old = make_shard(shard)
        scores = tmp / "new.i16"
        write_scores(scores, [score + 7 for score in old])
        manifest = tmp / "labels.json"
        created = run(["manifest", str(shard), str(scores), str(manifest), "--json"])
        assert created.returncode == 0, created.stderr + created.stdout
        before = {path: sha256(path) for path in (shard, scores, manifest)}
        for destination in (shard, scores, manifest):
            proc = run(["apply", str(shard), str(scores), str(destination), "--manifest", str(manifest), "--force"])
            assert proc.returncode != 0
            assert "aliases declared input" in proc.stderr
        hardlink = tmp / "hardlink.bin"
        os.link(shard, hardlink)
        proc = run(["apply", str(shard), str(scores), str(hardlink), "--force"])
        assert proc.returncode != 0
        assert "aliases declared input" in proc.stderr
        symlink = tmp / "symlink.bin"
        symlink.symlink_to(shard.name)
        proc = run(["apply", str(shard), str(scores), str(symlink), "--force"])
        assert proc.returncode != 0
        assert "aliases declared input" in proc.stderr
        assert {path: sha256(path) for path in (shard, scores, manifest)} == before


def test_malformed_record_fails_before_publishing_and_keeps_old_destination():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "bad.bin"
        old = make_shard(shard, 3)
        raw = bytearray(shard.read_bytes())
        raw[REC_SIZE + 98] = 3
        shard.write_bytes(raw)
        scores = tmp / "new.i16"
        write_scores(scores, old)
        output = tmp / "out.bin"
        output.write_bytes(b"do not replace")
        old_output = output.read_bytes()
        proc = run(["apply", str(shard), str(scores), str(output), "--force", "--chunk-records", "1"])
        assert proc.returncode != 0
        assert "WDL" in proc.stderr
        assert output.read_bytes() == old_output
        assert not temporary_files(tmp, output)


def test_structural_validation_rejects_padding_kings_and_overlap_and_legacy_off_is_explicit():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        scores = tmp / "new.i16"
        write_scores(scores, [1])
        cases = []
        padding = bytearray(pack_record(0))
        padding[99] = 1
        cases.append(("padding", padding, "padding"))
        king = bytearray(pack_record(0))
        struct.pack_into("<Q", king, 5 * 8, 0)
        cases.append(("king", king, "exactly one king"))
        overlap = bytearray(pack_record(0))
        struct.pack_into("<Q", overlap, 0, 1 << 12)
        struct.pack_into("<Q", overlap, 8, 1 << 12)
        cases.append(("overlap", overlap, "piece-plane overlap"))
        for name, record, expected in cases:
            shard = tmp / f"{name}.bin"
            shard.write_bytes(record)
            proc = run(["compare", str(shard), str(scores)])
            assert proc.returncode != 0
            assert expected in proc.stderr
        # The deliberately explicit legacy policy suppresses only king/overlap,
        # never WDL/padding ABI checks.
        structural_only = tmp / "structural-only.bin"
        structural_only.write_bytes(cases[-1][1])
        proc = run(["compare", str(structural_only), str(scores), "--structural-check", "off"])
        assert proc.returncode == 0, proc.stderr + proc.stdout


def test_length_mismatch_odd_score_and_malformed_shard_size_are_fatal():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        make_shard(shard, 4)
        short_scores = tmp / "short.i16"
        write_scores(short_scores, [0, 0, 0])
        proc = run(["compare", str(shard), str(short_scores)])
        assert proc.returncode != 0
        assert "ERROR" in proc.stderr
        odd_scores = tmp / "odd.i16"
        odd_scores.write_bytes(b"x")
        proc = run(["compare", str(shard), str(odd_scores)])
        assert proc.returncode != 0
        assert "odd size" in proc.stderr
        malformed = tmp / "bad-size.bin"
        malformed.write_bytes(b"x")
        proc = run(["compare", str(malformed), str(short_scores)])
        assert proc.returncode != 0
        assert "not a multiple" in proc.stderr


def test_chunk_size_invariance_for_compare_and_apply():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        old = make_shard(shard, 101)
        scores = tmp / "new.i16"
        new = [(-score if index % 3 == 0 else score + index) for index, score in enumerate(old)]
        write_scores(scores, new)
        compared = []
        outputs = []
        for chunk_records in (1, 7, 256):
            compared_proc = run(["compare", str(shard), str(scores), "--chunk-records", str(chunk_records), "--json"])
            assert compared_proc.returncode == 0, compared_proc.stderr + compared_proc.stdout
            compared.append(result_json(compared_proc)["statistics"])
            output = tmp / f"out-{chunk_records}.bin"
            applied = run(["apply", str(shard), str(scores), str(output), "--chunk-records", str(chunk_records), "--json"])
            assert applied.returncode == 0, applied.stderr + applied.stdout
            outputs.append(output.read_bytes())
            assert result_json(applied)["statistics"] == compared[-1]
        assert compared[0] == compared[1] == compared[2]
        assert outputs[0] == outputs[1] == outputs[2]


def test_verify_rejects_wrong_score_and_immutable_corruption():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        old = make_shard(shard, 4)
        scores = tmp / "new.i16"
        write_scores(scores, [score + 8 for score in old])
        output = tmp / "out.bin"
        assert run(["apply", str(shard), str(scores), str(output)]).returncode == 0
        wrong_score = bytearray(output.read_bytes())
        struct.pack_into("<h", wrong_score, 96, 1234)
        output.write_bytes(wrong_score)
        proc = run(["verify", str(shard), str(scores), str(output)])
        assert proc.returncode != 0
        assert "does not match sidecar" in proc.stderr
        assert output.read_bytes() == wrong_score
        assert run(["apply", str(shard), str(scores), str(output), "--force"]).returncode == 0
        immutable = bytearray(output.read_bytes())
        immutable[0] ^= 1
        output.write_bytes(immutable)
        proc = run(["verify", str(shard), str(scores), str(output)])
        assert proc.returncode != 0
        assert "immutable fields differ" in proc.stderr
        assert output.read_bytes() == immutable


def test_versioned_manifest_binds_exact_source_scores_and_optional_provenance():
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        shard = tmp / "old.bin"
        old = make_shard(shard, 3)
        scores = tmp / "new.i16"
        write_scores(scores, [score + 2 for score in old])
        manifest = tmp / "labels.json"
        proc = run([
            "manifest", str(shard), str(scores), str(manifest), "--provenance-json",
            '{"teacher_type":"fixture-mock","nodes":1}', "--json",
        ])
        assert proc.returncode == 0, proc.stderr + proc.stdout
        data = json.loads(manifest.read_text())
        assert data["schema"] == "unchessed.nnue-label-package"
        assert data["version"] == 1
        assert data["source"]["immutable_sha256"]
        assert data["provenance"]["declared"]["teacher_type"] == "fixture-mock"
        output = tmp / "out.bin"
        proc = run(["apply", str(shard), str(scores), str(output), "--manifest", str(manifest)])
        assert proc.returncode == 0, proc.stderr + proc.stdout
        assert run(["verify", str(shard), str(scores), str(output), "--manifest", str(manifest)]).returncode == 0
        altered_scores = list(struct.unpack("<hhh", scores.read_bytes()))
        altered_scores[0] += 1
        write_scores(scores, altered_scores)
        rejected = run(["compare", str(shard), str(scores), "--manifest", str(manifest)])
        assert rejected.returncode != 0
        assert "sha256 does not match" in rejected.stderr


def test_only_stdlib_imports():
    source = TOOL.read_text()
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules == {
        "__future__", "argparse", "datetime", "hashlib", "json", "math", "os",
        "pathlib", "shutil", "struct", "sys", "tempfile",
    }, modules


if __name__ == "__main__":
    test_help_standalone()
    test_apply_replaces_scores_keeps_all_immutable_bytes_and_verifies()
    test_refuses_existing_destination_without_force_and_preserves_it()
    test_aliases_source_sidecar_hardlink_symlink_and_manifest_are_rejected_unchanged()
    test_malformed_record_fails_before_publishing_and_keeps_old_destination()
    test_structural_validation_rejects_padding_kings_and_overlap_and_legacy_off_is_explicit()
    test_length_mismatch_odd_score_and_malformed_shard_size_are_fatal()
    test_chunk_size_invariance_for_compare_and_apply()
    test_verify_rejects_wrong_score_and_immutable_corruption()
    test_versioned_manifest_binds_exact_source_scores_and_optional_provenance()
    test_only_stdlib_imports()
    print("ok")
