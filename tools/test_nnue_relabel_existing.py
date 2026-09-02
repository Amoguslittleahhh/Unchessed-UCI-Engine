#!/usr/bin/env python3
from __future__ import annotations

import ast
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TOOL = TOOLS / "nnue_relabel_existing.py"
REC_SIZE = 104


def pack_record(score: int, wdl: int = 1) -> bytes:
    rec = bytearray(REC_SIZE)
    struct.pack_into("<Q", rec, 5 * 8, 1 << 4)
    struct.pack_into("<Q", rec, 11 * 8, 1 << 60)
    struct.pack_into("<h", rec, 96, score)
    rec[98] = wdl
    return bytes(rec)


def make_shard(path: Path, n: int = 8) -> list[int]:
    scores = [i * 10 for i in range(n)]
    with open(path, "wb") as f:
        for s in scores:
            f.write(pack_record(s))
    return scores


def run(args):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_help_standalone():
    out = run(["--help"])
    assert out.returncode == 0, out.stderr
    assert "sidecar" in out.stdout or "score" in out.stdout


def test_apply_replaces_scores_keeps_boards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shard = tmp / "old.bin"
        old = make_shard(shard)
        new_scores = [s + 25 for s in old]
        scores_path = tmp / "new.i16"
        with open(scores_path, "wb") as f:
            f.write(struct.pack("<" + "h" * len(new_scores), *new_scores))
        out_path = tmp / "out.bin"
        proc = run(["apply", str(shard), str(scores_path), str(out_path)])
        assert proc.returncode == 0, proc.stderr + proc.stdout
        raw = out_path.read_bytes()
        assert len(raw) == REC_SIZE * len(old)
        for i, want in enumerate(new_scores):
            rec = raw[i * REC_SIZE : (i + 1) * REC_SIZE]
            got = struct.unpack_from("<h", rec, 96)[0]
            assert got == want
            assert rec[:96] == pack_record(old[i])[:96]
        assert "mae=" in proc.stdout


def test_length_mismatch_is_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shard = tmp / "old.bin"
        make_shard(shard, n=4)
        scores_path = tmp / "new.i16"
        with open(scores_path, "wb") as f:
            f.write(struct.pack("<hhh", 0, 0, 0))
        proc = run(["compare", str(shard), str(scores_path)])
        assert proc.returncode != 0
        assert "ERROR" in proc.stderr or "ERROR" in proc.stdout


def test_only_stdlib_imports():
    src = TOOL.read_text()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods == {
        "__future__",
        "argparse",
        "math",
        "os",
        "struct",
        "sys",
    }, mods


if __name__ == "__main__":
    test_help_standalone()
    test_apply_replaces_scores_keeps_boards()
    test_length_mismatch_is_fatal()
    test_only_stdlib_imports()
    print("ok")
