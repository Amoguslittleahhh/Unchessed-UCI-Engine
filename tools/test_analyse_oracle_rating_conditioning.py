#!/usr/bin/env python3
"""Dependency-free gates for the legacy-oracle conditioning analyser.

The production sweep needs a trusted PyTorch checkpoint, but these tests cover
all paths that must stay usable on a bare clone: help, missing inputs, runtime
package rejection, and explicit dual-Elo rejection with a small fake torch
module.  No test imports real torch or creates model weights.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "analyse_oracle_rating_conditioning.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


def load_tool_module():
    spec = importlib.util.spec_from_file_location("oracle_conditioning_tool_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BareEnvironmentCliTests(unittest.TestCase):
    def test_help_needs_no_torch_or_checkpoint(self):
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--oracle", result.stdout)
        self.assertIn("--corpus", result.stdout)

    def test_missing_oracle_is_explicit_and_writes_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            report = root / "must-not-exist.json"
            corpus.write_text('{"fen": "startpos"}\n', encoding="utf-8")
            missing = root / "absent-oracle.pt"
            result = run_cli(
                "--oracle", str(missing), "--corpus", str(corpus), "--json", str(report)
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                result.stderr, f"missing oracle checkpoint: {missing}\n"
            )
            self.assertFalse(report.exists())

    def test_missing_corpus_is_explicit_and_writes_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oracle = root / "present-but-never-opened.pt"
            report = root / "must-not-exist.json"
            oracle.write_bytes(b"not opened because corpus validation comes first")
            missing = root / "absent-corpus.jsonl"
            result = run_cli(
                "--oracle", str(oracle), "--corpus", str(missing), "--json", str(report)
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, f"missing corpus: {missing}\n")
            self.assertFalse(report.exists())

    def test_runtime_unarchv1_fixture_is_rejected_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime.unarchv1"
            corpus = root / "corpus.jsonl"
            runtime.write_bytes(b"UNARCHV1" + b"not a torch checkpoint")
            corpus.write_text('{"fen": "startpos"}\n', encoding="utf-8")
            result = run_cli("--oracle", str(runtime), "--corpus", str(corpus))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("unsupported oracle checkpoint format: UNARCHV1", result.stderr)
            self.assertIn("runtime packages", result.stderr)


class FormatFixtureTests(unittest.TestCase):
    def test_dual_elo_fixture_is_explicitly_rejected_before_model_import(self):
        module = load_tool_module()
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_load(*args, **kwargs):
            calls.append((args, kwargs))
            return {
                "format": "UNARCHV1_PRETRAIN_DUAL_ELO_V1",
                "config": {},
                "model": {},
            }

        fake_torch = types.SimpleNamespace(load=fake_load)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oracle = root / "dual-elo.pt"
            corpus = root / "corpus.jsonl"
            oracle.write_bytes(b"a non-runtime checkpoint fixture")
            corpus.write_text('{"fen": "startpos"}\n', encoding="utf-8")
            stderr = io.StringIO()
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with contextlib.redirect_stderr(stderr):
                    code = module.main(["--oracle", str(oracle), "--corpus", str(corpus)])

        self.assertEqual(code, 2)
        self.assertIn("unsupported oracle checkpoint format", stderr.getvalue())
        self.assertIn("UNARCHV1_PRETRAIN_DUAL_ELO_V1", stderr.getvalue())
        self.assertIn("two-input experiment", stderr.getvalue())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["map_location"], "cpu")
        self.assertIs(calls[0][1]["weights_only"], False)

    def test_legacy_loader_contract_remains_explicit(self):
        source = TOOL.read_text(encoding="utf-8")
        self.assertIn('LEGACY_FORMAT = "UNARCHV1_ORACLE_TRAINING_V1_DDP"', source)
        self.assertIn("torch.load(path, map_location=\"cpu\", weights_only=False)", source)
        self.assertIn('model.load_state_dict(checkpoint["model"], strict=True)', source)
        self.assertIn("torch.inference_mode()", source)


if __name__ == "__main__":
    unittest.main()
