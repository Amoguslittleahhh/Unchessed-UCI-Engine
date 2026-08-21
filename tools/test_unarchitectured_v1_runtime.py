#!/usr/bin/env python3

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import unarchitectured_v1_package as PACKAGE
import inspect_unarchitectured_v1 as INSPECTOR


class UnarchitecturedV1RuntimePackageTests(unittest.TestCase):
    def fixture(self):
        metadata = {
            "architecture": "Unarchitectured v1",
            "format": "UNARCHV1",
            "calibration": {"upper_regret_quantile": 1.25},
            "tensor_count": 2,
        }
        sections = [
            PACKAGE.Section(
                "linear.weight",
                PACKAGE.DTYPE_I8,
                (2, 2),
                bytes((1, 254, 3, 252)),
                scale=0.25,
                flags=PACKAGE.FLAG_QUANTIZED,
            ),
            PACKAGE.Section(
                "linear.bias",
                PACKAGE.DTYPE_F32,
                (2,),
                b"\x00\x00\x80?\x00\x00\x00\xc0",
            ),
        ]
        return PACKAGE.build_package(sections, metadata, model_uuid=bytes(range(16)))

    def test_round_trip_and_inspector(self):
        blob = self.fixture()
        package = PACKAGE.parse_package(blob)
        self.assertEqual(package.model_uuid, bytes(range(16)))
        self.assertEqual(package.section("linear.weight").shape, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.unarch"
            PACKAGE.atomic_write(path, blob)
            report = INSPECTOR.inspect(path)
        self.assertTrue(report["passed"])
        self.assertEqual(report["model_uuid"], bytes(range(16)).hex())
        self.assertEqual(report["sha256"], hashlib.sha256(blob).hexdigest())

    def test_corruption_and_duplicate_sections_are_rejected(self):
        blob = bytearray(self.fixture())
        blob[-1] ^= 1
        with self.assertRaisesRegex(PACKAGE.PackageError, "payload CRC32"):
            PACKAGE.parse_package(blob)
        section = PACKAGE.Section("duplicate", PACKAGE.DTYPE_I8, (1,), b"\0")
        with self.assertRaisesRegex(PACKAGE.PackageError, "duplicate"):
            PACKAGE.build_package([section, section], {"x": 1})

    def test_shape_and_alignment_contracts(self):
        blob = self.fixture()
        package = PACKAGE.parse_package(blob)
        for section in package.sections:
            section.validate()
        self.assertEqual(PACKAGE.HEADER.size, 64)
        self.assertEqual(PACKAGE.ENTRY.size, 200)


if __name__ == "__main__":
    unittest.main()
