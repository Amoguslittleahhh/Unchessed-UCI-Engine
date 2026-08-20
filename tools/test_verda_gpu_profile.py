#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
MODULE_PATH = TOOLS / "verda_gpu_profile.py"
SPEC = importlib.util.spec_from_file_location("verda_gpu_profile", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ROOT = TOOLS.parent
PROFILES = json.loads((ROOT / "config/verda_gpu_profiles.json").read_text())
BASE = json.loads((ROOT / "config/a100_hydra_v5_training.json").read_text())


class VerdaGpuProfileTests(unittest.TestCase):
    def resolve(self, name, memory, count=1):
        return MODULE.resolve(
            BASE,
            PROFILES,
            [{"index": index, "name": name, "memory_total_mib": memory} for index in range(count)],
        )

    def test_every_advertised_family_has_a_profile(self):
        cases = [
            ("NVIDIA GB300", 294912, "blackwell-xxl"),
            ("NVIDIA B300", 274432, "blackwell-xxl"),
            ("NVIDIA B200", 184320, "blackwell-xxl"),
            ("NVIDIA H200", 144384, "hopper-xl"),
            ("NVIDIA H100 80GB HBM3", 81920, "80gb-large"),
            ("NVIDIA A100-SXM4-80GB", 81920, "80gb-large"),
            ("NVIDIA RTX PRO 6000 Blackwell", 98304, "80gb-large"),
            ("NVIDIA A100-SXM4-40GB", 40960, "40-48gb-base"),
            ("NVIDIA L40S", 49152, "40-48gb-base"),
            ("NVIDIA RTX 6000 Ada Generation", 49152, "40-48gb-base"),
            ("NVIDIA RTX A6000", 49152, "40-48gb-base"),
            ("Tesla V100-SXM2-16GB", 16384, "v100-compat"),
        ]
        for name, memory, expected in cases:
            with self.subTest(name=name):
                resolved = self.resolve(name, memory)
                self.assertEqual(resolved["hardware"]["resolved_gpu_profile"], expected)

    def test_oracle_capacity_increases_with_vram(self):
        v100 = self.resolve("Tesla V100-SXM2-16GB", 16384)
        a10040 = self.resolve("NVIDIA A100-SXM4-40GB", 40960)
        a10080 = self.resolve("NVIDIA A100-SXM4-80GB", 81920)
        h200 = self.resolve("NVIDIA H200", 144384)
        b300 = self.resolve("NVIDIA B300", 274432)
        counts = [
            config["oracle"]["expected_parameters"]
            for config in (v100, a10040, a10080, h200, b300)
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(counts), len(set(counts)))
        self.assertEqual(
            counts,
            [29_144_367, 58_412_431, 230_537_295, 501_835_855, 878_114_575],
        )

    def test_one_to_eight_homogeneous_gpus_are_supported(self):
        for count in (1, 2, 4, 8):
            resolved = self.resolve("NVIDIA H100 80GB HBM3", 81920, count)
            self.assertEqual(resolved["hardware"]["gpu_count"], count)
            self.assertEqual(
                resolved["hardware"]["distributed_backend"],
                "none" if count == 1 else "nccl",
            )
        with self.assertRaisesRegex(ValueError, "1..8"):
            self.resolve("NVIDIA H100 80GB HBM3", 81920, 9)

    def test_v100_uses_fp16_and_newer_profiles_use_bf16(self):
        self.assertEqual(
            self.resolve("Tesla V100-SXM2-16GB", 16384)["hardware"]["precision"],
            "float16",
        )
        self.assertEqual(
            self.resolve("NVIDIA A100-SXM4-80GB", 81920)["hardware"]["precision"],
            "bfloat16",
        )

    def test_trainer_contains_ddp_no_sync_and_fp16_scaling_contracts(self):
        source = (TOOLS / "train_hydra_oracle_v5_a100.py").read_text()
        self.assertIn("DistributedDataParallel", source)
        self.assertIn("dist.init_process_group", source)
        self.assertIn(".no_sync()", source)
        self.assertIn("GradScaler", source)
        launcher = (
            TOOLS.parent / "scripts/training/a100_hydra_v5_train.sh"
        ).read_text()
        self.assertIn("torchrun --standalone", launcher)
        self.assertIn('nproc_per_node="$GPU_COUNT"', launcher)

    def test_mixed_gpu_nodes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            MODULE.select_profile(
                PROFILES,
                [
                    {"index": 0, "name": "NVIDIA H100", "memory_total_mib": 81920},
                    {"index": 1, "name": "NVIDIA A100", "memory_total_mib": 81920},
                ],
            )


if __name__ == "__main__":
    unittest.main()
