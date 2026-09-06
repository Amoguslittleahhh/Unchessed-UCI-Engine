#!/usr/bin/env python3

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

CANONICAL = (
    "unarchitectured_metal_base_data.py",
    "unarchitectured_metal_data.py",
    "unarchitectured_metal_uci_teacher_worker.py",
    "train_unarchitectured_metal_student_a100.py",
    "train_unarchitectured_metal_a100.py",
    "calibrate_unarchitectured_metal_throughput.py",
    "reference_forward_unarchitectured_metal.py",
    "smoke_unarchitectured_metal_uci.py",
    "unarchitectured_metal_runtime_readiness.py",
)
LEGACY = (
    "aegis_v3_data.py",
    "aegis_v4_data.py",
    "v5_uci_teacher_worker.py",
    "train_chessformer_v4_a100.py",
    "train_hydra_oracle_v5_a100.py",
    "reference_forward_aegis_v4.py",
    "v5_runtime_readiness.py",
)


class UnarchitecturedV1ScriptNamingTests(unittest.TestCase):
    def test_canonical_entrypoints_replace_predecessor_names(self):
        for name in CANONICAL:
            self.assertTrue((TOOLS / name).is_file(), name)
        for name in LEGACY:
            self.assertFalse((TOOLS / name).exists(), name)

    def test_current_scripts_parse_and_use_canonical_classes(self):
        student = ast.parse((TOOLS / "train_unarchitectured_metal_student_a100.py").read_text())
        oracle = ast.parse((TOOLS / "train_unarchitectured_metal_a100.py").read_text())
        student_classes = {node.name for node in ast.walk(student) if isinstance(node, ast.ClassDef)}
        oracle_classes = {node.name for node in ast.walk(oracle) if isinstance(node, ast.ClassDef)}
        self.assertIn("UnarchitecturedV1Student", student_classes)
        self.assertIn("UnarchitecturedV1Oracle", oracle_classes)

        combined = "\n".join((TOOLS / name).read_text() for name in CANONICAL)
        # Frozen wire descriptors intentionally retain predecessor-era bytes;
        # current module names, class names, defaults and diagnostics must not.
        for stale in (
            "AegisV4Chessformer",
            "HydraOracleV5",
            "config/a100_hydra_v4_training.json",
            "hydra-apex-v5.pt",
            "chessformer-v4.pt",
        ):
            self.assertNotIn(stale, combined)

    def test_canonical_data_record_round_trip(self):
        import unarchitectured_metal_base_data as base_data
        import unarchitectured_metal_data as data

        move = 8 | (16 << 6)
        base = base_data.UnarchitecturedV1BaseRecord(
            bitboards=(0,) * 12,
            move=move,
            promotion=0,
            wdl=1,
            rating=1800,
            castling=0,
            ep_file=base_data.UNKNOWN_EP,
            halfmove=0,
            time_class=base_data.TIME_RAPID,
            flags=0,
            history_len=0,
            history=(0,) * base_data.HISTORY_PLIES,
            game_hash=1,
            player_hash=2,
        )
        action = data.encode_action(move, 0)
        record = data.UnarchitecturedV1Record(
            base=base,
            legal_count=1,
            target_action=action,
            teacher_best_action=data.ACTION_SENTINEL,
            policy_kind=data.POLICY_HUMAN,
            legal_flags=0,
            legal_actions=(action,) + (data.ACTION_SENTINEL,) * (data.MAX_LEGAL_ACTIONS - 1),
            legal_regrets=(data.REGRET_SENTINEL,) * data.MAX_LEGAL_ACTIONS,
        )
        self.assertEqual(data.UnarchitecturedV1Record.unpack(record.pack()), record)

    def test_dependency_free_entrypoints_run_help(self):
        for name in (
            "unarchitectured_metal_base_data.py",
            "unarchitectured_metal_data.py",
            "unarchitectured_metal_uci_teacher_worker.py",
            "unarchitectured_metal_runtime_readiness.py",
            "train_unarchitectured_metal_student_a100.py",
            "train_unarchitectured_metal_a100.py",
            "calibrate_unarchitectured_metal_throughput.py",
            "reference_forward_unarchitectured_metal.py",
            "smoke_unarchitectured_metal_uci.py",
        ):
            result = subprocess.run(
                [sys.executable, str(TOOLS / name), "--help"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")
            self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
