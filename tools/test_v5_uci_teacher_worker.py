#!/usr/bin/env python3

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "v5_uci_teacher_worker.py"
SPEC = importlib.util.spec_from_file_location("v5_uci_teacher_worker", MODULE_PATH)
WORKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKER
SPEC.loader.exec_module(WORKER)
import aegis_v3_data as V3
import aegis_v4_data as V4


def padded(values, fill):
    return tuple(values) + (fill,) * (V4.MAX_LEGAL_ACTIONS - len(values))


def human_record():
    target = 12 | (28 << 6)
    actions = sorted((target, 1 | (18 << 6), 6 | (21 << 6)))
    base = V3.AegisV3Record(
        bitboards=(0xFF00, 0x42, 0x24, 0x81, 0x08, 0x10, 0x00FF000000000000, 0x4200000000000000, 0x2400000000000000, 0x8100000000000000, 0x0800000000000000, 0x1000000000000000),
        move=target,
        promotion=0,
        wdl=1,
        rating=1700,
        castling=15,
        ep_file=0xFF,
        halfmove=0,
        time_class=2,
        flags=0,
        history_len=0,
        history=(0,) * 8,
        game_hash=11,
        player_hash=12,
    )
    return V4.AegisV4Record(
        base=base,
        legal_count=len(actions),
        target_action=target,
        teacher_best_action=V4.ACTION_SENTINEL,
        policy_kind=V4.POLICY_HUMAN,
        legal_flags=0,
        legal_actions=padded(actions, V4.ACTION_SENTINEL),
        legal_regrets=padded((V4.REGRET_SENTINEL,) * len(actions), V4.REGRET_SENTINEL),
    )


class V5UciTeacherWorkerTests(unittest.TestCase):
    def test_action_and_fen_encoding(self):
        self.assertEqual(WORKER.action_to_uci(12 | (28 << 6)), "e2e4")
        promotion = 48 | (56 << 6) | (1 << 12)
        self.assertEqual(WORKER.action_to_uci(promotion), "a7a8n")
        fen = WORKER.record_to_fen(human_record())
        self.assertTrue(fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"))

    def test_score_parser_handles_cp_and_mate(self):
        self.assertEqual(WORKER.parse_score("info depth 4 score cp -37 nodes 10".split()), -37)
        self.assertEqual(WORKER.parse_score("info score mate 3".split()), WORKER.MATE_SCORE - 3)
        self.assertEqual(WORKER.parse_score("info score mate -2".split()), -WORKER.MATE_SCORE + 2)

    def test_fake_uci_engine_annotates_every_legal_action(self):
        script = """#!/usr/bin/env python3
import sys
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name FakeTeacher', flush=True); print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('go '):
        fields = line.split(); move = fields[fields.index('searchmoves') + 1]
        score = sum(ord(ch) for ch in move) % 200
        print(f'info depth 1 score cp {score} nodes 1', flush=True)
        print(f'bestmove {move}', flush=True)
    elif line == 'quit':
        break
"""
        with tempfile.TemporaryDirectory() as directory:
            engine_path = Path(directory) / "fake-engine.py"
            engine_path.write_text(script, encoding="utf-8")
            engine_path.chmod(0o755)
            with WORKER.UciEngine(
                [str(engine_path)],
                threads=1,
                hash_mb=1,
                timeout=5.0,
                clear_hash_per_action=True,
                options=[],
            ) as engine:
                annotated = WORKER.annotate_record(human_record(), engine, nodes=1)
        self.assertEqual(annotated.policy_kind, V4.POLICY_GUIDE)
        self.assertTrue(annotated.base.flags & V3.FLAG_TEACHER)
        self.assertTrue(annotated.legal_flags & V4.LEGAL_FLAG_REGRETS)
        active = annotated.legal_regrets[: annotated.legal_count]
        self.assertEqual(min(active), 0)
        self.assertEqual(active[annotated.legal_actions.index(annotated.teacher_best_action)], 0)
        annotated.validate()


if __name__ == "__main__":
    unittest.main()
