#!/usr/bin/env python3

import dataclasses
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "aegis_v4_data.py"
SPEC = importlib.util.spec_from_file_location("aegis_v4_data", MODULE_PATH)
DATA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATA
SPEC.loader.exec_module(DATA)
V3 = sys.modules["aegis_v3_data"]


def padded(values, fill):
    return tuple(values) + (fill,) * (DATA.MAX_LEGAL_ACTIONS - len(values))


def base_record(game=11, player=101, teacher=False):
    flags = 0
    kwargs = {}
    if teacher:
        flags |= V3.FLAG_TEACHER
        best_move = 12 | (28 << 6)
        kwargs = dict(teacher_score=80, best_move=best_move, best_score=100, move_score=40)
    return V3.AegisV3Record(
        bitboards=(0x10, 0x42, 0x24, 0x81, 0x08, 0x10) * 2,
        move=12 | (28 << 6),
        promotion=0,
        wdl=2,
        rating=1800,
        castling=15,
        ep_file=V3.UNKNOWN_EP,
        halfmove=4,
        time_class=V3.TIME_RAPID,
        flags=flags,
        history_len=0,
        history=(0,) * 8,
        game_hash=game,
        player_hash=player,
        ply=17,
        **kwargs,
    )


def human_record(game=11, player=101):
    base = base_record(game, player)
    target = DATA.encode_action(base.move, base.promotion)
    actions = sorted((target, 1 | (18 << 6), 6 | (21 << 6)))
    return DATA.AegisV4Record(
        base=base,
        legal_count=len(actions),
        target_action=target,
        teacher_best_action=DATA.ACTION_SENTINEL,
        policy_kind=DATA.POLICY_HUMAN,
        legal_flags=0,
        legal_actions=padded(actions, DATA.ACTION_SENTINEL),
        legal_regrets=padded((DATA.REGRET_SENTINEL,) * len(actions), DATA.REGRET_SENTINEL),
    )


def guide_record(game=11, player=101):
    base = base_record(game, player, teacher=True)
    target = DATA.encode_action(base.move, 0)
    actions = sorted((target, 1 | (18 << 6), 6 | (21 << 6)))
    regrets_by_action = {target: 0, actions[0]: 0 if actions[0] == target else 55, actions[1]: 0 if actions[1] == target else 120, actions[2]: 0 if actions[2] == target else 90}
    regrets = tuple(regrets_by_action[action] for action in actions)
    return DATA.AegisV4Record(
        base=base,
        legal_count=len(actions),
        target_action=target,
        teacher_best_action=target,
        policy_kind=DATA.POLICY_GUIDE,
        legal_flags=DATA.LEGAL_FLAG_REGRETS,
        legal_actions=padded(actions, DATA.ACTION_SENTINEL),
        legal_regrets=padded(regrets, DATA.REGRET_SENTINEL),
    )


class AegisV4DataTests(unittest.TestCase):
    def test_frozen_legal_set_abi(self):
        self.assertEqual(DATA.HEADER.size, 64)
        self.assertEqual(DATA.V3_RECORD.size, 160)
        self.assertEqual(DATA.TAIL.size, 928)
        self.assertEqual(DATA.RECORD_BYTES, 1088)
        self.assertEqual(DATA.parse_header(DATA.make_header(19)), 19)

    def test_human_and_guide_round_trip(self):
        for record in (human_record(), guide_record()):
            payload = record.pack()
            self.assertEqual(len(payload), DATA.RECORD_BYTES)
            self.assertEqual(DATA.AegisV4Record.unpack(payload), record)

    def test_underpromotions_have_distinct_actions(self):
        move = 48 | (56 << 6)
        actions = [DATA.encode_action(move, promotion) for promotion in range(1, 5)]
        self.assertEqual(len(set(actions)), 4)
        self.assertTrue(all(0 <= action < DATA.ACTION_VOCABULARY for action in actions))
        self.assertEqual(DATA.mirror_action(actions[0]) >> 12, 1)

    def test_rejects_illegal_target_duplicates_and_bad_regret(self):
        record = human_record()
        bad = dataclasses.replace(record, target_action=123)
        with self.assertRaisesRegex(DATA.FormatError, "selected action"):
            bad.pack()
        active = list(record.legal_actions[: record.legal_count])
        active[1] = active[0]
        bad = dataclasses.replace(
            record,
            legal_actions=padded(active, DATA.ACTION_SENTINEL),
        )
        with self.assertRaisesRegex(DATA.FormatError, "strictly increasing"):
            bad.pack()
        guide = guide_record()
        regrets = list(guide.legal_regrets)
        regrets[guide.legal_actions.index(guide.teacher_best_action)] = 10
        bad = dataclasses.replace(guide, legal_regrets=tuple(regrets))
        with self.assertRaisesRegex(DATA.FormatError, "zero regret"):
            bad.pack()

    def test_atomic_shard_and_split_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.aegis4"
            valid = Path(directory) / "valid.aegis4"
            DATA.write_shard(train, [human_record(11, 101), guide_record(12, 102)])
            DATA.write_shard(valid, [human_record(21, 201)])
            self.assertEqual(DATA.shard_record_count(train), 2)
            self.assertEqual(len(list(DATA.iter_shard(train))), 2)
            report = DATA.validate_shard(train)
            self.assertEqual(report["regret_labelled"], 1)
            self.assertTrue(DATA.audit_disjoint_splits([train], [valid])["disjoint"])
            DATA.write_shard(valid, [human_record(21, 101)])
            with self.assertRaisesRegex(DATA.FormatError, "split leakage"):
                DATA.audit_disjoint_splits([train], [valid])

    def test_header_crc_and_physical_length_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.aegis4"
            DATA.write_shard(path, [human_record()])
            payload = bytearray(path.read_bytes())
            payload[60] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(DATA.FormatError, "CRC32"):
                DATA.shard_record_count(path)
            path.write_bytes(DATA.make_header(1) + human_record().pack() + b"x")
            with self.assertRaisesRegex(DATA.FormatError, "file length"):
                DATA.shard_record_count(path)


if __name__ == "__main__":
    unittest.main()
