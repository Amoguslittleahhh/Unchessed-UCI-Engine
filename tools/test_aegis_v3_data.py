#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("aegis_v3_data.py")
SPEC = importlib.util.spec_from_file_location("aegis_v3_data", MODULE_PATH)
DATA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATA
SPEC.loader.exec_module(DATA)


def sample(game=11, player=101, promotion=0, history=(1, 2, 3)):
    flags = 0
    if promotion:
        flags |= DATA.FLAG_PROMOTION
    if history:
        flags |= DATA.FLAG_HISTORY
    padded_history = tuple(history) + (0,) * (DATA.HISTORY_PLIES - len(history))
    return DATA.AegisV3Record(
        bitboards=(0x10, 0x42, 0x24, 0x81, 0x08, 0x10) * 2,
        move=12 | (28 << 6),
        promotion=promotion,
        wdl=2,
        rating=1800,
        castling=0b1111,
        ep_file=DATA.UNKNOWN_EP,
        halfmove=4,
        time_class=DATA.TIME_RAPID,
        flags=flags,
        history_len=len(history),
        history=padded_history,
        game_hash=game,
        player_hash=player,
        ply=17,
    )


class AegisV3DataTests(unittest.TestCase):
    def test_frozen_widths_and_schema_identity(self):
        self.assertEqual(DATA.HEADER.size, 64)
        self.assertEqual(DATA.RECORD.size, 160)
        self.assertEqual(len(DATA.SCHEMA_SHA256), 32)
        self.assertEqual(DATA.parse_header(DATA.make_header(37)), 37)
        config = json.loads(
            (MODULE_PATH.parent.parent / "config/unchessed_hydra_v3.json").read_text()
        )
        self.assertEqual(config["data_magic"], DATA.MAGIC.decode("ascii"))
        self.assertEqual(config["data"]["header_bytes"], DATA.HEADER.size)
        self.assertEqual(config["data"]["record_bytes"], DATA.RECORD.size)

    def test_record_and_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.aegis3"
            records = [sample(), sample(game=12, player=102, history=())]
            self.assertEqual(DATA.write_shard(path, records), 2)
            self.assertEqual(DATA.shard_record_count(path), 2)
            self.assertEqual(list(DATA.iter_shard(path)), records)
            report = DATA.validate_shard(path)
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["games"], 2)
            self.assertEqual(report["players"], 2)

    def test_promotion_identity_is_not_collapsed(self):
        knight = sample(promotion=1)
        queen = sample(promotion=4)
        self.assertNotEqual(knight.pack(), queen.pack())
        self.assertEqual(DATA.AegisV3Record.unpack(knight.pack()).promotion, 1)
        bad = sample(promotion=0)
        bad = DATA.dataclasses.replace(bad, flags=bad.flags | DATA.FLAG_PROMOTION)
        with self.assertRaisesRegex(DATA.FormatError, "promotion"):
            bad.pack()

    def test_header_and_length_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.aegis3"
            DATA.write_shard(path, [sample()])
            payload = bytearray(path.read_bytes())
            payload[60] ^= 0x80
            path.write_bytes(payload)
            with self.assertRaisesRegex(DATA.FormatError, "CRC32"):
                DATA.shard_record_count(path)
            path.write_bytes(DATA.make_header(1) + sample().pack() + b"x")
            with self.assertRaisesRegex(DATA.FormatError, "file length"):
                DATA.shard_record_count(path)

    def test_split_audit_rejects_game_or_player_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.aegis3"
            valid = Path(directory) / "valid.aegis3"
            DATA.write_shard(train, [sample(game=11, player=101)])
            DATA.write_shard(valid, [sample(game=12, player=102)])
            report = DATA.audit_disjoint_splits([train], [valid])
            self.assertTrue(report["disjoint"])
            DATA.write_shard(valid, [sample(game=12, player=101)])
            with self.assertRaisesRegex(DATA.FormatError, "split leakage"):
                DATA.audit_disjoint_splits([train], [valid])

    def test_teacher_and_clock_presence_are_explicit(self):
        base = sample(history=())
        teacher = DATA.dataclasses.replace(
            base,
            flags=base.flags | DATA.FLAG_TEACHER,
            teacher_score=80,
            best_move=123,
            best_score=100,
            move_score=40,
        )
        self.assertEqual(teacher.regret_cp, 60)
        self.assertEqual(DATA.AegisV3Record.unpack(teacher.pack()), teacher)
        bad_clock = DATA.dataclasses.replace(base, remaining_ms=5000)
        with self.assertRaisesRegex(DATA.FormatError, "clock"):
            bad_clock.pack()


if __name__ == "__main__":
    unittest.main()
