#!/usr/bin/env python3
"""Focused contract tests for ``analyse_adapter_telemetry``."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_adapter_telemetry import TelemetryError, analyse, parse_telemetry_text


OPTIONS = "adaptive=1 limit_strength=0 persona_smooth=0 engine_detect_v2=0 own_book=0 adapter_telemetry=1"
OBS = (
    "info string [UnchessedTelemetry] v=1 event=opponent_observation run=run-a "
    "game={game} ply={ply} observation={observation} source=probe {options} "
    "low_time=0 clock_available=0 opp_time_used_ms=none cp_loss=14 "
    "difficulty_weight_milli=800 legal_count=31 had_choice=1 estimate_elo=2187 "
    "confidence_cp=244 weight_milli=7410 suspicion_milli=0 low_loss_streak=2 "
    "samples=9 is_computer=0 declared_elo=none suspect={suspect} "
    "suspect_reason=none accelerated_score_milli=0 accelerated_evidence_milli=0 accelerated_streak=0 action_full=0"
)
DECISION = (
    "info string [UnchessedTelemetry] v=1 event=persona_decision run=run-a "
    "game={game} ply={ply} decision={decision} raw_eval_cp=-87 ema_cp=-53 "
    "mode_before=MATCH mode_after={mode_after} candidate=MATCH dwell=0 emergency=none "
    "{options} suspect=0 action_full=0 selected_move=g1f3"
)


def manifest_row(game: int, expected: bool = False) -> dict:
    return {
        "run": "run-a",
        "game": game,
        "arm": "baseline",
        "expected_suspect": expected,
        "options": {
            "Adaptive": True,
            "UCI_LimitStrength": False,
            "PersonaSmooth": False,
            "EngineDetectV2": False,
            "OwnBook": False,
        },
    }


class AdapterTelemetryParserTests(unittest.TestCase):
    def write_capture(self, telemetry: str, manifest: list[dict]) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        telemetry_path = root / "telemetry.log"
        manifest_path = root / "manifest.jsonl"
        telemetry_path.write_text(telemetry, encoding="utf-8")
        manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
        return telemetry_path, manifest_path, tmp

    def test_valid_capture_reports_persona_and_detection_counts(self):
        text = "\n".join(
            [
                "cutechess prefix ignored",
                OBS.format(game=1, ply=23, observation=1, options=OPTIONS, suspect=0),
                DECISION.format(game=1, ply=24, decision=1, mode_after="MATCH", options=OPTIONS),
                DECISION.format(game=1, ply=26, decision=2, mode_after="PUNISH", options=OPTIONS),
            ]
        )
        telemetry, manifest, tmp = self.write_capture(text, [manifest_row(1)])
        self.addCleanup(tmp.cleanup)
        report, games = analyse(telemetry, manifest)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["detection"], {"TP": 0, "FP": 0, "TN": 1, "FN": 0,
                                               "precision": None, "recall": None, "specificity": 1.0,
                                               "balanced_accuracy": None, "false_positive_rate": 0.0})
        self.assertEqual(report["persona"]["pooled_flips"], 1)
        self.assertEqual(report["persona"]["pooled_flip_rate"], 1.0)
        self.assertEqual(games[0]["persona_flip_rate"], 1.0)

    def test_old_schema_v1_capture_defaults_new_fusion_diagnostics(self):
        legacy = OBS.format(game=3, ply=23, observation=1, options=OPTIONS, suspect=0)
        legacy = legacy.replace(
            " accelerated_score_milli=0 accelerated_evidence_milli=0 accelerated_streak=0", ""
        )
        records = parse_telemetry_text(legacy)
        self.assertEqual(records[0]["accelerated_score_milli"], 0)
        self.assertEqual(records[0]["accelerated_evidence_milli"], 0)
        self.assertEqual(records[0]["accelerated_streak"], 0)

    def test_resilient_diagnostics_and_reason_parse(self):
        resilient = OBS.format(game=4, ply=31, observation=3, options=OPTIONS, suspect=1)
        resilient = resilient.replace(
            "suspect_reason=none accelerated_score_milli=0 accelerated_evidence_milli=0 accelerated_streak=0 action_full=0",
            "suspect_reason=legacy_accelerated_resilient accelerated_score_milli=0 accelerated_evidence_milli=0 accelerated_streak=0 accelerated_resilient_score_milli=712 accelerated_resilient_evidence_milli=533 accelerated_resilient_streak=2 action_full=1",
        )
        records = parse_telemetry_text(resilient)
        self.assertEqual(records[0]["suspect_reason"], "legacy_accelerated_resilient")
        self.assertEqual(records[0]["accelerated_resilient_score_milli"], 712)
        self.assertEqual(records[0]["accelerated_resilient_evidence_milli"], 533)
        self.assertEqual(records[0]["accelerated_resilient_streak"], 2)

    def test_malformed_unknown_schema_duplicate_field_and_duplicate_index_rejected(self):
        bad_schema = OBS.format(game=1, ply=23, observation=1, options=OPTIONS, suspect=0).replace("v=1", "v=2")
        with self.assertRaisesRegex(TelemetryError, "unknown schema version"):
            parse_telemetry_text(bad_schema)
        duplicate_field = OBS.format(game=1, ply=23, observation=1, options=OPTIONS, suspect=0) + " suspect=0"
        with self.assertRaisesRegex(TelemetryError, "duplicate field"):
            parse_telemetry_text(duplicate_field)
        repeated = "\n".join([
            OBS.format(game=1, ply=23, observation=1, options=OPTIONS, suspect=0),
            OBS.format(game=1, ply=25, observation=1, options=OPTIONS, suspect=0),
        ])
        with self.assertRaisesRegex(TelemetryError, "duplicate record key"):
            parse_telemetry_text(repeated)

    def test_interleaved_games_are_joined_by_run_and_game(self):
        text = "\n".join([
            OBS.format(game=1, ply=23, observation=1, options=OPTIONS, suspect=0),
            OBS.format(game=2, ply=23, observation=1, options=OPTIONS, suspect=0),
            DECISION.format(game=1, ply=24, decision=1, mode_after="MATCH", options=OPTIONS),
            DECISION.format(game=2, ply=24, decision=1, mode_after="MATCH", options=OPTIONS),
        ])
        telemetry, manifest, tmp = self.write_capture(text, [manifest_row(1), manifest_row(2)])
        self.addCleanup(tmp.cleanup)
        report, games = analyse(telemetry, manifest)
        self.assertEqual(report["game_count"], 2)
        self.assertEqual([game["game"] for game in games], [1, 2])
        self.assertEqual(report["detection"]["TN"], 2)

    def test_missing_label_and_midgame_option_change_rejected(self):
        text = OBS.format(game=9, ply=23, observation=1, options=OPTIONS, suspect=0)
        telemetry, manifest, tmp = self.write_capture(text, [])
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(TelemetryError, "missing manifest label"):
            analyse(telemetry, manifest)
        altered = OPTIONS.replace("persona_smooth=0", "persona_smooth=1")
        text = "\n".join([
            OBS.format(game=1, ply=23, observation=1, options=OPTIONS, suspect=0),
            DECISION.format(game=1, ply=24, decision=1, mode_after="MATCH", options=altered),
        ])
        telemetry, manifest, tmp = self.write_capture(text, [manifest_row(1)])
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(TelemetryError, "does not match manifest|option state changed"):
            analyse(telemetry, manifest)


if __name__ == "__main__":
    unittest.main()
