#!/usr/bin/env python3
"""Validate and summarize default-off AdapterTelemetry records.

The parser consumes arbitrary engine/cutechess text and extracts only lines
prefixed by ``info string [UnchessedTelemetry]``. Schema v1 is intentionally
flat, whitespace-delimited key=value data. It validates every field before
joining records to a runner-owned JSONL manifest, so it is suitable for a
measurement capture but makes no behavioural or strength claim itself.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PREFIX = "info string [UnchessedTelemetry] "
SCHEMA_VERSION = "1"
IDENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$|^0000$")
MODE_VALUES = {"FULL", "MATCH", "PUNISH", "CLINCH", "DEFEND"}
EMERGENCY_VALUES = {"none", "full", "defend", "punish"}
SUSPECT_REASON_VALUES = {
    "none",
    "legacy_computer",
    "legacy_clock",
    "legacy_ceiling",
    "v2_computer_threshold",
    "v2_clock_threshold",
    "v2_declared_exempt",
    "v2_anonymous_ceiling",
}
EVENTS = {"opponent_observation", "observation_skipped", "persona_decision"}
OPTION_FIELDS = (
    "adaptive",
    "limit_strength",
    "persona_smooth",
    "engine_detect_v2",
    "own_book",
    "adapter_telemetry",
)
COMMON_FIELDS = {"v", "event", "run", "game", "ply", *OPTION_FIELDS}
OBSERVATION_REQUIRED = COMMON_FIELDS | {
    "observation", "source", "low_time", "clock_available", "opp_time_used_ms",
    "cp_loss", "difficulty_weight_milli", "legal_count", "had_choice", "estimate_elo",
    "confidence_cp", "weight_milli", "suspicion_milli", "low_loss_streak", "samples",
    "is_computer", "declared_elo", "suspect", "suspect_reason", "action_full",
}
DECISION_REQUIRED = COMMON_FIELDS | {
    "decision", "raw_eval_cp", "ema_cp", "mode_before", "mode_after", "candidate",
    "dwell", "emergency", "suspect", "action_full", "selected_move",
}
OBSERVATION_ALLOWED = OBSERVATION_REQUIRED | {"reason"}
DECISION_ALLOWED = DECISION_REQUIRED


class TelemetryError(ValueError):
    """An input that cannot be treated as a valid telemetry capture."""


def fail(origin: str, line_no: int, message: str) -> TelemetryError:
    return TelemetryError(f"{origin}:{line_no}: {message}")


def parse_flat_record(payload: str, origin: str, line_no: int) -> dict[str, str]:
    if not payload or payload != payload.strip():
        raise fail(origin, line_no, "empty or surrounding-whitespace telemetry payload")
    fields: dict[str, str] = {}
    for token in payload.split(" "):
        if not token or token.count("=") != 1:
            raise fail(origin, line_no, f"malformed field {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or not IDENT_RE.fullmatch(key) or any(c.isspace() for c in value):
            raise fail(origin, line_no, f"malformed field {token!r}")
        if key in fields:
            raise fail(origin, line_no, f"duplicate field {key!r}")
        fields[key] = value
    return fields


def uint(value: str, field: str, origin: str, line_no: int) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise fail(origin, line_no, f"{field} must be an unsigned decimal integer")
    return int(value)


def sint(value: str, field: str, origin: str, line_no: int) -> int:
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        raise fail(origin, line_no, f"{field} must be a decimal integer")
    return int(value)


def bit(value: str, field: str, origin: str, line_no: int) -> bool:
    if value not in {"0", "1"}:
        raise fail(origin, line_no, f"{field} must be 0 or 1")
    return value == "1"


def optional_uint(value: str, field: str, origin: str, line_no: int) -> int | None:
    return None if value == "none" else uint(value, field, origin, line_no)


def optional_sint(value: str, field: str, origin: str, line_no: int) -> int | None:
    return None if value == "none" else sint(value, field, origin, line_no)


def check_exact_fields(
    fields: dict[str, str], required: set[str], allowed: set[str], origin: str, line_no: int
) -> None:
    missing = sorted(required - fields.keys())
    unknown = sorted(fields.keys() - allowed)
    if missing:
        raise fail(origin, line_no, f"missing required field(s): {', '.join(missing)}")
    if unknown:
        raise fail(origin, line_no, f"unknown field(s): {', '.join(unknown)}")


def validate_record(fields: dict[str, str], origin: str, line_no: int) -> dict[str, Any]:
    if fields.get("v") != SCHEMA_VERSION:
        raise fail(origin, line_no, f"unknown schema version {fields.get('v')!r}")
    event = fields.get("event")
    if event not in EVENTS:
        raise fail(origin, line_no, f"unknown event {event!r}")
    if event == "persona_decision":
        check_exact_fields(fields, DECISION_REQUIRED, DECISION_ALLOWED, origin, line_no)
    else:
        check_exact_fields(fields, OBSERVATION_REQUIRED, OBSERVATION_ALLOWED, origin, line_no)
    for name in ("run",):
        if not IDENT_RE.fullmatch(fields[name]):
            raise fail(origin, line_no, f"{name} must use [A-Za-z0-9_.-]+")
    result: dict[str, Any] = {
        "event": event,
        "run": fields["run"],
        "game": uint(fields["game"], "game", origin, line_no),
        "ply": uint(fields["ply"], "ply", origin, line_no),
        "options": {name: bit(fields[name], name, origin, line_no) for name in OPTION_FIELDS},
        "line": line_no,
    }
    if not result["options"]["adapter_telemetry"]:
        raise fail(origin, line_no, "adapter_telemetry must be 1 for emitted records")
    if event == "persona_decision":
        result.update(
            decision=uint(fields["decision"], "decision", origin, line_no),
            raw_eval_cp=sint(fields["raw_eval_cp"], "raw_eval_cp", origin, line_no),
            ema_cp=sint(fields["ema_cp"], "ema_cp", origin, line_no),
            mode_before=fields["mode_before"],
            mode_after=fields["mode_after"],
            candidate=fields["candidate"],
            dwell=uint(fields["dwell"], "dwell", origin, line_no),
            emergency=fields["emergency"],
            suspect=bit(fields["suspect"], "suspect", origin, line_no),
            action_full=bit(fields["action_full"], "action_full", origin, line_no),
            selected_move=fields["selected_move"],
        )
        for name in ("mode_before", "mode_after", "candidate"):
            if result[name] not in MODE_VALUES:
                raise fail(origin, line_no, f"{name} has invalid mode {result[name]!r}")
        if result["emergency"] not in EMERGENCY_VALUES:
            raise fail(origin, line_no, "emergency has invalid value")
        if not MOVE_RE.fullmatch(result["selected_move"]):
            raise fail(origin, line_no, "selected_move has invalid UCI syntax")
    else:
        source = fields["source"]
        if source not in {"probe", "book"}:
            raise fail(origin, line_no, f"source has invalid value {source!r}")
        result.update(
            observation=uint(fields["observation"], "observation", origin, line_no),
            source=source,
            reason=fields.get("reason"),
            low_time=bit(fields["low_time"], "low_time", origin, line_no),
            clock_available=bit(fields["clock_available"], "clock_available", origin, line_no),
            opp_time_used_ms=optional_uint(fields["opp_time_used_ms"], "opp_time_used_ms", origin, line_no),
            cp_loss=optional_sint(fields["cp_loss"], "cp_loss", origin, line_no),
            difficulty_weight_milli=optional_sint(fields["difficulty_weight_milli"], "difficulty_weight_milli", origin, line_no),
            legal_count=optional_uint(fields["legal_count"], "legal_count", origin, line_no),
            had_choice=None if fields["had_choice"] == "none" else bit(fields["had_choice"], "had_choice", origin, line_no),
            estimate_elo=sint(fields["estimate_elo"], "estimate_elo", origin, line_no),
            confidence_cp=sint(fields["confidence_cp"], "confidence_cp", origin, line_no),
            weight_milli=sint(fields["weight_milli"], "weight_milli", origin, line_no),
            suspicion_milli=sint(fields["suspicion_milli"], "suspicion_milli", origin, line_no),
            low_loss_streak=uint(fields["low_loss_streak"], "low_loss_streak", origin, line_no),
            samples=uint(fields["samples"], "samples", origin, line_no),
            is_computer=bit(fields["is_computer"], "is_computer", origin, line_no),
            declared_elo=optional_sint(fields["declared_elo"], "declared_elo", origin, line_no),
            suspect=bit(fields["suspect"], "suspect", origin, line_no),
            suspect_reason=fields["suspect_reason"],
            action_full=bit(fields["action_full"], "action_full", origin, line_no),
        )
        if result["suspect_reason"] not in SUSPECT_REASON_VALUES:
            raise fail(origin, line_no, "suspect_reason has invalid value")
        if result["clock_available"] != (result["opp_time_used_ms"] is not None):
            raise fail(origin, line_no, "clock_available must match opp_time_used_ms")
        if event == "opponent_observation" and result["source"] == "probe":
            for name in ("cp_loss", "difficulty_weight_milli", "legal_count", "had_choice"):
                if result[name] is None:
                    raise fail(origin, line_no, f"{name} is required for probe observation")
        if event == "observation_skipped":
            if result["source"] != "probe" or not result["reason"]:
                raise fail(origin, line_no, "skipped observation requires probe source and reason")
    return result


def parse_telemetry_text(text: str, origin: str = "telemetry") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, int]] = set()
    for line_no, line in enumerate(text.splitlines(), 1):
        if PREFIX not in line:
            continue
        payload = line[line.index(PREFIX) + len(PREFIX):]
        record = validate_record(parse_flat_record(payload, origin, line_no), origin, line_no)
        index_name = "decision" if record["event"] == "persona_decision" else "observation"
        key = (record["run"], record["game"], record["event"], record[index_name])
        if key in seen:
            raise fail(origin, line_no, f"duplicate record key {key}")
        seen.add(key)
        records.append(record)
    return records


def parse_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TelemetryError(f"{path}: cannot read manifest: {error}") from error
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip():
            raise fail(str(path), line_no, "blank manifest line")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise fail(str(path), line_no, f"invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise fail(str(path), line_no, "manifest row must be an object")
        required = {"run", "game", "expected_suspect", "options"}
        missing = sorted(required - row.keys())
        if missing:
            raise fail(str(path), line_no, f"missing manifest field(s): {', '.join(missing)}")
        if not isinstance(row["run"], str) or not IDENT_RE.fullmatch(row["run"]):
            raise fail(str(path), line_no, "manifest run must use [A-Za-z0-9_.-]+")
        if isinstance(row["game"], bool) or not isinstance(row["game"], int) or row["game"] < 0:
            raise fail(str(path), line_no, "manifest game must be a non-negative integer")
        if not isinstance(row["expected_suspect"], bool):
            raise fail(str(path), line_no, "manifest expected_suspect must be boolean")
        if not isinstance(row["options"], dict):
            raise fail(str(path), line_no, "manifest options must be an object")
        unknown = sorted(set(row["options"]) - {"Adaptive", "UCI_LimitStrength", "PersonaSmooth", "EngineDetectV2", "OwnBook"})
        if unknown:
            raise fail(str(path), line_no, f"unknown manifest option(s): {', '.join(unknown)}")
        key = (row["run"], row["game"])
        if key in entries:
            raise fail(str(path), line_no, f"duplicate manifest game {key}")
        entries[key] = row
    return entries


def expected_options(manifest_options: dict[str, Any], record: dict[str, Any], origin: str) -> None:
    mapping = {
        "Adaptive": "adaptive",
        "UCI_LimitStrength": "limit_strength",
        "PersonaSmooth": "persona_smooth",
        "EngineDetectV2": "engine_detect_v2",
        "OwnBook": "own_book",
    }
    for manifest_name, telemetry_name in mapping.items():
        if manifest_name in manifest_options:
            expected = manifest_options[manifest_name]
            if not isinstance(expected, bool):
                raise TelemetryError(f"{origin}: manifest option {manifest_name} must be boolean")
            if record["options"][telemetry_name] != expected:
                raise TelemetryError(
                    f"{origin}: telemetry {telemetry_name} does not match manifest {manifest_name}"
                )


def summarize(records: list[dict[str, Any]], manifest: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    games: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["run"], record["game"])
        if key not in manifest:
            raise TelemetryError(f"telemetry:{record['line']}: missing manifest label for {key}")
        expected_options(manifest[key]["options"], record, f"telemetry:{record['line']}")
        games[key].append(record)
    report_games: list[dict[str, Any]] = []
    detection = Counter()
    total_by_event = Counter(record["event"] for record in records)
    coverage = Counter()
    for key, game_records in sorted(games.items()):
        option_states = {tuple(sorted(record["options"].items())) for record in game_records}
        if len(option_states) != 1:
            raise TelemetryError(f"game {key}: option state changed mid-game")
        decisions = sorted((r for r in game_records if r["event"] == "persona_decision"), key=lambda r: r["decision"])
        observations = [r for r in game_records if r["event"] == "opponent_observation"]
        skipped = [r for r in game_records if r["event"] == "observation_skipped"]
        flips = sum(a["mode_after"] != b["mode_after"] for a, b in zip(decisions, decisions[1:]))
        eligible = len(decisions)
        expected = manifest[key]["expected_suspect"]
        for record in observations:
            predicted = record["suspect"]
            detection[("TP" if predicted else "FN") if expected else ("FP" if predicted else "TN")] += 1
            coverage["clock_available"] += int(record["clock_available"])
            coverage["had_choice"] += int(record["had_choice"] is True)
        coverage["observations"] += len(observations)
        coverage["skipped"] += len(skipped)
        report_games.append({
            "run": key[0], "game": key[1], "expected_suspect": expected,
            "arm": manifest[key].get("arm"), "options": dict(option_states.pop()),
            "decision_count": eligible, "persona_flips": flips,
            "persona_flip_rate": flips / (eligible - 1) if eligible >= 2 else None,
            "observation_count": len(observations), "skipped_observation_count": len(skipped),
        })
    tp, fp, tn, fn = (detection[name] for name in ("TP", "FP", "TN", "FN"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    pooled_denominator = sum(max(g["decision_count"] - 1, 0) for g in report_games)
    report = {
        "schema_version": 1,
        "record_count": len(records),
        "event_counts": dict(sorted(total_by_event.items())),
        "game_count": len(report_games),
        "coverage": dict(coverage),
        "persona": {
            "pooled_flips": sum(g["persona_flips"] for g in report_games),
            "pooled_transition_opportunities": pooled_denominator,
            "pooled_flip_rate": sum(g["persona_flips"] for g in report_games) / pooled_denominator if pooled_denominator else None,
        },
        "detection": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": precision, "recall": recall, "specificity": specificity,
            "balanced_accuracy": balanced, "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        },
    }
    return report, report_games


def analyse(telemetry_path: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        text = telemetry_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TelemetryError(f"{telemetry_path}: cannot read telemetry: {error}") from error
    return summarize(parse_telemetry_text(text, str(telemetry_path)), parse_manifest(manifest_path))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path, help="engine/cutechess stdout or debug log")
    parser.add_argument("manifest", type=Path, help="runner-owned JSONL labels")
    parser.add_argument("--per-game-jsonl", type=Path, help="optional per-game output path")
    args = parser.parse_args(argv)
    try:
        report, games = analyse(args.telemetry, args.manifest)
    except TelemetryError as error:
        print(f"analyse_adapter_telemetry: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    if args.per_game_jsonl:
        args.per_game_jsonl.write_text("".join(json.dumps(game, sort_keys=True) + "\n" for game in games), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
