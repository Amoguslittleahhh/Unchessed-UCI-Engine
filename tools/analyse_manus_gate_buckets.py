#!/usr/bin/env python3
"""Bucket a main-vs-heavyopt SPRT gate's per-engine telemetry and cutechess
termination log into the categories Manus asked for: persona mode
distribution, low-time decision rate, persona transition counts, and
illegal-move/crash/time-forfeit counts.

Telemetry input is a directory of per-process log files produced by the
tee-wrapper scripts (each cutechess engine slot's raw UCI stdout, since
this cutechess-cli build's -debug flag does not work -- see
scripts/research/wsl_sprt_main_vs_heavyopt_fast.sh). Only lines prefixed
``info string [UnchessedTelemetry] `` are read; anything else in the file
is UCI protocol noise and is ignored.

This is a measurement summary, not a strength or promotion verdict --
the SPRT result in the cutechess log remains the actual gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_adapter_telemetry import PREFIX  # noqa: E402

MODE_VALUES = ("FULL", "MATCH", "PUNISH", "CLINCH", "DEFEND")

# This tool intentionally does NOT reuse analyse_adapter_telemetry's
# validate_record(): that validator enforces main's exact schema and
# rejects any record with an unrecognised field. The heavy-optimisation
# binary's persona_decision telemetry carries an extra `cooldown` field
# main doesn't have (a real schema drift between the two binaries, not
# a bug in this tool) -- strict validation silently dropped nearly all
# of its decision records. Parsing here is deliberately schema-tolerant:
# it reads only the specific fields this analysis needs and ignores
# anything else, on either side.
REQUIRED_DECISION_FIELDS = ("event", "game", "decision", "mode_before", "mode_after", "action_full")
REQUIRED_OBSERVATION_FIELDS = ("event", "game", "observation", "low_time")


def parse_flat_record(payload: str) -> dict[str, str] | None:
    fields: dict[str, str] = {}
    for token in payload.split(" "):
        if not token or token.count("=") != 1:
            return None
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def load_side(telemetry_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = 0
    for path in sorted(telemetry_dir.glob("proc-*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if PREFIX not in line:
                continue
            payload = line[line.index(PREFIX) + len(PREFIX):]
            fields = parse_flat_record(payload)
            if fields is None or "event" not in fields:
                skipped += 1
                continue
            event = fields["event"]
            required = REQUIRED_DECISION_FIELDS if event == "persona_decision" else REQUIRED_OBSERVATION_FIELDS
            if any(name not in fields for name in required):
                skipped += 1
                continue
            record: dict[str, Any] = {"_proc": path.name, "event": event, "game": fields["game"]}
            if event == "persona_decision":
                record["decision"] = int(fields["decision"])
                record["mode_before"] = fields["mode_before"]
                record["mode_after"] = fields["mode_after"]
                record["action_full"] = fields["action_full"] == "1"
            else:
                record["observation"] = int(fields["observation"])
                record["low_time"] = fields["low_time"] == "1"
            records.append(record)
    if skipped:
        print(f"warning: {telemetry_dir}: skipped {skipped} unparseable/incomplete telemetry line(s)", file=sys.stderr)
    return records


def summarize_side(records: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [r for r in records if r["event"] == "persona_decision"]
    observations = [r for r in records if r["event"] == "opponent_observation"]
    skipped = [r for r in records if r["event"] == "observation_skipped"]

    mode_counts = Counter(r["mode_after"] for r in decisions)
    for mode in MODE_VALUES:
        mode_counts.setdefault(mode, 0)

    by_game: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in decisions:
        by_game[(r["_proc"], r["game"])].append(r)
    flips = 0
    opportunities = 0
    for game_decisions in by_game.values():
        ordered = sorted(game_decisions, key=lambda r: r["decision"])
        for a, b in zip(ordered, ordered[1:]):
            opportunities += 1
            if a["mode_after"] != b["mode_after"]:
                flips += 1

    low_time_observations = sum(1 for r in observations if r["low_time"])
    low_time_skipped = sum(1 for r in skipped if r["low_time"])
    action_full_decisions = sum(1 for r in decisions if r["action_full"])

    return {
        "decision_count": len(decisions),
        "observation_count": len(observations),
        "observation_skipped_count": len(skipped),
        "mode_after_distribution": dict(sorted(mode_counts.items())),
        "persona_transition_flips": flips,
        "persona_transition_opportunities": opportunities,
        "persona_transition_rate": flips / opportunities if opportunities else None,
        "low_time_observation_count": low_time_observations,
        "low_time_observation_rate": low_time_observations / len(observations) if observations else None,
        "low_time_skipped_observation_count": low_time_skipped,
        "action_full_decision_count": action_full_decisions,
        "action_full_decision_rate": action_full_decisions / len(decisions) if decisions else None,
    }


TERMINATION_CATEGORY_PATTERNS = (
    ("illegal_move", re.compile(r"illegal move", re.IGNORECASE)),
    ("crash_or_disconnect", re.compile(r"disconnect|crash|terminated unexpectedly", re.IGNORECASE)),
    ("time_forfeit", re.compile(r"on time|time forfeit|lost on time", re.IGNORECASE)),
    ("stalled_or_timeout", re.compile(r"stalled|unresponsive|timeout", re.IGNORECASE)),
    ("adjudication", re.compile(r"adjudicat", re.IGNORECASE)),
)


def categorize_reason(reason: str) -> str:
    for category, pattern in TERMINATION_CATEGORY_PATTERNS:
        if pattern.search(reason):
            return category
    return "other"


def parse_termination_reasons(cutechess_log: Path) -> dict[str, Any]:
    text = cutechess_log.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    by_player: dict[str, Counter] = defaultdict(Counter)
    current_player = None
    reason_re = re.compile(r'^\s*"([^"]+)":\s*(\d+)\s*$')
    for line in lines:
        m = re.match(r"^Player:\s*(\S+)\s*$", line)
        if m:
            current_player = m.group(1)
            continue
        m = reason_re.match(line)
        if m and current_player is not None:
            reason, count = m.group(1), int(m.group(2))
            by_player[current_player][categorize_reason(reason)] += count
    return {player: dict(sorted(counts.items())) for player, counts in by_player.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-telemetry-dir", type=Path, required=True)
    parser.add_argument("--heavyopt-telemetry-dir", type=Path, required=True)
    parser.add_argument("--cutechess-log", type=Path, required=True,
                         help="the SPRT run's stdout/stderr log, for the Player: ... termination-reason summary")
    args = parser.parse_args(argv)

    main_records = load_side(args.main_telemetry_dir)
    heavyopt_records = load_side(args.heavyopt_telemetry_dir)

    report = {
        "schema_version": 1,
        "note": "measurement summary only; the cutechess SPRT result in the log is the actual gate",
        "main": summarize_side(main_records),
        "heavyopt": summarize_side(heavyopt_records),
        "termination_reasons_by_engine_name": parse_termination_reasons(args.cutechess_log),
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
