#!/usr/bin/env python3
"""Benchmark the timing ACF signal on real online-chess service exports.

Supported inputs:
- Lichess derived JSONL from timing_classifier_validation.py;
- Chess.com PGNs containing per-move ``[%clk ...]`` comments;
- FICS PGNs containing per-move ``[%emt ...]`` comments.

Only aggregate results are emitted. Raw usernames, game IDs, moves, and PGNs
are never written to the report. Chess.com accounts are unmarked because its
published archive has no affirmative computer-account field. FICS ``IsComp``
tags and Lichess ``BOT`` titles are retained as affirmative computer labels.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, Sequence

MODULE_PATH = Path(__file__).with_name("timing_classifier_validation.py")
SPEC = importlib.util.spec_from_file_location("timing_validation", MODULE_PATH)
TV = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TV
SPEC.loader.exec_module(TV)

EMT_RE = re.compile(r"\[%emt\s+(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)\]")
INTEGER_TC_RE = re.compile(r"\d+")
INCREMENT_TC_RE = re.compile(r"\d+\+\d+")
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_service_time_control(value: str) -> tuple[int, int] | None:
    if INTEGER_TC_RE.fullmatch(value):
        base, increment = int(value), 0
    elif INCREMENT_TC_RE.fullmatch(value):
        base, increment = (int(field) for field in value.split("+"))
    else:
        return None
    return (base, increment) if base > 0 and increment >= 0 else None


def elapsed_seconds(match: re.Match[str]) -> float:
    first, second, last = match.groups()
    if second is None:
        return int(first or 0) * 60 + float(last)
    return int(first or 0) * 3600 + int(second) * 60 + float(last)


def timing_score_from_clocks(
    clocks: Sequence,
    side: int,
    base: int,
    increment: int,
    skip_moves: int,
    window_size: int,
) -> tuple[float, int] | None:
    values: list[float] = []
    previous = float(base)
    for move_index, clock in enumerate(clocks[side::2]):
        used = previous + increment - clock.seconds
        previous = clock.seconds
        if move_index < skip_moves or used <= 0.0:
            continue
        before_move = clock.seconds + used
        if before_move > 0.0:
            fraction = min(1.0, max(1e-6, used / before_move))
            values.append(math.log(fraction))
    window = values[-window_size:]
    score = TV.lag1_autocorrelation(window)
    return (score, len(window)) if score is not None else None


def timing_score_from_elapsed(
    elapsed: Sequence[float],
    side: int,
    base: int,
    increment: int,
    skip_moves: int,
    window_size: int,
) -> tuple[float, int] | None:
    values: list[float] = []
    remaining = float(base)
    for move_index, used in enumerate(elapsed[side::2]):
        after_move = max(0.0, remaining + increment - used)
        remaining = after_move
        if move_index < skip_moves or used <= 0.0:
            continue
        # This deliberately mirrors the adapter's denominator:
        # remaining-after + used, including the post-move increment.
        before_move = after_move + used
        if before_move > 0.0:
            fraction = min(1.0, max(1e-6, used / before_move))
            values.append(math.log(fraction))
    window = values[-window_size:]
    score = TV.lag1_autocorrelation(window)
    return (score, len(window)) if score is not None else None


def row(account: str, label: str, score: float, samples: int, base: int, increment: int) -> dict:
    return {
        "account": TV.stable_hash("service-account", account),
        "label": label,
        "acf1": score,
        "samples": samples,
        "time_control": f"{base}+{increment}",
        "time_class": TV.classify_time_control(base, increment),
    }


def parse_chesscom(paths: Iterable[Path], config: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    excluded: collections.Counter = collections.Counter()
    games = 0
    for _, game in TV.iter_pgn_games(paths):
        games += 1
        headers = dict(TV.TAG_RE.findall(game))
        site = headers.get("Site", "").lower()
        link = headers.get("Link", "").lower()
        if "chess.com" not in site and "chess.com" not in link:
            excluded["non_chesscom"] += 1
            continue
        control = parse_service_time_control(headers.get("TimeControl", ""))
        if control is None:
            excluded["time_control"] += 1
            continue
        base, increment = control
        clocks = TV.parse_clocks(game)
        if not clocks:
            excluded["no_clocks"] += 1
            continue
        for side, color in enumerate(("White", "Black")):
            account = headers.get(color, "").strip()
            result = timing_score_from_clocks(
                clocks,
                side,
                base,
                increment,
                config["skip_initial_player_moves"],
                config["window_size"],
            )
            if not account or result is None:
                excluded["insufficient_perspective"] += 1
                continue
            score, samples = result
            # Chess.com's published archive does not affirm a computer label.
            rows.append(row(account, "unmarked", score, samples, base, increment))
    return rows, {"games_seen": games, "excluded": dict(sorted(excluded.items()))}


def parse_fics(paths: Iterable[Path], config: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    excluded: collections.Counter = collections.Counter()
    games = 0
    for _, game in TV.iter_pgn_games(paths):
        games += 1
        headers = dict(TV.TAG_RE.findall(game))
        if "fics" not in headers.get("Site", "").lower():
            excluded["non_fics"] += 1
            continue
        control = parse_service_time_control(headers.get("TimeControl", ""))
        if control is None:
            excluded["time_control"] += 1
            continue
        base, increment = control
        header_end = game.find("\n\n")
        movetext = game[header_end + 2 :] if header_end >= 0 else game
        mainline = TV.strip_variations(movetext)
        elapsed = [elapsed_seconds(match) for match in EMT_RE.finditer(mainline)]
        if not elapsed:
            excluded["no_elapsed_times"] += 1
            continue
        for side, color in enumerate(("White", "Black")):
            account = headers.get(color, "").strip()
            result = timing_score_from_elapsed(
                elapsed,
                side,
                base,
                increment,
                config["skip_initial_player_moves"],
                config["window_size"],
            )
            if not account or result is None:
                excluded["insufficient_perspective"] += 1
                continue
            score, samples = result
            label = "bot" if headers.get(f"{color}IsComp", "").lower() == "yes" else "unmarked"
            rows.append(row(account, label, score, samples, base, increment))
    return rows, {"games_seen": games, "excluded": dict(sorted(excluded.items()))}


def parse_lichess_records(path: Path) -> list[dict]:
    rows = []
    for record in TV.read_jsonl(path):
        rows.append(
            {
                "account": record["account"],
                "label": record["label"],
                "acf1": float(record["acf1"]),
                "samples": int(record["positive_samples"]),
                "time_control": record["time_control"],
                "time_class": record["time_class"],
            }
        )
    return rows


def account_scores(rows: Sequence[dict], label: str, time_class: str | None = None) -> list[float]:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for value in rows:
        if value["label"] != label:
            continue
        if time_class is not None and value["time_class"] != time_class:
            continue
        grouped[value["account"]].append(float(value["acf1"]))
    return [float(statistics.median(values)) for values in grouped.values()]


def class_summary(rows: Sequence[dict], threshold: float, label: str = "unmarked") -> dict | None:
    perspective_scores = [float(value["acf1"]) for value in rows if value["label"] == label]
    accounts = account_scores(rows, label)
    if not accounts:
        return None
    return {
        "perspectives": len(perspective_scores),
        "accounts": len(accounts),
        "perspective_median_acf1": statistics.median(perspective_scores),
        "perspective_threshold_share": sum(score >= threshold for score in perspective_scores)
        / len(perspective_scores),
        "account_median_acf1": statistics.median(accounts),
        "account_threshold_share": sum(score >= threshold for score in accounts) / len(accounts),
    }


def service_summary(rows: Sequence[dict], threshold: float, details: dict) -> dict:
    result = {
        "input": details,
        "unmarked": class_summary(rows, threshold, "unmarked"),
        "bot": class_summary(rows, threshold, "bot"),
        "by_time_class": {},
    }
    for time_class in ("bullet", "blitz", "rapid", "classical"):
        selected = [row for row in rows if row["time_class"] == time_class]
        summary = class_summary(selected, threshold, "unmarked")
        if summary is not None:
            result["by_time_class"][time_class] = summary
    bot_scores = account_scores(rows, "bot")
    unmarked_scores = account_scores(rows, "unmarked")
    result["raw_account_auc_higher_is_bot"] = (
        TV.roc_auc(bot_scores, unmarked_scores)
        if len(bot_scores) >= 5 and len(unmarked_scores) >= 5
        else None
    )
    result["auc_warning"] = (
        "descriptive unmatched AUC; use the separately matched Lichess validation as primary"
        if result["raw_account_auc_higher_is_bot"] is not None
        else "not reported: fewer than five affirmative BOT or unmarked accounts"
    )
    return result


def source_entry(paths: Sequence[Path], repository: str | None, commit: str | None) -> dict:
    return {
        "repository": repository,
        "commit": commit,
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "content_sha256": sorted(sha256_file(path) for path in paths),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def build_report(args: argparse.Namespace, config: dict) -> dict:
    threshold = config["production_threshold"]
    services = {}

    lichess_rows = parse_lichess_records(args.lichess_records)
    lichess_details = source_entry([args.lichess_records], None, None)
    lichess_details["kind"] = "pseudonymous CC0 Lichess-derived records"
    services["lichess"] = service_summary(lichess_rows, threshold, lichess_details)

    chesscom_rows, chesscom_parse = parse_chesscom(args.chesscom_pgn, config)
    chesscom_details = source_entry(args.chesscom_pgn, args.chesscom_repository, args.chesscom_commit)
    chesscom_details.update(chesscom_parse)
    chesscom_details["kind"] = "public Chess.com archive PGN mirror"
    services["chess.com"] = service_summary(chesscom_rows, threshold, chesscom_details)

    fics_rows, fics_parse = parse_fics(args.fics_pgn, config)
    fics_details = source_entry(args.fics_pgn, args.fics_repository, args.fics_commit)
    fics_details.update(fics_parse)
    fics_details["kind"] = "public FICS game archive with fractional-second elapsed times"
    services["fics"] = service_summary(fics_rows, threshold, fics_details)

    matched = json.loads(args.lichess_matched_report.read_text(encoding="utf-8"))
    return rounded(
        {
            "schema": SCHEMA_VERSION,
            "feature": "lag-1 autocorrelation of log clock fraction over the last 32 positive observations",
            "threshold": threshold,
            "services": services,
            "primary_matched_lichess": {
                "matching": matched["matching"],
                "account_level": matched["account_level"],
                "decision": matched["production_decision"],
            },
            "decision": "timing-only classification rejected across real-service benchmarks",
            "commercial_platform_coverage": {
                "chess.com": "benchmarked from public archive PGNs; no affirmative BOT field, so threshold stress test only",
                "internet_chess_club": "not benchmarked: no public bulk clock archive and affirmative computer labels available locally",
                "playchess_chessbase": "not benchmarked: no public bulk clock archive available locally",
                "fide_online_arena": "not benchmarked: no public bulk move-clock archive available locally",
                "chess24": "not benchmarked: playing service closed in 2024",
            },
            "limitations": [
                "Only Lichess supplies enough affirmative BOT labels for an AUC benchmark in this snapshot.",
                "Chess.com and FICS unmarked traffic can include automation; threshold shares are stress tests, not verified-human false-positive rates.",
                "Chess.com and FICS archives are account-centered samples, not natural-traffic samples.",
                "No result is fabricated for a commercial service without a lawful public export or user-supplied licensed data.",
            ],
        }
    )


def markdown(report: dict) -> str:
    lines = [
        "# Real-service timing benchmarks",
        "",
        "> Aggregate results only; no game-level service account identifiers, game IDs, moves, or raw commercial-service PGNs are committed.",
        "",
        f"**Decision: {report['decision']}.**",
        "",
        "## Cross-service account-level stress test",
        "",
        "| Service / label | Perspectives | Accounts | Median ACF1 | Accounts at ACF1 >= 0.45 |",
        "|---|---:|---:|---:|---:|",
    ]
    for service_name, service in report["services"].items():
        for label in ("bot", "unmarked"):
            summary = service[label]
            if summary is None:
                continue
            display_label = label
            if label == "bot" and summary["accounts"] < 5:
                display_label += " (insufficient; descriptive only)"
            lines.append(
                f"| {service_name} / {display_label} | {summary['perspectives']} | {summary['accounts']} | "
                f"{summary['account_median_acf1']:.3f} | {summary['account_threshold_share']:.1%} |"
            )
    primary = report["primary_matched_lichess"]["account_level"]
    lines.extend(
        [
            "",
            "## Primary matched Lichess classification result",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Account AUC (higher = BOT) | {primary['auc']:.3f} |",
            f"| Account-bootstrap 95% CI | [{primary['auc_bootstrap_95'][0]:.3f}, {primary['auc_bootstrap_95'][1]:.3f}] |",
            f"| Sensitivity at 0.45 | {primary['threshold_sensitivity']:.1%} |",
            f"| Unmarked threshold share at 0.45 | {primary['threshold_false_positive_rate']:.1%} |",
            "",
            "## Commercial platform availability",
            "",
            "| Platform | Status |",
            "|---|---|",
        ]
    )
    for platform, status in report["commercial_platform_coverage"].items():
        lines.append(f"| {platform.replace('_', ' ')} | {status} |")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lichess-records", type=Path, required=True)
    parser.add_argument("--lichess-matched-report", type=Path, required=True)
    parser.add_argument("--chesscom-pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--chesscom-repository")
    parser.add_argument("--chesscom-commit")
    parser.add_argument("--fics-pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--fics-repository")
    parser.add_argument("--fics-commit")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(args, config)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(
        "service timing benchmark: "
        + ", ".join(
            f"{name}={service['unmarked']['accounts'] if service['unmarked'] else 0} unmarked accounts"
            for name, service in report["services"].items()
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
