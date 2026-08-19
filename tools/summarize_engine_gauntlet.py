#!/usr/bin/env python3
"""Summarize paired real-engine Fastchess PGNs without making Elo claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

TAG_RE = re.compile(r'^\[([^ ]+) "(.*)"\]$', re.MULTILINE)
MOVE_COMMENT_RE = re.compile(
    r"([a-h][1-8][a-h][1-8][qrbn]?)\s*\{([^{}]*)\}", re.IGNORECASE
)
SEARCH_RE = re.compile(r"\bn=(\d+),\s*nps=(\d+)\b")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_games(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [match.start() for match in re.finditer(r"(?m)^\[Event ", text)]
    starts.append(len(text))
    for start, end in zip(starts, starts[1:]):
        game = text[start:end]
        if game.strip():
            yield game


def candidate_points(headers: dict[str, str], candidate: str) -> float:
    result = headers["Result"]
    candidate_is_white = headers["White"] == candidate
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if candidate_is_white else 0.0
    if result == "0-1":
        return 0.0 if candidate_is_white else 1.0
    raise ValueError(f"unsupported result: {result}")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(pair_points: list[float], seed: int, iterations: int) -> list[float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        total = sum(rng.choice(pair_points) for _ in pair_points)
        samples.append(100.0 * total / (2.0 * len(pair_points)))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def summarize(path: Path, candidate: str, seed: int, iterations: int) -> dict:
    wins = losses = draws = 0
    by_round: dict[str, list[float]] = defaultdict(list)
    terminations: Counter = Counter()
    throughput: dict[str, list[int]] = defaultdict(list)
    completed_nodes: dict[str, list[int]] = defaultdict(list)
    engine_ids: dict[str, set[str]] = defaultdict(set)
    games = 0
    opponent = None

    for game in iter_games(path):
        headers = dict(TAG_RE.findall(game))
        if candidate not in (headers.get("White"), headers.get("Black")):
            raise ValueError(f"{path}: candidate {candidate!r} missing from game")
        other = headers["Black"] if headers["White"] == candidate else headers["White"]
        if opponent is None:
            opponent = other
        elif opponent != other:
            raise ValueError(f"{path}: more than one opponent")
        point = candidate_points(headers, candidate)
        if point == 1.0:
            wins += 1
        elif point == 0.5:
            draws += 1
        else:
            losses += 1
        by_round[headers.get("Round", str(games + 1))].append(point)
        terminations[headers.get("Termination", "unknown")] += 1
        games += 1

        for color in ("White", "Black"):
            displayed = headers.get(color)
            identity = headers.get(f"Engine{color}Name")
            if displayed and identity:
                engine_ids[displayed].add(identity)

        header_end = game.find("\n\n")
        movetext = game[header_end + 2 :] if header_end >= 0 else game
        # Fastchess emits one move/comment token per ply, including book moves.
        for ply, (_, comment) in enumerate(MOVE_COMMENT_RE.findall(movetext)):
            search = SEARCH_RE.search(comment)
            if search is None:
                continue
            engine = headers["White"] if ply % 2 == 0 else headers["Black"]
            completed_nodes[engine].append(int(search.group(1)))
            throughput[engine].append(int(search.group(2)))

    if games == 0:
        raise ValueError(f"{path}: no games")
    pair_points = []
    for round_name, points in sorted(by_round.items()):
        if len(points) != 2:
            raise ValueError(f"{path}: round {round_name} has {len(points)} games, expected 2")
        pair_points.append(sum(points))

    points = wins + 0.5 * draws
    score_percent = 100.0 * points / games
    return {
        "pgn": path.name,
        "pgn_sha256": file_sha256(path),
        "candidate": candidate,
        "opponent": opponent,
        "games": games,
        "pairs": len(pair_points),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "points": points,
        "score_percent": round(score_percent, 6),
        "paired_bootstrap_score_95": [
            round(value, 6)
            for value in paired_bootstrap(pair_points, seed, iterations)
        ],
        "pentanomial_0_to_2": [
            sum(abs(point - target) < 1e-9 for point in pair_points)
            for target in (0.0, 0.5, 1.0, 1.5, 2.0)
        ],
        "terminations": dict(sorted(terminations.items())),
        "engine_ids": {
            name: sorted(values) for name, values in sorted(engine_ids.items())
        },
        "median_last_completed_nodes": {
            name: int(statistics.median(values))
            for name, values in sorted(completed_nodes.items())
        },
        "median_reported_nps": {
            name: int(statistics.median(values))
            for name, values in sorted(throughput.items())
        },
    }


def markdown(report: dict) -> str:
    settings = report["settings"]
    lines = [
        "# Real-engine gauntlet",
        "",
        "These are actual paired UCI games, not synthetic positions or self-play.",
        "",
        "## Conditions",
        "",
        f"- {settings['nodes_per_move']:,} nodes per move for both engines",
        f"- {settings['threads']} thread and {settings['hash_mb']} MiB hash per engine",
        f"- {settings['opening_pairs']} opening pairs with colors reversed",
        f"- opening source: `{settings['openings']}` ({settings['opening_plies']} plies)",
        "- no tablebases; one physical host core; concurrency 1",
        "- results include adjudication under the recorded Fastchess settings",
        "",
        "## Results from the Unchessed perspective",
        "",
        "| Opponent | W-L-D | Score | Pair-bootstrap 95% | Median Unchessed NPS | Median opponent NPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for match in report["matches"]:
        candidate_nps = match["median_reported_nps"].get(match["candidate"], 0)
        opponent_nps = match["median_reported_nps"].get(match["opponent"], 0)
        interval = match["paired_bootstrap_score_95"]
        lines.append(
            f"| {match['opponent']} | {match['wins']}-{match['losses']}-{match['draws']} | "
            f"{match['score_percent']:.2f}% | [{interval[0]:.2f}%, {interval[1]:.2f}%] | "
            f"{candidate_nps:,} | {opponent_nps:,} |"
        )
    total_games = sum(match["games"] for match in report["matches"])
    total_points = sum(match["points"] for match in report["matches"])
    lines.extend(
        [
            "",
            f"Combined descriptive score: **{total_points:.1f}/{total_games} "
            f"({100.0 * total_points / total_games:.2f}%)**.",
            "",
            "## Interpretation",
            "",
            "The sample is deliberately small and its intervals are wide. It proves UCI interoperability and provides a real-engine smoke baseline; it does not establish an Elo rating or a statistically significant ordering. Stockfish 12 was compiled in classical-evaluation mode because current Stockfish 18 network assets were unavailable on this runner. Ethereal 14 and Berserk 4.7 are also real tagged releases using their HCE paths.",
            "",
            "Full PGNs and exact source/build provenance are committed beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="Unchessed")
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    report = {
        "schema": 1,
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap,
        **provenance,
        "matches": [
            summarize(path, args.candidate, args.seed + index, args.bootstrap)
            for index, path in enumerate(args.pgn)
        ],
        "decision": "real-engine interoperability passed; sample too small for Elo claims",
    }
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown(report)
    if args.check:
        if args.json.read_text(encoding="utf-8") != json_text:
            raise ValueError(f"generated report differs: {args.json}")
        if args.markdown.read_text(encoding="utf-8") != markdown_text:
            raise ValueError(f"generated report differs: {args.markdown}")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json_text, encoding="utf-8")
        args.markdown.write_text(markdown_text, encoding="utf-8")
    print(f"summarized {sum(match['games'] for match in report['matches'])} real-engine games")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
