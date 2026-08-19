#!/usr/bin/env python3
"""Extract and validate Unchessed's conservative timing signal.

The production engine uses lag-1 autocorrelation of log(spent / clock-before)
over at most 32 positive-time observations.  This tool mirrors that feature on
Lichess PGNs, pseudonymises account/game identifiers, performs rating and exact
clock-control matching, and evaluates at the account boundary.

Only the Python standard library is required.  Raw PGNs are deliberately not
committed; the compact derived records contain no player names or moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

TAG_RE = re.compile(r'^\[([^ ]+) "(.*)"\]$', re.MULTILINE)
CLOCK_RE = re.compile(
    r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]"
    r"|\[%clkc\s+(\d+(?:\.\d+)?)\]"
)
SCHEMA_VERSION = 1
PSEUDONYM_NAMESPACE = "unchessed-timing-validation-v1"


@dataclass(frozen=True)
class Clock:
    seconds: float
    precision_seconds: float


def stable_hash(kind: str, value: str, length: int = 20) -> str:
    material = f"{PSEUDONYM_NAMESPACE}\0{kind}\0{value.lower()}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:length]


def deterministic_rank(seed: int, *parts: object) -> str:
    value = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iter_pgn_games(paths: Iterable[Path]) -> Iterator[tuple[Path, str]]:
    """Yield games without loading a multi-gigabyte PGN into memory."""
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            game: list[str] = []
            for line in handle:
                if line.startswith("[Event ") and game:
                    yield path, "".join(game)
                    game = []
                game.append(line)
            if game:
                yield path, "".join(game)


def strip_variations(movetext: str) -> str:
    """Remove recursive annotation variations while retaining mainline comments."""
    output: list[str] = []
    variation_depth = 0
    in_comment = False
    for char in movetext:
        if char == "{" and not in_comment:
            in_comment = True
        elif char == "}" and in_comment:
            in_comment = False

        if not in_comment:
            if char == "(":
                variation_depth += 1
                continue
            if char == ")" and variation_depth:
                variation_depth -= 1
                continue
        if variation_depth == 0:
            output.append(char)
    return "".join(output)


def parse_clocks(game: str) -> list[Clock]:
    header_end = game.find("\n\n")
    movetext = game[header_end + 2 :] if header_end >= 0 else game
    mainline = strip_variations(movetext)
    clocks: list[Clock] = []
    for match in CLOCK_RE.finditer(mainline):
        if match.group(1) is not None:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            second_text = match.group(3)
            seconds = hours * 3600 + minutes * 60 + float(second_text)
            decimals = len(second_text.partition(".")[2])
            precision = 10.0 ** (-decimals) if decimals else 1.0
        else:
            # Lichess's historical universal dump stores integer centiseconds.
            seconds = float(match.group(4)) / 100.0
            precision = 0.01
        clocks.append(Clock(seconds, precision))
    return clocks


def lag1_autocorrelation(values: Sequence[float]) -> float | None:
    """Match OpponentModel::timing_autocorrelation in adapt.rs."""
    if len(values) < 6:
        return None
    left = values[:-1]
    right = values[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator > 1e-12 else None


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def parse_time_control(value: str) -> tuple[int, int] | None:
    fields = value.split("+")
    if len(fields) != 2:
        return None
    try:
        base, increment = (int(field) for field in fields)
    except ValueError:
        return None
    if base <= 0 or increment < 0:
        return None
    return base, increment


def classify_time_control(base: int, increment: int) -> str:
    estimated = base + 40 * increment
    if estimated < 30:
        return "ultrabullet"
    if estimated < 180:
        return "bullet"
    if estimated < 480:
        return "blitz"
    if estimated < 1500:
        return "rapid"
    return "classical"


def timing_record(
    headers: dict[str, str],
    clocks: Sequence[Clock],
    side: int,
    label: str,
    source: str,
    config: dict,
) -> dict | None:
    color = "White" if side == 0 else "Black"
    account = headers.get(color, "").strip()
    if not account:
        return None
    try:
        rating = int(headers[f"{color}Elo"])
    except (KeyError, ValueError):
        return None
    parsed_control = parse_time_control(headers.get("TimeControl", ""))
    if parsed_control is None:
        return None
    base, increment = parsed_control

    side_clocks = clocks[side::2]
    if len(side_clocks) <= config["skip_initial_player_moves"]:
        return None

    previous = float(base)
    positive_logs: list[float] = []
    zero_or_negative = 0
    precision = 0.0
    for move_index, clock in enumerate(side_clocks):
        used = previous + increment - clock.seconds
        previous = clock.seconds
        precision = max(precision, clock.precision_seconds)
        if move_index < config["skip_initial_player_moves"]:
            continue
        # The runtime deliberately ignores a zero millisecond observation.
        # Rounded PGNs can also make a short positive think look non-positive.
        if used <= 0.0:
            zero_or_negative += 1
            continue
        before_move = clock.seconds + used
        if before_move <= 0.0:
            zero_or_negative += 1
            continue
        fraction = min(1.0, max(1e-6, used / before_move))
        positive_logs.append(math.log(fraction))

    positive_observations = len(positive_logs)
    window = positive_logs[-config["window_size"] :]
    acf1 = lag1_autocorrelation(window)
    if acf1 is None:
        return None
    mean_log = sum(window) / len(window)
    log_stddev = math.sqrt(sum((value - mean_log) ** 2 for value in window) / len(window))
    eligible_count = positive_observations + zero_or_negative
    site = headers.get("Site") or headers.get("GameId") or headers.get("UTCDate", "unknown")
    return {
        "schema": SCHEMA_VERSION,
        "account": stable_hash("account", account),
        "game": stable_hash("game", site),
        "label": label,
        "source": source,
        "rating": rating,
        "base_seconds": base,
        "increment_seconds": increment,
        "time_control": f"{base}+{increment}",
        "time_class": classify_time_control(base, increment),
        "date": headers.get("UTCDate", headers.get("Date", "unknown")),
        "clock_precision_seconds": precision,
        "positive_samples": len(window),
        "nonpositive_samples": zero_or_negative,
        "nonpositive_share": zero_or_negative / eligible_count if eligible_count else 0.0,
        "acf1": acf1,
        "median_log_fraction": median(window),
        "log_fraction_stddev": log_stddev,
    }


def extract_records(
    config: dict,
    bot_paths: Iterable[Path],
    unmarked_paths: Iterable[Path],
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    stats: Counter = Counter()

    corpora = (
        ("bot", "lichess-bot-title", list(bot_paths)),
        ("unmarked", "lichess-unmarked", list(unmarked_paths)),
    )
    for wanted_label, source, paths in corpora:
        for _, game in iter_pgn_games(paths):
            stats[f"{wanted_label}_games_seen"] += 1
            headers = dict(TAG_RE.findall(game))
            if "rated" not in headers.get("Event", "").lower():
                stats["excluded_not_rated"] += 1
                continue
            if headers.get("Variant", "Standard").lower() != "standard":
                stats["excluded_not_standard"] += 1
                continue
            if not headers.get("Site", "").startswith("https://lichess.org/"):
                stats["excluded_non_lichess"] += 1
                continue
            clocks = parse_clocks(game)
            if not clocks:
                stats["excluded_without_clocks"] += 1
                continue
            for side, color in enumerate(("White", "Black")):
                is_bot = headers.get(f"{color}Title", "").upper() == "BOT"
                if wanted_label == "bot" and not is_bot:
                    continue
                if wanted_label == "unmarked" and is_bot:
                    continue
                record = timing_record(headers, clocks, side, wanted_label, source, config)
                if record is None:
                    stats[f"{wanted_label}_perspectives_insufficient"] += 1
                    continue
                key = (record["account"], record["game"], record["label"])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                records.append(record)
                stats[f"{wanted_label}_records"] += 1

    records.sort(key=lambda row: (row["label"], row["account"], row["game"]))
    summary = {
        "schema": SCHEMA_VERSION,
        "records": len(records),
        "accounts": {
            label: len({row["account"] for row in records if row["label"] == label})
            for label in ("bot", "unmarked")
        },
        "statistics": dict(sorted(stats.items())),
    }
    return records, summary


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported schema")
            records.append(record)
    return records


def match_cell(record: dict, elo_bin_width: int) -> tuple[str, int]:
    return record["time_control"], int(record["rating"]) // elo_bin_width


def capped_records(records: Sequence[dict], config: dict, label: str) -> list[dict]:
    match_config = config["matching"]
    candidates = [record for record in records if record["label"] == label]
    candidates.sort(
        key=lambda row: deterministic_rank(
            config["seed"], label, row["account"], row["game"]
        )
    )
    account_counts: Counter = Counter()
    account_cell_counts: Counter = Counter()
    selected: list[dict] = []
    for record in candidates:
        cell = match_cell(record, match_config["elo_bin_width"])
        account = record["account"]
        if account_counts[account] >= match_config["max_records_per_account"]:
            continue
        if (
            account_cell_counts[(account, cell)]
            >= match_config["max_records_per_account_cell"]
        ):
            continue
        account_counts[account] += 1
        account_cell_counts[(account, cell)] += 1
        selected.append(record)
    return selected


def matched_records(records: Sequence[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """Greedy 1:1 matching by exact clock control and source-rating bin."""
    unmarked = capped_records(records, config, "unmarked")
    width = config["matching"]["elo_bin_width"]
    available_cells = {match_cell(record, width) for record in unmarked}
    # Apply account caps only after removing BOT cells that cannot possibly be
    # matched. Otherwise a prolific account's unmatched controls can consume
    # its cap and arbitrarily suppress valid controls.
    bot_eligible = [
        record
        for record in records
        if record["label"] != "bot"
        or match_cell(record, width) in available_cells
    ]
    bots = capped_records(bot_eligible, config, "bot")
    pools: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in unmarked:
        pools[match_cell(record, width)].append(record)
    bots.sort(
        key=lambda row: deterministic_rank(
            config["seed"], "bot-order", row["account"], row["game"]
        )
    )
    matched_bots: list[dict] = []
    matched_unmarked: list[dict] = []
    for bot in bots:
        pool = pools[match_cell(bot, width)]
        if not pool:
            continue
        # The cell prevents broad rating shortcuts; nearest-neighbour choice
        # further reduces residual imbalance without fitting to ACF1.
        candidate_index = min(
            range(len(pool)),
            key=lambda index: (
                abs(int(pool[index]["rating"]) - int(bot["rating"])),
                deterministic_rank(
                    config["seed"],
                    "pool",
                    pool[index]["account"],
                    pool[index]["game"],
                ),
            ),
        )
        matched_bots.append(bot)
        matched_unmarked.append(pool.pop(candidate_index))
    return matched_bots, matched_unmarked


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def aggregate_accounts(records: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["label"], record["account"])].append(record)
    output: list[dict] = []
    for (label, account), rows in sorted(grouped.items()):
        output.append(
            {
                "label": label,
                "account": account,
                "records": len(rows),
                "acf1": median([row["acf1"] for row in rows]),
                "median_log_fraction": median(
                    [row["median_log_fraction"] for row in rows]
                ),
                "log_fraction_stddev": median(
                    [row["log_fraction_stddev"] for row in rows]
                ),
                "nonpositive_share": median(
                    [row["nonpositive_share"] for row in rows]
                ),
                "positive_samples": median(
                    [float(row["positive_samples"]) for row in rows]
                ),
            }
        )
    return output


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_metric(
    bot_scores: Sequence[float],
    unmarked_scores: Sequence[float],
    iterations: int,
    seed: int,
    metric,
) -> tuple[float, list[float]]:
    estimate = metric(bot_scores, unmarked_scores)
    if estimate is None:
        return float("nan"), [float("nan"), float("nan")]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_bots = [rng.choice(bot_scores) for _ in bot_scores]
        sampled_unmarked = [rng.choice(unmarked_scores) for _ in unmarked_scores]
        value = metric(sampled_bots, sampled_unmarked)
        if value is not None and math.isfinite(value):
            samples.append(value)
    return float(estimate), [percentile(samples, 0.025), percentile(samples, 0.975)]


def rate_metric(threshold: float, positive: bool):
    def metric(bot_scores: Sequence[float], unmarked_scores: Sequence[float]) -> float:
        scores = bot_scores if positive else unmarked_scores
        return sum(score >= threshold for score in scores) / len(scores)

    return metric


def permutation_p_value(
    bot_scores: Sequence[float],
    unmarked_scores: Sequence[float],
    observed_auc: float,
    iterations: int,
    seed: int,
) -> float:
    """One-sided account permutation test for the production (higher=bot) sign."""
    rng = random.Random(seed)
    scores = list(bot_scores) + list(unmarked_scores)
    bot_count = len(bot_scores)
    extreme = 0
    for _ in range(iterations):
        rng.shuffle(scores)
        value = roc_auc(scores[:bot_count], scores[bot_count:])
        if value is not None and value >= observed_auc:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def fit_scalar_logistic(
    rows: Sequence[dict], held_out: str, l2: float = 0.1
) -> tuple[float, float, tuple[float, float]]:
    training = [row for row in rows if row["account"] != held_out]
    values = [row["acf1"] for row in training]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance) if variance > 1e-12 else 1.0
    positives = sum(row["label"] == "bot" for row in training)
    negatives = len(training) - positives
    intercept = 0.0
    weight = 0.0
    # Equal total class weight prevents the larger unmarked class from
    # becoming an accidental prior in every leave-one-account-out fold.
    for _ in range(800):
        grad_intercept = 0.0
        grad_weight = l2 * weight
        total_weight = 0.0
        for row in training:
            target = 1.0 if row["label"] == "bot" else 0.0
            class_weight = 0.5 / (positives if target else negatives)
            x_value = (row["acf1"] - mean) / scale
            error = sigmoid(intercept + weight * x_value) - target
            grad_intercept += class_weight * error
            grad_weight += class_weight * error * x_value
            total_weight += class_weight
        intercept -= 0.2 * grad_intercept / total_weight
        weight -= 0.2 * grad_weight / total_weight
    return intercept, weight, (mean, scale)


def loao_predictions(accounts: Sequence[dict]) -> list[tuple[int, float]]:
    predictions: list[tuple[int, float]] = []
    for held in accounts:
        intercept, weight, normalization = fit_scalar_logistic(accounts, held["account"])
        mean, scale = normalization
        probability = sigmoid(intercept + weight * (held["acf1"] - mean) / scale)
        predictions.append((1 if held["label"] == "bot" else 0, probability))
    return predictions


def expected_calibration_error(predictions: Sequence[tuple[int, float]], bins: int = 5) -> float:
    total = len(predictions)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            (label, probability)
            for label, probability in predictions
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not selected:
            continue
        accuracy = sum(label for label, _ in selected) / len(selected)
        confidence = sum(probability for _, probability in selected) / len(selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return error


def rounded(value: float | None, digits: int = 6):
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def score_summary(scores: Sequence[float]) -> dict:
    return {
        "count": len(scores),
        "median": rounded(median(scores)),
        "mean": rounded(sum(scores) / len(scores)),
    }


def evaluate(records: Sequence[dict], config: dict) -> dict:
    bot_ids = {row["account"] for row in records if row["label"] == "bot"}
    unmarked_ids = {
        row["account"] for row in records if row["label"] == "unmarked"
    }
    overlap = bot_ids & unmarked_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} accounts occur in both label classes; refusing leakage"
        )
    matched_bots, matched_unmarked = matched_records(records, config)
    if not matched_bots:
        raise ValueError("no BOT/unmarked records survive exact matching")
    matched = [*matched_bots, *matched_unmarked]
    accounts = aggregate_accounts(matched)
    bot_accounts = [row for row in accounts if row["label"] == "bot"]
    unmarked_accounts = [row for row in accounts if row["label"] == "unmarked"]
    bot_scores = [row["acf1"] for row in bot_accounts]
    unmarked_scores = [row["acf1"] for row in unmarked_accounts]
    iterations = config["bootstrap_iterations"]
    seed = config["seed"]

    account_auc, account_auc_ci = bootstrap_metric(
        bot_scores, unmarked_scores, iterations, seed + 1, roc_auc
    )
    threshold = config["production_threshold"]
    sensitivity, sensitivity_ci = bootstrap_metric(
        bot_scores,
        unmarked_scores,
        iterations,
        seed + 2,
        rate_metric(threshold, True),
    )
    false_positive_rate, false_positive_ci = bootstrap_metric(
        bot_scores,
        unmarked_scores,
        iterations,
        seed + 3,
        rate_metric(threshold, False),
    )

    sequence_bot_scores = [row["acf1"] for row in matched_bots]
    sequence_unmarked_scores = [row["acf1"] for row in matched_unmarked]
    sequence_auc = roc_auc(sequence_bot_scores, sequence_unmarked_scores)
    rating_differences = [
        abs(int(bot["rating"]) - int(unmarked["rating"]))
        for bot, unmarked in zip(matched_bots, matched_unmarked)
    ]

    predictions = loao_predictions(accounts)
    loao_auc = roc_auc(
        [probability for label, probability in predictions if label == 1],
        [probability for label, probability in predictions if label == 0],
    )
    brier = sum((probability - label) ** 2 for label, probability in predictions) / len(
        predictions
    )

    controls: list[dict] = []
    for control in sorted({row["time_control"] for row in matched_bots}):
        positive = [
            row["acf1"] for row in matched_bots if row["time_control"] == control
        ]
        negative = [
            row["acf1"] for row in matched_unmarked if row["time_control"] == control
        ]
        controls.append(
            {
                "time_control": control,
                "pairs": min(len(positive), len(negative)),
                "auc": rounded(roc_auc(positive, negative)),
                "bot_median": rounded(median(positive)),
                "unmarked_median": rounded(median(negative)),
            }
        )

    source_counts = {
        label: {
            "records": sum(row["label"] == label for row in records),
            "accounts": len({row["account"] for row in records if row["label"] == label}),
        }
        for label in ("bot", "unmarked")
    }
    gates_config = config["gates"]
    gates = {
        "source_bot_accounts": {
            "value": source_counts["bot"]["accounts"],
            "required": gates_config["minimum_source_bot_accounts"],
            "pass": source_counts["bot"]["accounts"]
            >= gates_config["minimum_source_bot_accounts"],
        },
        "matched_bot_accounts": {
            "value": len(bot_accounts),
            "required": gates_config["minimum_matched_bot_accounts"],
            "pass": len(bot_accounts) >= gates_config["minimum_matched_bot_accounts"],
        },
        "account_auc_lower_95": {
            "value": rounded(account_auc_ci[0]),
            "required": gates_config["minimum_account_auc_lower_95"],
            "pass": account_auc_ci[0] >= gates_config["minimum_account_auc_lower_95"],
        },
        "threshold_fpr_upper_95": {
            "value": rounded(false_positive_ci[1]),
            "required_maximum": gates_config["maximum_threshold_fpr_upper_95"],
            "pass": false_positive_ci[1]
            <= gates_config["maximum_threshold_fpr_upper_95"],
        },
    }
    all_gates_pass = all(gate["pass"] for gate in gates.values())

    return {
        "schema": SCHEMA_VERSION,
        "seed": seed,
        "label_definition": {
            "positive": "Lichess BOT title (affirmative Bot API account)",
            "negative": "unmarked account from a separate rated-game corpus; noisy human proxy, not affirmative human identity",
        },
        "feature": "lag-1 autocorrelation of log(spent / clock-before), last 32 positive observations",
        "source": source_counts,
        "matching": {
            "pairs": len(matched_bots),
            "bot_accounts": len(bot_accounts),
            "unmarked_accounts": len(unmarked_accounts),
            "exact_time_control": True,
            "elo_bin_width": config["matching"]["elo_bin_width"],
            "median_absolute_rating_difference": rounded(median(rating_differences)),
            "maximum_absolute_rating_difference": max(rating_differences),
            "max_records_per_account": config["matching"]["max_records_per_account"],
            "max_records_per_account_cell": config["matching"][
                "max_records_per_account_cell"
            ],
        },
        "account_level": {
            "bot": score_summary(bot_scores),
            "unmarked": score_summary(unmarked_scores),
            "auc": rounded(account_auc),
            "auc_bootstrap_95": [rounded(value) for value in account_auc_ci],
            "higher_is_bot_permutation_p": rounded(
                permutation_p_value(
                    bot_scores,
                    unmarked_scores,
                    account_auc,
                    config["permutation_iterations"],
                    seed + 4,
                )
            ),
            "threshold": threshold,
            "threshold_sensitivity": rounded(sensitivity),
            "threshold_sensitivity_bootstrap_95": [
                rounded(value) for value in sensitivity_ci
            ],
            "threshold_false_positive_rate": rounded(false_positive_rate),
            "threshold_false_positive_rate_bootstrap_95": [
                rounded(value) for value in false_positive_ci
            ],
        },
        "matched_sequence_level": {
            "bot": score_summary(sequence_bot_scores),
            "unmarked": score_summary(sequence_unmarked_scores),
            "auc_descriptive_only": rounded(sequence_auc),
            "warning": "sequences from one account are correlated; account-level metrics are primary",
        },
        "leave_one_account_out_logistic": {
            "feature": "acf1 only",
            "accounts": len(predictions),
            "auc": rounded(loao_auc),
            "brier": rounded(brier),
            "ece_5_bin": rounded(expected_calibration_error(predictions)),
            "warning": "exploratory calibration only; it may learn a reversed sign and is not used by the engine",
        },
        "by_time_control": controls,
        "gates": gates,
        "all_standalone_classifier_gates_pass": all_gates_pass,
        "production_decision": (
            "eligible-for-separate-shadow-review" if all_gates_pass else "reject-standalone-timing-classifier"
        ),
        "runtime_policy": "timing remains a weak modulator for independently ceiling-level play and can never trigger classification alone",
        "limitations": [
            "Lichess %clk values in this snapshot have one-second precision, so sub-second behavior is censored.",
            "The unmarked class is a human proxy and can contain undeclared automation or assisted play.",
            "PGN-only extraction cannot reproduce the runtime legal-choice and search-difficulty filter.",
            "The matched Bot API account sample is small and not engine-family-disjoint.",
            "A result on these public accounts does not establish generalization to unseen accounts or platforms.",
        ],
    }


def markdown_report(report: dict) -> str:
    account = report["account_level"]
    loao = report["leave_one_account_out_logistic"]
    matching = report["matching"]
    lines = [
        "# Timing-classifier validation result",
        "",
        "> Generated by `tools/timing_classifier_validation.py`; do not edit the numerical tables by hand.",
        "",
        "## Decision",
        "",
        f"**{report['production_decision']}**.",
        "",
        report["runtime_policy"] + ".",
        "",
        "## Primary account-level result",
        "",
        f"The matched sample contains **{matching['pairs']} game-perspective pairs**, "
        f"**{matching['bot_accounts']} BOT accounts**, and "
        f"**{matching['unmarked_accounts']} unmarked accounts**. Matching uses exact "
        f"time control, nearest-neighbour selection within {matching['elo_bin_width']}-point "
        f"source-rating cells, and has a median absolute rating difference of "
        f"{matching['median_absolute_rating_difference']:.0f} points.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| BOT median ACF1 | {account['bot']['median']:.3f} |",
        f"| Unmarked median ACF1 | {account['unmarked']['median']:.3f} |",
        f"| Account AUC (higher = BOT) | {account['auc']:.3f} |",
        f"| Account AUC, account-bootstrap 95% CI | [{account['auc_bootstrap_95'][0]:.3f}, {account['auc_bootstrap_95'][1]:.3f}] |",
        f"| One-sided account permutation p | {account['higher_is_bot_permutation_p']:.3f} |",
        f"| Sensitivity at ACF1 >= {account['threshold']:.2f} | {account['threshold_sensitivity']:.3f} [{account['threshold_sensitivity_bootstrap_95'][0]:.3f}, {account['threshold_sensitivity_bootstrap_95'][1]:.3f}] |",
        f"| False-positive rate at ACF1 >= {account['threshold']:.2f} | {account['threshold_false_positive_rate']:.3f} [{account['threshold_false_positive_rate_bootstrap_95'][0]:.3f}, {account['threshold_false_positive_rate_bootstrap_95'][1]:.3f}] |",
        f"| Strict LOAO scalar-logistic AUC | {loao['auc']:.3f} |",
        f"| Strict LOAO Brier / 5-bin ECE | {loao['brier']:.3f} / {loao['ece_5_bin']:.3f} |",
        "",
        "The fixed `0.45` threshold is not validated as a standalone classifier. "
        "The LOAO model is exploratory and is not shipped; in particular, allowing a "
        "fitted model to reverse the expected sign would not validate the production rule.",
        "",
        "## Time-control slices",
        "",
        "| Time control | Matched pairs | AUC | BOT median | Unmarked median |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["by_time_control"]:
        lines.append(
            f"| {row['time_control']} | {row['pairs']} | {row['auc']:.3f} | "
            f"{row['bot_median']:.3f} | {row['unmarked_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Configured standalone gates",
            "",
            "| Gate | Observed | Requirement | Pass |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name, gate in report["gates"].items():
        requirement = gate.get("required", gate.get("required_maximum"))
        operator = ">=" if "required" in gate else "<="
        lines.append(
            f"| `{name}` | {gate['value']} | {operator} {requirement} | "
            f"{'yes' if gate['pass'] else '**no**'} |"
        )
    lines.extend(
        [
            "",
            "## Label and measurement limits",
            "",
            f"Positive label: {report['label_definition']['positive']}.",
            "",
            f"Negative label: {report['label_definition']['negative']}.",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise SystemExit(f"generated output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="derive pseudonymous records from PGNs")
    extract.add_argument("--config", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--summary", type=Path)
    extract.add_argument("--bot-pgn", type=Path, nargs="+", required=True)
    extract.add_argument("--unmarked-pgn", type=Path, nargs="+", required=True)

    validate = subparsers.add_parser("validate", help="evaluate a derived JSONL snapshot")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--records", type=Path, required=True)
    validate.add_argument(
        "--manifest",
        type=Path,
        help="verify records bytes/SHA-256 and record the manifest hash",
    )
    validate.add_argument("--json", type=Path, required=True)
    validate.add_argument("--markdown", type=Path, required=True)
    validate.add_argument("--check", action="store_true")
    validate.add_argument(
        "--strict-gates",
        action="store_true",
        help="return nonzero when standalone-classifier gates fail",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema")
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(records_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["derived_outputs"]["records.jsonl"]
    actual_size = records_path.stat().st_size
    actual_sha256 = file_sha256(records_path)
    if actual_size != expected["bytes"]:
        raise ValueError(
            f"{records_path}: {actual_size} bytes, manifest expects {expected['bytes']}"
        )
    if actual_sha256 != expected["sha256"]:
        raise ValueError(f"{records_path}: SHA-256 does not match {manifest_path}")
    return {
        "records_bytes": actual_size,
        "records_sha256": actual_sha256,
        "source_manifest_sha256": file_sha256(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.command == "extract":
        records, summary = extract_records(config, args.bot_pgn, args.unmarked_pgn)
        write_jsonl(args.output, records)
        summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(summary_text, encoding="utf-8", newline="\n")
        print(summary_text, end="")
        return 0

    snapshot = (
        verify_snapshot(args.records, args.manifest)
        if args.manifest
        else {
            "records_bytes": args.records.stat().st_size,
            "records_sha256": file_sha256(args.records),
            "source_manifest_sha256": None,
        }
    )
    records = read_jsonl(args.records)
    report = evaluate(records, config)
    report["snapshot"] = snapshot
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown = markdown_report(report)
    write_or_check(args.json, json_text, args.check)
    write_or_check(args.markdown, markdown, args.check)
    print(
        f"timing validation: {report['production_decision']}; "
        f"account AUC={report['account_level']['auc']} "
        f"CI={report['account_level']['auc_bootstrap_95']}"
    )
    if args.strict_gates and not report["all_standalone_classifier_gates_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
