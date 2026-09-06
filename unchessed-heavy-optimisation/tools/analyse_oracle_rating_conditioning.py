#!/usr/bin/env python3
"""Offline diagnostic for scalar-rating conditioning in a trusted legacy oracle.

This tool is analysis-only: it loads no engine code and changes no production
behaviour.  It accepts only project-controlled
``UNARCHV1_ORACLE_TRAINING_V1_DDP`` PyTorch checkpoints.  Those checkpoints
use Python serialization, so do not supply an untrusted file.

The default probe is the frozen first 200 calibration FENs evaluated in paired
seven-row batches at ratings 600, 1000, 1400, 1800, 2200, 2600, and 3200.  It
holds move history at zero, time class at 2, and policy kind at 1, then records
raw response metrics rather than inferring a strength or promotion conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LEGACY_FORMAT = "UNARCHV1_ORACLE_TRAINING_V1_DDP"
DUAL_ELO_FORMAT = "UNARCHV1_PRETRAIN_DUAL_ELO_V1"
RUNTIME_FORMAT = "UNARCHV1"
RUNTIME_MAGIC = b"UNARCHV1"
DEFAULT_RATINGS = [600, 1000, 1400, 1800, 2200, 2600, 3200]
MAX_LEGAL_ACTIONS = 218


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--oracle", type=Path, required=True,
        help="trusted legacy UNARCHV1_ORACLE_TRAINING_V1_DDP checkpoint",
    )
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="calibration corpus JSONL (its manifest row is skipped)",
    )
    parser.add_argument(
        "--labels", type=Path,
        help="optional FEN-keyed fixed calibration-score labels JSON",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--ratings", type=int, nargs="+", default=DEFAULT_RATINGS)
    parser.add_argument("--policy-kind", type=int, default=1)
    parser.add_argument("--time-class", type=int, default=2)
    parser.add_argument("--json", type=Path, help="write the completed report here")
    return parser


def file_sha256(path: Path) -> str:
    """Return a file digest without materialising a checkpoint in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def validate_paths(args: argparse.Namespace) -> int | None:
    """Validate all user-supplied files before importing Torch or reading data."""
    if not args.oracle.is_file():
        return fail(f"missing oracle checkpoint: {args.oracle}")
    if not args.corpus.is_file():
        return fail(f"missing corpus: {args.corpus}")
    if args.labels is not None and not args.labels.is_file():
        return fail(f"missing labels: {args.labels}")
    if args.limit <= 0:
        return fail("limit must be positive")
    if not args.ratings:
        return fail("at least one rating is required")
    if len(set(args.ratings)) != len(args.ratings):
        return fail("ratings must not contain duplicates")
    return None


def preflight_runtime_package(path: Path) -> int | None:
    """Reject the non-PyTorch runtime package without requiring Torch.

    A runtime package is not a potentially valid legacy checkpoint, and its
    fixed eight-byte magic can be identified without deserialising anything.
    Other checkpoint formats are identified from their loaded mapping below.
    """
    try:
        with path.open("rb") as handle:
            magic = handle.read(len(RUNTIME_MAGIC))
    except OSError as exc:
        return fail(f"unable to read oracle checkpoint: {path}: {exc}")
    if magic == RUNTIME_MAGIC:
        return fail(
            "unsupported oracle checkpoint format: UNARCHV1 "
            "(runtime packages are not legacy oracle checkpoints)"
        )
    return None


def format_error(checkpoint_format: Any) -> int:
    if checkpoint_format == RUNTIME_FORMAT:
        return fail(
            "unsupported oracle checkpoint format: UNARCHV1 "
            "(runtime packages are not legacy oracle checkpoints)"
        )
    if checkpoint_format == DUAL_ELO_FORMAT:
        return fail(
            "unsupported oracle checkpoint format: "
            "UNARCHV1_PRETRAIN_DUAL_ELO_V1 "
            "(dual-Elo checkpoints require a separate two-input experiment)"
        )
    return fail(
        "unsupported oracle checkpoint format: "
        f"{checkpoint_format!r}; expected {LEGACY_FORMAT}"
    )


def load_legacy_oracle(path: Path):
    """Load one trusted, legacy checkpoint on CPU and instantiate it strictly."""
    try:
        import torch
    except ModuleNotFoundError:
        return None, None, fail(
            "missing dependency: torch; install it on the trusted analysis host"
        )

    # The legacy trainer/evaluator uses this exact pickle-capable loading mode.
    # This is intentionally reached only after every requested file was checked.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # trusted input can still be corrupt or incompatible
        return None, None, fail(f"unable to load trusted oracle checkpoint: {path}: {exc}")

    if not isinstance(checkpoint, Mapping):
        return None, None, fail(
            "unsupported oracle checkpoint format: checkpoint is not a mapping; "
            f"expected {LEGACY_FORMAT}"
        )
    if checkpoint.get("format") != LEGACY_FORMAT:
        return None, None, format_error(checkpoint.get("format"))
    if not isinstance(checkpoint.get("config"), Mapping):
        return None, None, fail(
            "unsupported oracle checkpoint format: missing mapping-valued config"
        )
    if not isinstance(checkpoint.get("model"), Mapping):
        return None, None, fail(
            "unsupported oracle checkpoint format: missing mapping-valued model"
        )

    try:
        from train_unarchitectured_metal_a100 import UnarchitecturedV1Oracle

        model = UnarchitecturedV1Oracle(checkpoint["config"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to("cpu")
        model.eval()
    except Exception as exc:
        return None, None, fail(f"incompatible oracle checkpoint: {path}: {exc}")
    return torch, (checkpoint, model), None


def load_corpus(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"corpus line {line_number} is not an object")
            # The first corpus record is a manifest, which intentionally has no FEN.
            if "fen" in record:
                if not isinstance(record["fen"], str):
                    raise ValueError(f"corpus line {line_number} has a non-string FEN")
                rows.append(dict(record))
                if len(rows) == limit:
                    break
    if not rows:
        raise ValueError("corpus contains no FEN records")
    return rows


def load_labels(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, Mapping):
        raise ValueError("labels JSON is not a mapping")
    return labels


def make_batch(
    torch: Any,
    encoded: Mapping[str, Any],
    config: Mapping[str, Any],
    ratings: list[int],
    policy_kinds: list[int],
    time_class: int,
) -> tuple[dict[str, Any], int]:
    """Make paired CPU rows with an identical board, actions, and zero history."""
    if len(ratings) != len(policy_kinds):
        raise ValueError("ratings and policy kinds must have equal lengths")
    history_plies = config.get("history_plies")
    if not isinstance(history_plies, int) or history_plies <= 0:
        raise ValueError("oracle config has no positive integer history_plies")

    actions = list(encoded["legal_actions"])
    legal_count = len(actions)
    if not 1 <= legal_count <= MAX_LEGAL_ACTIONS:
        raise ValueError(
            f"position has {legal_count} legal moves; expected 1..{MAX_LEGAL_ACTIONS}"
        )
    rows = len(ratings)
    safe_actions = actions + [0] * (MAX_LEGAL_ACTIONS - legal_count)
    legal_mask = [index < legal_count for index in range(MAX_LEGAL_ACTIONS)]

    return {
        "pieces": torch.tensor(encoded["pieces"], dtype=torch.long).view(1, 64).repeat(rows, 1),
        "castling": torch.full((rows,), int(encoded["castling"]), dtype=torch.long),
        "ep_file": torch.full((rows,), int(encoded["ep_file"]), dtype=torch.long),
        "halfmove_bucket": torch.full(
            (rows,), int(encoded["halfmove_bucket"]), dtype=torch.long
        ),
        "history": torch.zeros((rows, history_plies), dtype=torch.long),
        "history_len": torch.zeros((rows,), dtype=torch.long),
        "rating": torch.tensor(ratings, dtype=torch.long),
        "time_class": torch.full((rows,), time_class, dtype=torch.long),
        "policy_kind": torch.tensor(policy_kinds, dtype=torch.long),
        "safe_actions": torch.tensor(safe_actions, dtype=torch.long).view(1, -1).repeat(rows, 1),
        "legal_mask": torch.tensor(legal_mask, dtype=torch.bool).view(1, -1).repeat(rows, 1),
    }, legal_count


def fixed_label_best_uci(labels: Mapping[str, Any], fen: str) -> str | None:
    label = labels.get(fen)
    if not isinstance(label, Mapping):
        return None
    scores = label.get("scores")
    if not isinstance(scores, Mapping) or not scores:
        return None
    return max(scores, key=scores.__getitem__)


def new_rating_bucket() -> dict[str, Any]:
    return {
        "top1_changed": 0,
        "labels_hit": 0,
        "labels_scored": 0,
        "position_max_abs_deltas": [],
        "all_legal_abs_deltas": [],
    }


def finish_bucket(bucket: dict[str, Any], positions: int) -> dict[str, Any]:
    position_deltas = bucket.pop("position_max_abs_deltas")
    all_deltas = bucket.pop("all_legal_abs_deltas")
    scored = bucket.pop("labels_scored")
    hits = bucket.pop("labels_hit")
    return {
        "top1_changed": bucket["top1_changed"],
        "top1_changed_pct": bucket["top1_changed"] / positions * 100.0,
        "mean_position_max_abs_delta": statistics.fmean(position_deltas),
        "max_abs_delta": max(position_deltas),
        "mean_abs_delta_all_legal": statistics.fmean(all_deltas),
        "fixed_label_agreement": hits / scored if scored else None,
        "fixed_labels_scored": scored,
        "position_max_abs_deltas": position_deltas,
    }


def run_sweep(
    torch: Any,
    model: Any,
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    labels: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import chess
        from unarchitectured_metal_position_encoding import encode_position
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"missing dependency: {exc.name}; install it on the trusted analysis host"
        ) from exc

    per_rating = {rating: new_rating_bucket() for rating in args.ratings}
    kind_bucket = new_rating_bucket()

    with torch.inference_mode():
        for position_index, record in enumerate(rows):
            try:
                board = chess.Board(record["fen"])
                if not board.is_valid():
                    raise ValueError("invalid chess position")
                encoded = encode_position(board)
                batch, legal_count = make_batch(
                    torch, encoded, config, args.ratings,
                    [args.policy_kind] * len(args.ratings), args.time_class,
                )
                logits = model(batch)["logits"][:, :legal_count].detach().cpu().tolist()
            except Exception as exc:
                raise RuntimeError(
                    f"cannot analyse corpus position {position_index} ({record['fen']!r}): {exc}"
                ) from exc

            base_logits = logits[0]
            base_top = max(range(legal_count), key=base_logits.__getitem__)
            moves = encoded["legal_moves"]
            best_uci = fixed_label_best_uci(labels, record["fen"])
            for row_index, rating in enumerate(args.ratings):
                rating_logits = logits[row_index]
                top = max(range(legal_count), key=rating_logits.__getitem__)
                deltas = [abs(value - base) for value, base in zip(rating_logits, base_logits)]
                bucket = per_rating[rating]
                bucket["top1_changed"] += int(top != base_top)
                bucket["position_max_abs_deltas"].append(max(deltas))
                bucket["all_legal_abs_deltas"].extend(deltas)
                if best_uci is not None:
                    bucket["labels_hit"] += int(moves[top].uci() == best_uci)
                    bucket["labels_scored"] += 1

            kind_batch, _ = make_batch(
                torch, encoded, config, [1500, 1500], [0, 1], args.time_class
            )
            kind_logits = model(kind_batch)["logits"][:, :legal_count].detach().cpu().tolist()
            human_top = max(range(legal_count), key=kind_logits[0].__getitem__)
            guide_top = max(range(legal_count), key=kind_logits[1].__getitem__)
            kind_deltas = [abs(human - guide) for human, guide in zip(*kind_logits)]
            kind_bucket["top1_changed"] += int(human_top != guide_top)
            kind_bucket["position_max_abs_deltas"].append(max(kind_deltas))
            kind_bucket["all_legal_abs_deltas"].extend(kind_deltas)
            if best_uci is not None:
                kind_bucket["labels_hit"] += int(moves[guide_top].uci() == best_uci)
                kind_bucket["labels_scored"] += 1

    rating_metrics = {
        str(rating): finish_bucket(per_rating[rating], len(rows))
        for rating in args.ratings
    }
    policy_kind_metrics = finish_bucket(kind_bucket, len(rows))
    policy_kind_metrics.update({
        "rating": 1500,
        "human_policy_kind": 0,
        "guide_policy_kind": 1,
        "fixed_label_agreement_for_guide": policy_kind_metrics.pop("fixed_label_agreement"),
    })
    return rating_metrics, policy_kind_metrics


def render_summary(report: Mapping[str, Any]) -> str:
    lines = [
        f"positions: {report['positions_used']}   baseline rating: {report['baseline_rating']}",
        "",
        f"{'rating':>7s} {'top1 changed':>13s} {'mean position max |dlogit|':>27s} {'max |dlogit|':>13s}",
        "-" * 72,
    ]
    for rating, metrics in report["rating_metrics"].items():
        lines.append(
            f"{int(rating):7d} {metrics['top1_changed']:6d} "
            f"({metrics['top1_changed_pct']:4.1f}%) "
            f"{metrics['mean_position_max_abs_delta']:27.6f} "
            f"{metrics['max_abs_delta']:13.6f}"
        )
    policy = report["policy_kind_comparison"]
    lines.append(
        "\nPOLICY_HUMAN vs POLICY_GUIDE at rating 1500: "
        f"top-1 differs in {policy['top1_changed']}/{report['positions_used']} "
        f"({policy['top1_changed_pct']:.1f}%)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    validation_error = validate_paths(args)
    if validation_error is not None:
        return validation_error
    runtime_error = preflight_runtime_package(args.oracle)
    if runtime_error is not None:
        return runtime_error

    torch, loaded, load_error = load_legacy_oracle(args.oracle)
    if load_error is not None:
        return load_error
    checkpoint, model = loaded

    try:
        rows = load_corpus(args.corpus, args.limit)
        labels = load_labels(args.labels)
        rating_metrics, policy_kind_metrics = run_sweep(
            torch, model, checkpoint["config"], rows, labels, args
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return fail(f"oracle rating-conditioning analysis failed: {exc}")

    report: dict[str, Any] = {
        "schema": "UNARCHV1_ORACLE_RATING_CONDITIONING_V1",
        "schema_version": 1,
        "status": "completed",
        "oracle": {
            "path": str(args.oracle),
            "sha256": file_sha256(args.oracle),
            "format": checkpoint["format"],
        },
        "corpus": {
            "path": str(args.corpus),
            "sha256": file_sha256(args.corpus),
        },
        "labels": (
            {"path": str(args.labels), "sha256": file_sha256(args.labels)}
            if args.labels is not None else None
        ),
        "positions_requested": args.limit,
        "positions_used": len(rows),
        "ratings": args.ratings,
        "baseline_rating": args.ratings[0],
        "fixed_inputs": {
            "history": "all-zero encoded move history",
            "history_shape_per_position": [len(args.ratings), checkpoint["config"]["history_plies"]],
            "history_len": 0,
            "time_class": args.time_class,
            "policy_kind": args.policy_kind,
            "legal_action_slots": MAX_LEGAL_ACTIONS,
        },
        "rating_metrics": rating_metrics,
        "policy_kind_comparison": policy_kind_metrics,
    }
    print(render_summary(report))
    if args.json is not None:
        try:
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            return fail(f"unable to write JSON report: {args.json}: {exc}")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
