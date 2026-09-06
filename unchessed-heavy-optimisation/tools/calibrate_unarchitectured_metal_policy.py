#!/usr/bin/env python3
"""Provenance-disjoint calibration of Unarchitectured Metal against a real teacher.

This is round 6 step 1. The only calibration numbers that existed before this
came from eight hand-picked positions labelled by this engine's own HCE search
at depth 4 (top-1 0.50-0.625, barely above chance). Those numbers cannot inform
a deployment threshold: the sample is tiny and the "teacher" is the same search
the model would be hinting.

This harness instead:

  * reads a source-disjoint over-the-board corpus (see
    `tools/build_unarchitectured_metal_calibration_corpus.py`);
  * labels every legal root move with a real independent oracle (Stockfish,
    via UCI `MultiPV` over the full legal move list); and
  * reports policy top-1/top-3, mean teacher-best rank, regret MAE, and WDL
    Brier per elastic exit, using the *same* metric definitions as the existing
    Rust smoke calibration (`benchmark_fixture_disjoint_exit_calibration`) so
    the two are directly comparable.

Crucially it also reports a **random-ordering baseline** for the same positions.
Top-1 accuracy alone is not interpretable: it depends on how many legal moves
each position has. Without the baseline there is no way to tell "real signal"
from "looks high because the endgame positions have few legal moves".

The forward pass is the repository's own validated PyTorch reference
(`tools/reference_forward_unarchitectured_metal.py`), which is cross-checked
against the Rust runtime at 5e-3 — so these numbers describe the shipped model,
not a reimplementation of it.

Usage:
  python3 tools/calibrate_unarchitectured_metal_policy.py \
      --package artifacts/unarchitectured-metal-final.unmetal \
      --corpus artifacts/unarchitectured-metal-calibration-corpus.jsonl \
      --engine /path/to/stockfish \
      --report artifacts/unarchitectured-metal-calibration-report.json

Requires NumPy, PyTorch, python-chess, and a UCI teacher engine binary.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Mirrors POLICY_GUIDE in unarchitectured_metal_position_encoding; duplicated as a
# literal so `--help` works before the chess/torch imports below.
_DEFAULT_POLICY_KIND = 1


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, default=Path("artifacts/unarchitectured-metal-final.unmetal"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=None, help="UCI teacher engine binary")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="teacher-label cache; reused if present, written if absent",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--nodes", type=int, default=400000, help="teacher nodes per position")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="only label the first N positions")
    parser.add_argument("--rating", type=int, default=2700)
    parser.add_argument("--time-class", type=int, default=2)
    parser.add_argument("--policy-kind", type=int, default=_DEFAULT_POLICY_KIND)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    argument_parser().parse_args()

import chess  # noqa: E402
import chess.engine  # noqa: E402
import torch  # noqa: E402

from unarchitectured_metal_position_encoding import (  # noqa: E402
    MAX_ACTIONS,
    POLICY_GUIDE,
    encode_position,
)

import importlib.util  # noqa: E402

_REFERENCE_PATH = TOOLS / "reference_forward_unarchitectured_metal.py"


def load_reference_module():
    """Import the repository's validated Python reference forward pass."""
    spec = importlib.util.spec_from_file_location("_unmetal_reference", _REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    # The reference exits early on -h/--help, so hide our argv while importing.
    saved_argv = sys.argv
    sys.argv = [str(_REFERENCE_PATH)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv
    return module


EXITS = ((2, 128), (4, 192), (8, 256))

# Matches the Rust smoke calibration: regret targets are centipawn gaps scaled
# by 400, and WDL outcome buckets use a +/-50cp deadband around the teacher's
# best score.
REGRET_SCALE = 400.0
WDL_DEADBAND = 50


def build_batch(board: chess.Board, rating: int, time_class: int, policy_kind: int) -> tuple[dict, list]:
    encoded = encode_position(board)
    actions = list(encoded["legal_actions"])
    legal_count = len(actions)
    padded = actions + [0xFFFF] * (MAX_ACTIONS - legal_count)
    batch = {
        "pieces": torch.tensor([encoded["pieces"]], dtype=torch.long),
        "castling": torch.tensor([encoded["castling"]], dtype=torch.long),
        "ep_file": torch.tensor([encoded["ep_file"]], dtype=torch.long),
        "halfmove_bucket": torch.tensor([encoded["halfmove_bucket"]], dtype=torch.long),
        "rating": torch.tensor([rating], dtype=torch.long),
        "time_class": torch.tensor([time_class], dtype=torch.long),
        "policy_kind": torch.tensor([policy_kind], dtype=torch.long),
        "safe_actions": torch.tensor([padded], dtype=torch.long),
        "legal_mask": torch.tensor([[i < legal_count for i in range(MAX_ACTIONS)]]),
    }
    return batch, encoded["legal_moves"]


# Piece values for the cheap static ordering baseline (centipawns).
_MVV_LVA_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def heuristic_move_score(board: chess.Board, move: chess.Move) -> int:
    """A nearly-free static ordering score: MVV-LVA, promotions, checks.

    This exists to answer the question that actually decides whether the
    neural hint is worth an SPRT slot: does an ~11ms transformer forward pass
    order root moves better than a heuristic any engine can compute for free?
    """
    score = 0
    victim = board.piece_type_at(move.to_square)
    if board.is_en_passant(move):
        victim = chess.PAWN
    if victim is not None:
        attacker = board.piece_type_at(move.from_square)
        score += 10 * _MVV_LVA_VALUE[victim] - _MVV_LVA_VALUE.get(attacker, 0)
    if move.promotion:
        score += _MVV_LVA_VALUE[move.promotion]
    if board.gives_check(move):
        score += 50
    return score


def teacher_label(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> tuple[dict, int, bool] | None:
    """Score every legal root move with the teacher. Returns (scores, best, mate)."""
    legal_count = board.legal_moves.count()
    try:
        infos = engine.analyse(
            board, chess.engine.Limit(nodes=nodes), multipv=legal_count
        )
    except chess.engine.EngineError:
        return None

    scores: dict[str, int] = {}
    best_score = None
    best_uci = None
    saw_mate = False
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        score = info["score"].pov(board.turn)
        if score.is_mate():
            saw_mate = True
            centipawns = score.score(mate_score=100000)
        else:
            centipawns = score.score()
        if centipawns is None:
            continue
        uci = pv[0].uci()
        scores[uci] = centipawns
        if best_score is None or centipawns > best_score:
            best_score = centipawns
            best_uci = uci
    if best_uci is None or len(scores) < legal_count:
        return None
    return scores, best_score, saw_mate


def wdl_bucket(best_score: int) -> int:
    if best_score > WDL_DEADBAND:
        return 0
    if best_score < -WDL_DEADBAND:
        return 2
    return 1


def mcnemar_pvalue(a_hits: list[int], b_hits: list[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired top-1 hit/miss vectors.

    Answers "is the full exit's top-1 advantage over the cheap static
    heuristic real, or could the same positions have produced it by chance?"
    """
    only_a = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    only_b = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    discordant = only_a + only_b
    if discordant == 0:
        return only_a, only_b, 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(0, min(only_a, only_b) + 1)
    )
    pvalue = min(1.0, 2.0 * tail / (2 ** discordant))
    return only_a, only_b, pvalue


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval, so top-1 is reported with real uncertainty."""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main() -> int:
    args = argument_parser().parse_args()

    for path in (args.package, args.corpus):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    label_cache: dict[str, dict] = {}
    if args.labels and args.labels.exists():
        label_cache = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.engine is None and not label_cache:
        print("either --engine or an existing --labels cache is required", file=sys.stderr)
        return 2
    if args.engine is not None and not args.engine.exists():
        print(f"missing input: {args.engine}", file=sys.stderr)
        return 2

    reference = load_reference_module()
    weights = reference.read_package(str(args.package))
    config = {"d_model": 256, "heads": 8, "history_width": 32, "policy_adapter_rank": 16}

    records = []
    manifest = {}
    with args.corpus.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if "manifest" in entry:
                manifest = entry["manifest"]
                continue
            records.append(entry)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("empty corpus", file=sys.stderr)
        return 1

    engine = None
    teacher_id = label_cache.get("__teacher__", {}).get("name", "cached") if label_cache else "unknown"
    if args.engine is not None:
        engine = chess.engine.SimpleEngine.popen_uci(str(args.engine))
        engine.configure({"Threads": args.threads, "Hash": args.hash})
        teacher_id = engine.id.get("name", "unknown")

    stats = {
        exit_key: {
            "positions": 0,
            "top1": 0,
            "top3": 0,
            "rank_sum": 0,
            "reciprocal_rank": 0.0,
            "regret_abs_error": 0.0,
            "regret_samples": 0,
            "wdl_brier": 0.0,
            "centipawn_loss": [],
            "by_phase": {},
        }
        for exit_key in EXITS
    }
    random_top1 = 0.0
    random_top3 = 0.0
    random_centipawn_loss: list[float] = []
    heuristic_stats = {
        "top1": 0,
        "top3": 0,
        "rank_sum": 0,
        "centipawn_loss": [],
        "hits": [],
    }
    full_exit_hits: list[int] = []
    legal_counts: list[int] = []
    labelled = 0
    skipped = 0
    mate_positions = 0
    started = time.time()

    for index, record in enumerate(records):
        board = chess.Board(record["fen"])
        cached = label_cache.get(record["fen"])
        if cached is not None:
            teacher_scores = cached["scores"]
            best_score = cached["best_score"]
            saw_mate = cached["saw_mate"]
        else:
            if engine is None:
                skipped += 1
                continue
            label = teacher_label(engine, board, args.nodes)
            if label is None:
                skipped += 1
                continue
            teacher_scores, best_score, saw_mate = label
            label_cache[record["fen"]] = {
                "scores": teacher_scores,
                "best_score": best_score,
                "saw_mate": saw_mate,
            }
        if saw_mate:
            mate_positions += 1
        labelled += 1

        batch, legal_moves = build_batch(board, args.rating, args.time_class, args.policy_kind)
        legal_count = len(legal_moves)
        legal_counts.append(legal_count)
        phase = record.get("phase", "unknown")

        # A uniformly random ordering is the honest floor for these metrics.
        random_top1 += 1.0 / legal_count
        random_top3 += min(3, legal_count) / legal_count
        if not saw_mate:
            # Expected centipawn loss of picking a legal move uniformly at
            # random, for the same position set.
            gaps = [max(0, best_score - value) for value in teacher_scores.values()]
            random_centipawn_loss.append(sum(gaps) / len(gaps))

        best_uci = max(teacher_scores, key=lambda uci: teacher_scores[uci])
        outcome = wdl_bucket(best_score)
        mate_involved = saw_mate

        # Cheap static-heuristic ordering over the same legal move list.
        heuristic_order = sorted(
            range(legal_count),
            key=lambda i: heuristic_move_score(board, legal_moves[i]),
            reverse=True,
        )
        heuristic_teacher_index = next(
            i for i, move in enumerate(legal_moves) if move.uci() == best_uci
        )
        heuristic_rank = heuristic_order.index(heuristic_teacher_index)
        heuristic_stats["hits"].append(int(heuristic_rank == 0))
        heuristic_stats["top1"] += int(heuristic_rank == 0)
        heuristic_stats["top3"] += int(heuristic_rank < 3)
        heuristic_stats["rank_sum"] += heuristic_rank + 1
        if not mate_involved:
            heuristic_choice = teacher_scores.get(legal_moves[heuristic_order[0]].uci())
            if heuristic_choice is not None:
                heuristic_stats["centipawn_loss"].append(max(0, best_score - heuristic_choice))

        for exit_key in EXITS:
            layers, width = exit_key
            with torch.no_grad():
                output = reference.forward(weights, batch, config, layers=layers, width=width)
            logits = output["logits"][0][:legal_count].tolist()
            order = sorted(range(legal_count), key=lambda i: logits[i], reverse=True)
            teacher_index = next(
                i for i, move in enumerate(legal_moves) if move.uci() == best_uci
            )
            rank = order.index(teacher_index)

            if exit_key == (8, 256):
                full_exit_hits.append(int(rank == 0))
            bucket = stats[exit_key]
            bucket["positions"] += 1
            bucket["top1"] += int(rank == 0)
            bucket["top3"] += int(rank < 3)
            bucket["rank_sum"] += rank + 1
            bucket["reciprocal_rank"] += 1.0 / (rank + 1)

            phase_bucket = bucket["by_phase"].setdefault(
                phase, {"positions": 0, "top1": 0, "top3": 0}
            )
            phase_bucket["positions"] += 1
            phase_bucket["top1"] += int(rank == 0)
            phase_bucket["top3"] += int(rank < 3)

            # How much the model's own first choice actually gives up, in
            # centipawns, against the teacher. Top-1 accuracy treats "second
            # best by 3cp" and "blunders a rook" identically; this does not.
            # Mate-scored positions are excluded (a mate score is not a cp gap).
            if not mate_involved:
                chosen_uci = legal_moves[order[0]].uci()
                chosen_score = teacher_scores.get(chosen_uci)
                if chosen_score is not None:
                    bucket["centipawn_loss"].append(max(0, best_score - chosen_score))

            # Regret MAE, skipping mate-scored positions exactly as the Rust
            # smoke harness does (a mate score is not a centipawn gap).
            if not mate_involved:
                regret_mean = output["regret_mean"][0][:legal_count].tolist()
                for move_index, move in enumerate(legal_moves):
                    score = teacher_scores.get(move.uci())
                    if score is None:
                        continue
                    target = max(0, best_score - score) / REGRET_SCALE
                    bucket["regret_abs_error"] += abs(regret_mean[move_index] - target)
                    bucket["regret_samples"] += 1

            evidence = output["evidence"][0].tolist()
            evidence_sum = sum(evidence) + 3.0
            probabilities = [(value + 1.0) / evidence_sum for value in evidence]
            bucket["wdl_brier"] += sum(
                (probability - (1.0 if i == outcome else 0.0)) ** 2
                for i, probability in enumerate(probabilities)
            )

        if args.progress_every and (index + 1) % args.progress_every == 0:
            rate = (time.time() - started) / max(1, index + 1)
            print(
                f"  labelled {index + 1}/{len(records)} ({rate:.2f}s/pos)",
                file=sys.stderr,
                flush=True,
            )

    if engine is not None:
        engine.quit()
    if args.labels:
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        if teacher_id not in ("cached", "unknown"):
            label_cache["__teacher__"] = {"name": teacher_id, "nodes": args.nodes}
        args.labels.write_text(json.dumps(label_cache), encoding="utf-8")

    if labelled == 0:
        print("no positions labelled", file=sys.stderr)
        return 1

    report = {
        "teacher": teacher_id,
        "teacher_nodes_per_position": args.nodes,
        "package": str(args.package),
        "corpus": str(args.corpus),
        "corpus_manifest": manifest,
        "positions_labelled": labelled,
        "positions_skipped": skipped,
        "positions_with_mate_scores": mate_positions,
        "mean_legal_moves": statistics.mean(legal_counts),
        "median_legal_moves": statistics.median(legal_counts),
        "random_baseline": {
            "top1": random_top1 / labelled,
            "top3": random_top3 / labelled,
            "centipawn_loss_mean": (
                statistics.mean(random_centipawn_loss) if random_centipawn_loss else None
            ),
        },
        "heuristic_baseline": {
            "description": "MVV-LVA + promotion + check static ordering",
            "top1": heuristic_stats["top1"] / labelled,
            "top3": heuristic_stats["top3"] / labelled,
            "mean_teacher_best_rank": heuristic_stats["rank_sum"] / labelled,
            "centipawn_loss_mean": (
                statistics.mean(heuristic_stats["centipawn_loss"])
                if heuristic_stats["centipawn_loss"]
                else None
            ),
        },
        "full_exit_vs_heuristic_mcnemar": (
            lambda pair: {
                "full_exit_only_correct": pair[0],
                "heuristic_only_correct": pair[1],
                "p_value": pair[2],
            }
        )(mcnemar_pvalue(full_exit_hits, heuristic_stats["hits"])),
        "rating": args.rating,
        "time_class": args.time_class,
        "policy_kind": args.policy_kind,
        "elapsed_seconds": time.time() - started,
        "exits": {},
    }

    for exit_key in EXITS:
        bucket = stats[exit_key]
        positions = bucket["positions"]
        low, high = wilson_interval(bucket["top1"], positions)
        losses = bucket["centipawn_loss"]
        losses_sorted = sorted(losses)
        report["exits"][f"{exit_key[0]}/{exit_key[1]}"] = {
            "top1_centipawn_loss_mean": statistics.mean(losses) if losses else None,
            "top1_centipawn_loss_median": statistics.median(losses) if losses else None,
            "top1_centipawn_loss_p90": (
                losses_sorted[min(len(losses_sorted) - 1, int(0.9 * len(losses_sorted)))]
                if losses
                else None
            ),
            "top1_centipawn_loss_samples": len(losses),
            "positions": positions,
            "top1": bucket["top1"] / positions,
            "top1_ci95": [low, high],
            "top3": bucket["top3"] / positions,
            "mean_teacher_best_rank": bucket["rank_sum"] / positions,
            "mean_reciprocal_rank": bucket["reciprocal_rank"] / positions,
            "regret_mae": bucket["regret_abs_error"] / max(1, bucket["regret_samples"]),
            "regret_samples": bucket["regret_samples"],
            "wdl_brier": bucket["wdl_brier"] / positions,
            "by_phase": {
                phase: {
                    "positions": values["positions"],
                    "top1": values["top1"] / values["positions"],
                    "top3": values["top3"] / values["positions"],
                }
                for phase, values in sorted(bucket["by_phase"].items())
            },
        }

    print(json.dumps(report, indent=2))

    baseline = report["random_baseline"]
    print()
    print(f"teacher: {teacher_id} @ {args.nodes} nodes/position")
    print(f"positions: {labelled}  mean legal moves: {report['mean_legal_moves']:.2f}")
    random_loss = baseline["centipawn_loss_mean"]
    random_loss_text = f" cp_loss={random_loss:.1f}" if random_loss is not None else ""
    print(
        f"random-ordering baseline: top1={baseline['top1']:.4f} "
        f"top3={baseline['top3']:.4f}{random_loss_text}"
    )
    heuristic = report["heuristic_baseline"]
    heuristic_loss = heuristic["centipawn_loss_mean"]
    heuristic_loss_text = f" cp_loss={heuristic_loss:.1f}" if heuristic_loss is not None else ""
    print(
        f"static-heuristic baseline (MVV-LVA/promo/check): "
        f"top1={heuristic['top1']:.4f} top3={heuristic['top3']:.4f}"
        f"{heuristic_loss_text}"
    )
    mcnemar = report["full_exit_vs_heuristic_mcnemar"]
    print(
        f"8/256 vs heuristic (paired McNemar): "
        f"model-only-correct={mcnemar['full_exit_only_correct']} "
        f"heuristic-only-correct={mcnemar['heuristic_only_correct']} "
        f"p={mcnemar['p_value']:.3g}"
    )
    print()
    header = (
        f"{'exit':>8} {'top1':>8} {'top1 95% CI':>18} {'top3':>8} {'meanrank':>9} "
        f"{'MRR':>7} {'cploss':>8} {'cpP90':>7} {'regretMAE':>10} {'Brier':>8}"
    )
    print(header)
    for name, values in report["exits"].items():
        low, high = values["top1_ci95"]
        loss = values["top1_centipawn_loss_mean"]
        p90 = values["top1_centipawn_loss_p90"]
        loss_text = f"{loss:>8.1f}" if loss is not None else f"{'n/a':>8}"
        p90_text = f"{p90:>7.0f}" if p90 is not None else f"{'n/a':>7}"
        print(
            f"{name:>8} {values['top1']:>8.4f} {f'[{low:.3f}, {high:.3f}]':>18} "
            f"{values['top3']:>8.4f} {values['mean_teacher_best_rank']:>9.3f} "
            f"{values['mean_reciprocal_rank']:>7.3f} {loss_text} {p90_text} "
            f"{values['regret_mae']:>10.4f} {values['wdl_brier']:>8.4f}"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
