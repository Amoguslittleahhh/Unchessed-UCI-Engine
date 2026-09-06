#!/usr/bin/env python3
"""Tag calibration positions with tactical themes, and break policy accuracy
down by theme.

Motivation
----------

`Grounded Chess Reasoning in Language Models via Master Distillation`
(arXiv:2603.20510) attributes much of its result to **theme-balanced data
sampling for comprehensive tactical coverage** -- not to model scale. The
underlying claim generalises past LLMs: a single aggregate accuracy number
hides systematic blindness to particular tactical motifs, because motifs are
not uniformly distributed in natural game positions. A model can look fine on
average while being reliably bad at, say, deflection.

Our calibration corpus is balanced by **phase** (200 opening / 200 middlegame
/ 200 endgame) and by nothing else. Nothing in this repository tags themes, so
the headline numbers -- top-1 0.2683, mean first-move regret 146cp -- are
averages over an uncontrolled motif mix. This tool asks whether those averages
are hiding anything.

What it does
------------

1. **Tags each position with tactical themes** using the teacher's own
   MultiPV scores plus `python-chess` board logic. Themes are derived from the
   position and the teacher's best move, never from the model being evaluated,
   so the tagging cannot be contaminated by what it is used to judge.

2. **Reports policy accuracy per theme**, so a systematic weakness shows up
   instead of averaging away.

Theme definitions are deliberately mechanical and conservative -- each is a
property checkable from the board, not a subjective label:

  hanging_piece      teacher's best move captures an undefended piece
  capture            teacher's best move is any capture
  check              teacher's best move gives check
  promotion          teacher's best move promotes
  quiet              teacher's best move is none of the above
  fork               after the best move, the moved piece attacks >=2 valuable
                     enemy pieces
  mate_available     a forced mate exists (teacher score at mate range)
  only_good_move     the best move is >=150cp better than the second best
  many_good_moves    >=3 moves within 20cp of best
  endgame_technique  <=6 non-king pieces on the board

`only_good_move` is the one that matters most for search: it marks positions
where ordering genuinely decides the outcome, because everything else loses
material.

Usage
-----
    python3 tools/tag_calibration_themes.py \\
        artifacts/unarchitectured-metal-calibration-corpus.jsonl \\
        artifacts/unarchitectured-metal-calibration-labels.json \\
        --package artifacts/unarchitectured-metal-final.unmetal

Without `--package` it only tags and reports theme coverage (no torch needed).
With it, it additionally scores the real checkpoint per theme.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import chess  # noqa: E402

MATE_CLAMP_CP = 2000
ONLY_GOOD_MOVE_CP = 150
CLOSE_MOVE_CP = 20
VALUABLE = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING}


def clamp_cp(v: int) -> float:
    return float(max(-MATE_CLAMP_CP, min(MATE_CLAMP_CP, v)))


def classify(board: chess.Board, best_uci: str, scores: dict, saw_mate: bool):
    """Return the set of themes for this position.

    Everything is derived from the board and the *teacher's* choice, so the
    tagging is independent of the model under evaluation.
    """
    themes = set()
    try:
        best = chess.Move.from_uci(best_uci)
    except ValueError:
        return themes
    if best not in board.legal_moves:
        return themes

    is_capture = board.is_capture(best)
    gives_check = board.gives_check(best)

    if is_capture:
        themes.add("capture")
        victim_sq = best.to_square
        if not board.is_en_passant(best):
            # Undefended victim => hanging material was on offer.
            if not board.attackers(not board.turn, victim_sq):
                themes.add("hanging_piece")
    if gives_check:
        themes.add("check")
    if best.promotion:
        themes.add("promotion")
    if not (is_capture or gives_check or best.promotion):
        themes.add("quiet")

    if saw_mate:
        themes.add("mate_available")

    # Fork: after the move, the moved piece attacks two or more valuable
    # enemy pieces at once.
    after = board.copy()
    after.push(best)
    attacked = 0
    for sq in after.attacks(best.to_square):
        piece = after.piece_at(sq)
        if piece and piece.color != board.turn and piece.piece_type in VALUABLE:
            attacked += 1
    if attacked >= 2:
        themes.add("fork")

    # Decisiveness of the position, from the teacher's own score spread.
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) >= 2:
        if clamp_cp(ordered[0]) - clamp_cp(ordered[1]) >= ONLY_GOOD_MOVE_CP:
            themes.add("only_good_move")
        close = sum(
            1 for s in ordered if clamp_cp(ordered[0]) - clamp_cp(s) <= CLOSE_MOVE_CP
        )
        if close >= 3:
            themes.add("many_good_moves")

    non_king = sum(
        1 for _, p in board.piece_map().items() if p.piece_type != chess.KING
    )
    if non_king <= 6:
        themes.add("endgame_technique")

    return themes


def build_batch(board: chess.Board, rating: int, time_class: int, policy_kind: int):
    import torch

    from unarchitectured_metal_position_encoding import encode_position

    enc = encode_position(board)
    actions = list(enc["legal_actions"])
    n = len(actions)
    padded = actions + [0xFFFF] * (218 - n)
    batch = {
        "pieces": torch.tensor([enc["pieces"]], dtype=torch.long),
        "castling": torch.tensor([enc["castling"]], dtype=torch.long),
        "ep_file": torch.tensor([enc["ep_file"]], dtype=torch.long),
        "halfmove_bucket": torch.tensor([enc["halfmove_bucket"]], dtype=torch.long),
        "rating": torch.tensor([rating], dtype=torch.long),
        "time_class": torch.tensor([time_class], dtype=torch.long),
        "policy_kind": torch.tensor([policy_kind], dtype=torch.long),
        "safe_actions": torch.tensor([padded], dtype=torch.long),
        "legal_mask": torch.tensor([[i < n for i in range(218)]]),
    }
    return batch, enc["legal_moves"], n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--package", type=Path, help="score the checkpoint per theme")
    ap.add_argument("--rating", type=int, default=2700)
    ap.add_argument("--time-class", type=int, default=2)
    ap.add_argument("--policy-kind", type=int, default=1)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for p in (args.corpus, args.labels):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    labels = json.loads(args.labels.read_text())
    rows = []
    with args.corpus.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if "fen" in rec:
                rows.append(rec)

    weights = config = None
    forward = None
    if args.package:
        if not args.package.is_file():
            print(f"missing: {args.package}", file=sys.stderr)
            return 2
        from reference_forward_unarchitectured_metal import forward as fwd
        from reference_forward_unarchitectured_metal import read_package

        forward = fwd
        weights = read_package(args.package)
        config = {
            "d_model": 256,
            "heads": 8,
            "history_width": 32,
            "policy_adapter_rank": 16,
        }

    counts = Counter()
    per_theme = defaultdict(lambda: {"hits": 0, "n": 0, "regret": []})
    tagged = []
    scored = 0

    for rec in rows:
        fen = rec["fen"]
        label = labels.get(fen)
        if not label:
            continue
        scores = label["scores"]
        board = chess.Board(fen)
        best_uci = max(scores, key=lambda k: scores[k])
        themes = classify(board, best_uci, scores, label.get("saw_mate", False))
        counts.update(themes)
        tagged.append({"fen": fen, "phase": rec.get("phase"), "themes": sorted(themes)})

        if forward is None:
            continue

        batch, legal_moves, n = build_batch(
            board, args.rating, args.time_class, args.policy_kind
        )
        if not all(m.uci() in scores for m in legal_moves):
            continue
        out = forward(weights, batch, config, layers=8, width=256)
        logits = out["logits"][0][:n].tolist()
        top = max(range(n), key=lambda i: logits[i])
        chosen = legal_moves[top].uci()
        regret = clamp_cp(scores[best_uci]) - clamp_cp(scores[chosen])
        hit = int(chosen == best_uci)
        scored += 1
        for t in themes:
            per_theme[t]["hits"] += hit
            per_theme[t]["n"] += 1
            per_theme[t]["regret"].append(regret)

    print(f"positions tagged: {len(tagged)}\n")
    print(f"{'theme':22s} {'count':>6s} {'share':>7s}")
    print("-" * 38)
    for theme, c in counts.most_common():
        print(f"{theme:22s} {c:6d} {c/len(tagged)*100:6.1f}%")

    report = {
        "positions_tagged": len(tagged),
        "theme_counts": dict(counts),
        "tagged": tagged,
    }

    if forward is not None and scored:
        print(f"\npolicy accuracy by theme ({scored} scored):\n")
        print(f"{'theme':22s} {'n':>5s} {'top1':>7s} {'mean regret':>12s}")
        print("-" * 50)
        overall_hits = sum(1 for t in tagged if False)  # placeholder, computed below
        rows_out = {}
        for theme in sorted(per_theme, key=lambda t: -per_theme[t]["n"]):
            d = per_theme[theme]
            acc = d["hits"] / d["n"]
            reg = statistics.fmean(d["regret"])
            rows_out[theme] = {
                "positions": d["n"],
                "top1_accuracy": acc,
                "mean_regret_cp": reg,
            }
            print(f"{theme:22s} {d['n']:5d} {acc:7.4f} {reg:12.1f}")
        report["per_theme"] = rows_out
        report["positions_scored"] = scored
        del overall_hits

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
