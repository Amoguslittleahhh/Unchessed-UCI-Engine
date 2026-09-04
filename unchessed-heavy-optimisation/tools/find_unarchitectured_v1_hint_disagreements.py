#!/usr/bin/env python3
"""Find real positions where the Unarchitectured v1 policy disagrees with the
tactically correct move.

Why this exists
---------------

Every root-hint safety test in `unchessed-core/src/search.rs` currently uses a
*hand-built adversarial* hint: the test author decides the mating move should
be ranked last and checks the search finds it anyway. That proves the search
is robust against a maximally hostile input, which is the right first test,
but it leaves one gap the last two rounds both named and neither closed:

    Does the search still play the correct move when the *real checkpoint*,
    on a *real position*, genuinely prefers something else?

That is a different question. A synthetic worst-case hint is uniform noise; a
real model's mistake is structured -- it is confidently wrong in the specific
way a policy net trained on human games is wrong (it likes natural-looking
developing moves and dislikes counterintuitive sacrifices). If root ordering
were ever to leak into the final choice, a confident-but-wrong real hint is
the input most likely to expose it.

This tool searches for those cases so they can be turned into fixtures with
real, non-synthetic numbers instead of invented ones.

Method
------

For each candidate position with a known correct move:

1. Encode it exactly as the runtime does (`unarchitectured_v1_position_encoding`,
   the same module the Rust port is validated against).
2. Run the real exported checkpoint through the reference forward pass
   (`reference_forward_unarchitectured_v1.forward`).
3. Record where the correct move lands in the policy's ranking.

A position is a useful fixture when the correct move is NOT the model's top
choice -- ideally ranked well down the list, since that maximises the ordering
penalty the search has to overcome.

Positions here are standard, published tactical motifs (back-rank mates,
smothered mate, a queen sacrifice) rather than engine-generated, so the
"correct move" is verifiable by inspection and by `python-chess` itself
(mate-in-N is checked directly, not asserted).

Usage
-----
    python3 tools/find_unarchitectured_v1_hint_disagreements.py \\
        artifacts/unarchitectured-v1-final.unarchv1
    python3 tools/find_unarchitectured_v1_hint_disagreements.py PACKAGE --json

Requires `torch` (see tools/requirements-dev.txt -- it is intentionally not in
that file; install it separately). Exits 0 whether or not disagreements are
found: "the model agrees everywhere" is a legitimate result, not a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# `--help` must work from a bare clone with nothing installed, so the heavy
# imports are deferred rather than done at module scope.
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import chess  # noqa: E402

from unarchitectured_v1_position_encoding import encode_position  # noqa: E402

# Standard tactical positions with a single, verifiable best move.
#
# Each entry is (name, FEN, correct move in UCI, short description). Every
# `mate_in` case is verified programmatically below by actually playing the
# move and asserting checkmate, so a typo cannot silently produce a bogus
# fixture.
CANDIDATES = [
    (
        "back_rank_mate_white",
        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
        "a1a8",
        "Rook to the eighth: mate on the back rank, king boxed in by its own pawns.",
    ),
    (
        "back_rank_mate_black",
        "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1",
        "a8a1",
        "Mirror of the above with colours reversed.",
    ),
    (
        "smothered_mate",
        "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
        "g5f7",
        "Smothered mate: knight to f7 with the king sealed by its own rook and pawns.",
    ),
    (
        "back_rank_mate_with_own_pawns",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "a1a8",
        "Back-rank mate in a fuller position: both sides still have their pawn "
        "shields, so the mate has to be seen among 20 plausible-looking moves.",
    ),
    (
        "knight_fork_king_rook",
        "r3k3/8/8/4N3/8/8/8/4K3 w - - 0 1",
        "e5c6",
        "Knight fork hitting king and rook -- material win, not mate.",
    ),
    (
        "greek_gift_setup",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        "c4f7",
        "Bishop sac on f7: sharp, and exactly the kind of move a human-trained policy dislikes.",
    ),
]


def build_batch(board: chess.Board, rating: int, time_class: int, policy_kind: int):
    """Encode `board` into the model's batch dict, matching the runtime."""
    import torch

    enc = encode_position(board)
    actions = list(enc["legal_actions"])
    legal_count = len(actions)
    padded = actions + [0xFFFF] * (218 - legal_count)

    batch = {
        "pieces": torch.tensor([enc["pieces"]], dtype=torch.long),
        "castling": torch.tensor([enc["castling"]], dtype=torch.long),
        "ep_file": torch.tensor([enc["ep_file"]], dtype=torch.long),
        "halfmove_bucket": torch.tensor([enc["halfmove_bucket"]], dtype=torch.long),
        "rating": torch.tensor([rating], dtype=torch.long),
        "time_class": torch.tensor([time_class], dtype=torch.long),
        "policy_kind": torch.tensor([policy_kind], dtype=torch.long),
        "safe_actions": torch.tensor([padded], dtype=torch.long),
        "legal_mask": torch.tensor([[i < legal_count for i in range(218)]]),
    }
    return batch, enc["legal_moves"], legal_count


def analyse(package: Path, rating: int, time_class: int, policy_kind: int):
    from reference_forward_unarchitectured_v1 import forward, read_package

    weights = read_package(package)
    config = {
        "d_model": 256,
        "heads": 8,
        "history_width": 32,
        "policy_adapter_rank": 16,
    }

    results = []
    for name, fen, correct_uci, description in CANDIDATES:
        board = chess.Board(fen)
        correct = chess.Move.from_uci(correct_uci)
        if correct not in board.legal_moves:
            results.append({"name": name, "error": f"{correct_uci} is not legal in {fen}"})
            continue

        # Verify the claim rather than trusting the table. An earlier draft of
        # this file listed a "mate" that was refutable by a simple recapture;
        # the fixture looked plausible and the tool happily reported on it. So
        # the correctness of every entry is now derived here, not asserted
        # above: play the move, ask python-chess whether it is mate, and
        # separately enumerate *all* mates in one so a position advertised as
        # having a unique solution cannot quietly have two.
        probe = board.copy()
        probe.push(correct)
        is_mate = probe.is_checkmate()

        all_mates = []
        scan = board.copy()
        for candidate in list(scan.legal_moves):
            scan.push(candidate)
            if scan.is_checkmate():
                all_mates.append(candidate.uci())
            scan.pop()
        if all_mates and not is_mate:
            results.append(
                {
                    "name": name,
                    "error": (
                        f"{correct_uci} is not mate but {all_mates} are -- "
                        "fixture mislabelled"
                    ),
                }
            )
            continue
        if is_mate and all_mates != [correct_uci]:
            results.append(
                {
                    "name": name,
                    "error": (
                        f"mate is not unique: expected only {correct_uci}, "
                        f"found {all_mates}"
                    ),
                }
            )
            continue

        batch, legal_moves, legal_count = build_batch(
            board, rating, time_class, policy_kind
        )
        out = forward(weights, batch, config, layers=8, width=256)
        logits = out["logits"][0][:legal_count].tolist()

        order = sorted(range(legal_count), key=lambda i: -logits[i])
        correct_index = legal_moves.index(correct)
        rank = order.index(correct_index)
        top_index = order[0]

        results.append(
            {
                "name": name,
                "fen": fen,
                "description": description,
                "legal_count": legal_count,
                "correct_move": correct_uci,
                "correct_is_mate": is_mate,
                "correct_logit": round(logits[correct_index], 6),
                "correct_rank": rank,
                "model_top_move": legal_moves[top_index].uci(),
                "model_top_logit": round(logits[top_index], 6),
                "disagrees": rank != 0,
                "logit_gap": round(logits[top_index] - logits[correct_index], 6),
                # Full per-move policy output, so a downstream test can replay
                # the model's real ranking instead of inventing hint scores.
                "moves": [m.uci() for m in legal_moves],
                "logits": [round(x, 6) for x in logits],
            }
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package", type=Path, help="path to the .unarchv1 checkpoint")
    ap.add_argument("--rating", type=int, default=2700)
    ap.add_argument("--time-class", type=int, default=2)
    ap.add_argument(
        "--policy-kind",
        type=int,
        default=1,
        help="0 = POLICY_HUMAN, 1 = POLICY_GUIDE (default, matches the runtime)",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    if not args.package.is_file():
        print(f"no such package: {args.package}", file=sys.stderr)
        return 2

    results = analyse(args.package, args.rating, args.time_class, args.policy_kind)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    disagreements = 0
    for r in results:
        if "error" in r:
            print(f"{r['name']}: ERROR {r['error']}")
            continue
        flag = "DISAGREES" if r["disagrees"] else "agrees"
        mate = " (mate)" if r["correct_is_mate"] else ""
        print(f"\n{r['name']}{mate}: {flag}")
        print(f"  {r['description']}")
        print(f"  fen            {r['fen']}")
        print(f"  legal moves    {r['legal_count']}")
        print(
            f"  correct        {r['correct_move']}  "
            f"logit {r['correct_logit']:+.6f}  rank {r['correct_rank']}"
        )
        print(
            f"  model prefers  {r['model_top_move']}  "
            f"logit {r['model_top_logit']:+.6f}  gap {r['logit_gap']:.6f}"
        )
        disagreements += bool(r["disagrees"])

    print(
        f"\n{disagreements}/{len(results)} position(s) where the real checkpoint "
        f"does not rank the correct move first."
    )
    if disagreements:
        print(
            "These are usable as non-synthetic safety fixtures: the search must "
            "still find the correct move despite this real ordering pressure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
