#!/usr/bin/env python3
"""Maia-3 self-play with random UCI elo limits, one side against the other.

Generates the "mix of maia 3 with random uci elo limits against each
other": every game is played by two Maia-3 instances whose UCI elo
limits are drawn independently and uniformly from [--elo-min,
--elo-max] (default 100..3200) and fixed for the whole game. Each move
is a sample from the model's elo-conditioned policy distribution
(temperature defaults to 1.0 — the same stochastic mechanism the
official Maia-3 UCI uses for human-like play).

The model is the Maia-3 ONNX export that ships with the official
maia-platform-frontend (45.7 MB, `maia3_simplified.onnx`), mirrored at
mcognetta/simple-maia3-inference. It is GPL-3.0 and is NOT committed to
this repo — fetch it with:

    python3 tools/selfplay_elo_mixer.py fetch-model --out /tmp/maia3-onnx

then run:

    python3 tools/selfplay_elo_mixer.py play \
        --model /tmp/maia3-onnx/simple-maia3-inference/simple_maia3_inference/maia3_simplified.onnx \
        --games 200 --seed 42 \
        --pgn data/selfplay/maia3-100-3200.pgn \
        --labels data/selfplay/maia3-100-3200-labels.jsonl

Outputs:
  * PGN with WhiteElo/BlackElo headers set to the drawn elo limits, so
    the games flow through the same level-conditioned labeling pipeline
    as real-human data (tools/build_level_conditioned_moves.py).
  * per-move JSONL labels {game, elo_white, elo_black, fen, move_uci,
    move_ply, side, elo_self, elo_oppo, top1_prob, ldw} — the dual-elo
    label schema of
    docs/research-notes-maia-levels-reverse-engineering.md, plus a
    top1_prob diagnostic per move.

Model interface (single-position export, no history):
  inputs  tokens (N,64,12) one-hot board (mirrored so the side to move
          plays White; white pieces in columns 0-5, black in 6-11),
          elo_self (N,), elo_oppo (N,) as float32
  outputs logits_move (N,4352), logits_value (N,3)
  move indices: base = frm*64+to (flat 64x64, 4096 entries; diagonal
          entries exist but are never legal); rank-7->8 promotions:
          4096 + ((frm-48)*8 + (to-56))*4 + {q:0, r:1, b:2, n:3}.
  The 4352-index scheme was verified entry-by-entry against the
  reference ALL_MOVES_MAIA3 table (test_selfplay_elo_mixer.py).

The coordinate transform is the inverse of chess.Board.mirror() for a
side-to-move-is-White board: rank r -> 9-r, file unchanged (mirror()
flips colors and ranks, not files). The full transform round-trips the
legal-move set exactly (tested on a black position).

The `ldw` label is (loss, draw, win) from the side-to-move's
perspective: the export's value head emits (win, draw, loss) for the
(board's) white, which after mirroring is the side to move.

Honest limits: this is the "simplified" export (single position, no
multi-position history, no clock/time inputs) and move selection is
temperature sampling without the official UCI's one-ply
opponent-response ranking. Both are documented approximations of
`maia3-uci`; the elo conditioning itself is the model's real one.

Dependencies: python-chess, numpy (tools/requirements-dev.txt) and
onnxruntime (CPU) — onnxruntime is deliberately NOT in
requirements-dev.txt (only this generator needs it);
`pip install onnxruntime` into the working venv.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import chess
import numpy as np

MODEL_REPO = "https://github.com/mcognetta/simple-maia3-inference"
MODEL_COMMIT = "05ede11a0ec43c7d1c5d55d16ec86fbe5b6a3fcc"
MODEL_FILE = "simple_maia3_inference/maia3_simplified.onnx"
FILES = "abcdefgh"
PROMO_ORDER = {"q": 0, "r": 1, "b": 2, "n": 3}
# piece_type -> one-hot column: white P,N,B,R,Q,K in 0-5, black in 6-11
PIECE_COL = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
             chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def square_index(s: str) -> int:
    return FILES.index(s[0]) + (int(s[1]) - 1) * 8


def index_square(i: int) -> str:
    return FILES[i % 8] + str(i // 8 + 1)


def move_to_index(uci: str) -> int:
    frm, to = square_index(uci[:2]), square_index(uci[2:4])
    base = frm * 64 + to
    if len(uci) > 4:
        return 4096 + ((frm - 48) * 8 + (to - 56)) * 4 + PROMO_ORDER[uci[4]]
    return base


def index_to_move(i: int) -> str:
    if i < 4096:
        return index_square(i // 64) + index_square(i % 64)
    j = i - 4096
    grp, off = divmod(j, 4)
    return index_square(48 + grp // 8) + index_square(56 + grp % 8) + "qrbn"[off]


def mirror_square(s: str) -> str:
    # inverse of the spatial part of chess.Board.mirror(): rank -> 9-rank
    return s[0] + str(9 - int(s[1]))


def mirror_move(uci: str) -> str:
    promo = uci[4:] if len(uci) > 4 else ""
    return mirror_square(uci[:2]) + mirror_square(uci[2:4]) + promo


class Maia3:
    """Minimal onnxruntime driver for the maia3_simplified.onnx export."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort  # lazy: other subcommands don't need it

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])

    def probs(self, fen: str, elo_self: float, elo_oppo: float):
        """Return (move_probs over legal UCI moves, ldw, top1_prob)."""
        board = chess.Board(fen)
        mirrored = board.turn == chess.BLACK
        if mirrored:
            board = board.mirror()
        tokens = np.zeros((64, 12), dtype=np.float32)
        for sq, pc in board.piece_map().items():
            col = PIECE_COL[pc.piece_type] + (6 if pc.color == chess.BLACK else 0)
            tokens[sq, col] = 1.0
        legal = np.zeros(4352, dtype=np.float32)
        for mv in board.legal_moves:
            legal[move_to_index(mv.uci())] = 1.0
        feeds = {
            "tokens": tokens.reshape(1, 64, 12),
            "elo_self": np.array([elo_self], dtype=np.float32),
            "elo_oppo": np.array([elo_oppo], dtype=np.float32),
        }
        lm, lv = self.session.run(["logits_move", "logits_value"], feeds)
        logits = lm[0].astype(np.float64)
        logits[legal == 0.0] = -np.inf
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top1 = float(probs.max())
        move_probs = {}
        for i in [int(i) for i in np.where(legal > 0)[0]]:
            uci = index_to_move(i)
            if mirrored:
                uci = mirror_move(uci)
            move_probs[uci] = float(probs[i])
        out = np.exp(lv[0].astype(np.float64))
        out = out / out.sum()  # (win, draw, loss) for the side to move
        ldw = (float(out[2]), float(out[1]), float(out[0]))
        return move_probs, ldw, top1


def play_game(model: Maia3, rng: random.Random, elo_w: int, elo_b: int,
              temperature: float, max_ply: int = 240):
    board = chess.Board()
    rows: list[dict] = []
    sans: list[str] = []
    while not board.is_game_over() and board.ply() < max_ply:
        white_turn = board.turn == chess.WHITE
        elo_self = int(elo_w if white_turn else elo_b)
        elo_oppo = int(elo_b if white_turn else elo_w)
        move_probs, ldw, top1 = model.probs(board.fen(), elo_self, elo_oppo)
        weights = [p ** temperature if temperature > 0 else 1.0
                   for p in move_probs.values()]
        uci = rng.choices(list(move_probs), weights=weights)[0]
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves, (board.fen(), uci)
        rows.append({
            "fen": board.fen(), "move_uci": uci, "move_ply": board.ply() + 1,
            "side": "white" if white_turn else "black",
            "elo_self": elo_self, "elo_oppo": elo_oppo,
            "top1_prob": round(top1, 4),
            "ldw": [round(x, 4) for x in ldw],
        })
        sans.append(board.san(mv))
        board.push(mv)
    return rows, sans, board.result(), board


def write_pgn(path: Path, games: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for g in games:
        out.append("\n".join(f'[{k} "{v}"]' for k, v in g["headers"].items()))
        out.append("")
        out.append(" ".join(
            (f"{i // 2 + 1}. " if i % 2 == 0 else "") + s
            for i, s in enumerate(g["san"])) + " " + g["result"])
        out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def cmd_play(args: argparse.Namespace) -> int:
    model = Maia3(Path(args.model))
    rng = random.Random(args.seed)
    pgn_games: list[dict] = []
    labels: list[dict] = []
    for game_no in range(1, args.games + 1):
        elo_w = int(rng.randint(args.elo_min, args.elo_max))
        elo_b = int(rng.randint(args.elo_min, args.elo_max))
        rows, sans, result, _board = play_game(
            model, rng, elo_w, elo_b, args.temperature)
        pgn_games.append({
            "headers": {
                "Event": "Maia3 self-play (random UCI elo)",
                "Site": "selfplay", "Date": date.today().strftime("%Y.%m.%d"),
                "Round": str(game_no),
                "White": f"Maia3 (elo {elo_w})", "Black": f"Maia3 (elo {elo_b})",
                "WhiteElo": str(elo_w), "BlackElo": str(elo_b),
                "Result": result,
            },
            "san": sans, "result": result,
        })
        labels.extend({"game": game_no, "elo_white": elo_w,
                       "elo_black": elo_b, **r} for r in rows)
        if args.report:
            mean_top1 = sum(r["top1_prob"] for r in rows) / max(1, len(rows))
            print(f"game {game_no}: {elo_w} vs {elo_b} -> {result} "
                  f"({len(rows)} plies, mean top1 {mean_top1:.3f})")
    if args.pgn:
        write_pgn(Path(args.pgn), pgn_games)
    if args.labels:
        Path(args.labels).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.labels).open("w", encoding="utf-8") as fh:
            for r in labels:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    if args.report:
        agg: dict[int, list[float]] = defaultdict(list)
        for r in labels:
            agg[r["elo_self"] // 100 * 100].append(r["top1_prob"])
        print("mean top1 probability by elo_self band (conditioning check):")
        for band in sorted(agg):
            v = agg[band]
            print(f"  {band:5d}-{band + 99:5d}: {sum(v) / len(v):.4f}  (n={len(v)})")
    print(json.dumps({"games": len(pgn_games), "label_rows": len(labels)},
                     indent=2))
    return 0


def cmd_fetch_model(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "simple-maia3-inference"
    if not (dst / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none",
                        "--no-checkout", MODEL_REPO, str(dst)], check=True)
        subprocess.run(["git", "-C", str(dst), "fetch", "-q", "origin",
                        MODEL_COMMIT], check=True)
    head = subprocess.run(
        ["git", "-C", str(dst), "rev-parse", MODEL_COMMIT + "^{}"],
        capture_output=True, text=True, check=True).stdout.strip()
    if head != MODEL_COMMIT:
        raise SystemExit(f"pinned commit {MODEL_COMMIT} not found (got {head})")
    subprocess.run(["git", "-C", str(dst), "checkout", "-q", MODEL_COMMIT,
                    "--", MODEL_FILE], check=True)
    print(f"model staged at {dst / MODEL_FILE}")
    return 0


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-model", help="git-stage the pinned ONNX mirror")
    f.add_argument("--out", required=True)
    pl = sub.add_parser("play", help="play self-play games at random elo limits")
    pl.add_argument("--model", required=True)
    pl.add_argument("--games", type=int, default=200)
    pl.add_argument("--elo-min", type=int, default=100)
    pl.add_argument("--elo-max", type=int, default=3200)
    pl.add_argument("--seed", type=int, default=42)
    pl.add_argument("--temperature", type=float, default=1.0)
    pl.add_argument("--pgn", default=None)
    pl.add_argument("--labels", default=None)
    pl.add_argument("--report", action="store_true")
    f.set_defaults(fn=cmd_fetch_model)
    pl.set_defaults(fn=cmd_play)
    return p


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
