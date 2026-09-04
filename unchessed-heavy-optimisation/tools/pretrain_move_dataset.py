#!/usr/bin/env python3
"""Pretrain data bridge: labeled games -> next-move-prediction shards.

Converts the self-play / cloud / real-human labeled game sets of this
repo into flat, leakage-safe shards for the move-prediction pretrain
objective (predict the next legal action from the board + dual elo
conditioning). The board encoding is the STM-normalized 12-plane
bitboard convention of unchessed-datagen (planes 0-5 = side to move
P,N,B,R,Q,K; 6-11 = opponent; a1=0 square numbering; move = from |
to<<6; castling bits 0/1 mover-K/Q, 2/3 opponent-K/Q), and actions
live in the 4096x5 = 20480 move-promotion space (move*5 + promotion,
promotion 0 none / 1 knight / 2 bishop / 3 rook / 4 queen) — the same
action vocabulary as the Unarchitectured v1 student
(`policy_action_vocabulary: 20480`).

Inputs (all already in the repo or produced by the cloud generator):
  * label JSONL (+ nothing else): tools/maia3_cloud_selfplay output or
    data/selfplay/maia3-100-3200-labels.jsonl — dual elo per move from
    the labels themselves (engine + elo_quality tags included),
  * plain PGN with WhiteElo/BlackElo headers: data/training-elo — real
    rated humans; elo_self/elo_oppo come from the header ratings and
    the rows are tagged engine=human.

Leakage guard: the train/val split is by GAME (game ids are
contiguous and disjoint), never by row. A game's rows can only ever
appear on one side of the split; the split is recorded in the
manifest so the predictor can re-derive it.

Outputs under --out:
  shard-train.npz / shard-val.npz   numpy arrays (see below)
  manifest.json                     sources + shas, counts, split,
                                    engine/quality histograms

npz arrays (row i = one labeled position):
  bitboards    (N,12) u64   STM-normalized planes, a1=0
  action       (N,)   u32   target = move*5 + promotion
  legal        (N,218) u32  legal action indices, 0-padded
  legal_count  (N,)   u8    1..218 (the maximum in the v1 spec)
  elo_self     (N,)   u16   conditioning: the mover's elo
  elo_oppo     (N,)   u16   conditioning: the opponent's elo
  engine       (N,)   u8    0 maia3, 1 stockfish, 2 lc0, 3 rubichess,
                            4 human
  quality      (N,)   u8    0 calibrated, 1 native, 2 approximate,
                            3 human (real rated games)
  game_id      (N,)   i32
  ply          (N,)   u16

Run (sandbox, committed reference set):
  python3 tools/pretrain_move_dataset.py \
      --labels data/selfplay/maia3-100-3200-labels.jsonl \
      --out /tmp/pretrain-selfplay --val-games 50

Dependencies: python-chess (tools/requirements-dev.txt), numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np

MAX_LEGAL = 218  # maximum_legal_tokens in config/unarchitectured_v1.json
ENGINE_IDS = {"maia3": 0, "stockfish": 1, "lc0": 2, "rubichess": 3,
              "human": 4}
QUALITY_IDS = {"calibrated": 0, "native": 1, "approximate": 2,
               "human": 3}
PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
               chess.QUEEN, chess.KING)
PROMO_IDS = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
             chess.QUEEN: 4}


def move_of(mv: chess.Move) -> int:
    """Action-space encoding of a move that is ALREADY in the
    side-to-move-normalized view (squares a1=0, no rank flip)."""
    move = mv.from_square | (mv.to_square << 6)
    return move * 5 + PROMO_IDS[mv.promotion]


def board_arrays(board: chess.Board,
                 mv: chess.Move) -> tuple[np.ndarray, int, list[int], int]:
    """STM-normalized (bitboards(12,), castling byte, legal actions,
    target action).

    board may be from either side's perspective; mirroring is applied
    here so the mover always occupies the 'white' planes, matching
    the unchessed-datagen convention. The target move is mirrored with
    the board so target and legal set live in the same view (a
    double mirror would put the target outside the legal set for
    half the dataset).
    """
    if board.turn == chess.BLACK:
        board = board.mirror()
        mv = chess.Move(chess.square_mirror(mv.from_square),
                        chess.square_mirror(mv.to_square),
                        mv.promotion)
    planes = [int(board.pieces(pt, chess.WHITE)) for pt in PIECE_ORDER]
    planes += [int(board.pieces(pt, chess.BLACK)) for pt in PIECE_ORDER]
    # rights keyed by rook home square (python-chess): white=mover
    # after the mirror above
    castling = 0
    if board.has_castling_rights(chess.H1):
        castling |= 1
    if board.has_castling_rights(chess.A1):
        castling |= 2
    if board.has_castling_rights(chess.H8):
        castling |= 4
    if board.has_castling_rights(chess.A8):
        castling |= 8
    legal = [move_of(m) for m in board.legal_moves]
    return (np.asarray(planes, dtype=np.uint64), castling, legal,
            move_of(mv))


def rows_from_labels(path: Path) -> list[dict]:
    """Label JSONL (cloud generator / data/selfplay): self-contained
    (fen, move_uci, dual elo, engine, quality per row)."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append({
                "fen": r["fen"],
                "move_uci": r["move_uci"],
                "elo_self": int(r["elo_self"]),
                "elo_oppo": int(r["elo_oppo"]),
                "engine": r.get("engine", "maia3"),
                "quality": r.get("elo_quality", "calibrated"),
                "game_id": int(r["game"]),
                "ply": int(r["move_ply"]),
            })
    return out


def rows_from_pgn(path: Path, id_base: int = 0) -> list[dict]:
    """Plain PGN with WhiteElo/BlackElo headers (data/training-elo):
    real rated humans; the header ratings are the dual-elo labels.

    Game ids: the numeric Round header when present, otherwise
    `id_base - seq` (a separate negative id space per source file so
    mixing sources can never collide game ids across the split).
    """
    out = []
    stream = path.open("r", encoding="utf-8")
    try:
        while True:
            g = chess.pgn.read_game(stream)
            if g is None:
                break
            w = int(g.headers.get("WhiteElo", 0))
            b = int(g.headers.get("BlackElo", 0))
            if not w or not b:
                continue
            # game id: the Round header when numeric (cloud output),
            # otherwise a file-order counter (header may be "?")
            rnd = g.headers.get("Round", "")
            if rnd.isdigit():
                game_id = int(rnd) + id_base
            else:
                game_id = id_base - (len(out) + 1)
            board = chess.Board()
            for i, mv in enumerate(g.mainline_moves()):
                if not board.is_game_over() and mv in board.legal_moves:
                    white_turn = board.turn == chess.WHITE
                    out.append({
                        "fen": board.fen(),
                        "move_uci": mv.uci(),
                        "elo_self": w if white_turn else b,
                        "elo_oppo": b if white_turn else w,
                        "engine": "human",
                        "quality": "human",
                        "game_id": game_id,
                        "ply": board.ply() + 1,
                    })
                    board.push(mv)
    finally:
        stream.close()
    return out


def build(rows: list[dict]) -> dict:
    n = len(rows)
    bitboards = np.empty((n, 12), dtype=np.uint64)
    castling = np.empty(n, dtype=np.uint8)
    actions = np.empty(n, dtype=np.uint32)
    legal = np.zeros((n, MAX_LEGAL), dtype=np.uint32)
    legal_count = np.empty(n, dtype=np.uint8)
    elo_self = np.empty(n, dtype=np.uint16)
    elo_oppo = np.empty(n, dtype=np.uint16)
    engines = np.empty(n, dtype=np.uint8)
    qualities = np.empty(n, dtype=np.uint8)
    game_ids = np.empty(n, dtype=np.int32)
    plies = np.empty(n, dtype=np.uint16)
    valid = np.zeros(n, dtype=bool)
    bad = 0
    for i, r in enumerate(rows):
        try:
            board = chess.Board(r["fen"])
            mv = chess.Move.from_uci(r["move_uci"])
            if mv not in board.legal_moves:
                bad += 1
                continue
            bb, c, leg, target = board_arrays(board, mv)
            bitboards[i] = bb
            castling[i] = c
            actions[i] = target
            legal[i, :len(leg)] = leg
            legal_count[i] = len(leg)
            if target not in leg:
                # should be impossible: the played move is legal, and
                # target/legal are built in the same normalized view.
                # If this ever fires, the encoding changed — stop.
                raise SystemExit(
                    f"row {i}: target {target} not in legal set "
                    f"(len {len(leg)}) — encoding/view mismatch")
            elo_self[i] = r["elo_self"]
            elo_oppo[i] = r["elo_oppo"]
            engines[i] = ENGINE_IDS[r["engine"]]
            qualities[i] = QUALITY_IDS[r["quality"]]
            game_ids[i] = r["game_id"]
            plies[i] = r["ply"]
            valid[i] = True
        except (ValueError, KeyError) as exc:
            bad += 1
            if bad <= 3:
                print(f"skipping row {i}: {exc}", file=sys.stderr)
    return {
        "bitboards": bitboards, "castling": castling, "action": actions,
        "legal": legal, "legal_count": legal_count,
        "elo_self": elo_self, "elo_oppo": elo_oppo, "engine": engines,
        "quality": qualities, "game_id": game_ids, "ply": plies,
        "rows": n, "bad": bad, "valid": valid,
    }


def split_games(arrays: dict, val_games: int, games_desc: list[int]):
    """Game-disjoint split: the last `val_games` distinct game ids
    (in descending order of first appearance) form validation."""
    order = []
    seen = set()
    for g in arrays["game_id"]:
        if g not in seen:
            seen.add(g)
            order.append(int(g))
    if len(order) <= val_games:
        raise SystemExit(f"only {len(order)} games — val-games must be "
                         f"smaller")
    val = set(order[-val_games:])
    train_mask = np.asarray([g not in val for g in arrays["game_id"]],
                            dtype=bool)
    return train_mask, sorted(val)


def save_shard(path: Path, arrays: dict, mask: np.ndarray) -> None:
    sel = {k: v[mask] for k, v in arrays.items() if k not in
           ("rows", "bad")}
    # Write under a temp name and rename into place only once the file is
    # complete -- np.savez_compressed writes the real path directly
    # otherwise, so a crash or kill partway through (or, previously, the
    # histogram KeyError below) left a truncated .npz sitting at the real
    # filename, indistinguishable from a genuinely finished shard. Passing
    # an open file object (rather than a path string) keeps the exact name
    # we choose -- savez_compressed auto-appends ".npz" to a bare string
    # path that doesn't already end that way, which a ".npz.tmp" name
    # would trip over.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **sel)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", action="append", default=[],
                   help="label JSONL (cloud generator / data/selfplay); "
                        "repeatable, mixable with --pgn")
    p.add_argument("--pgn", action="append", default=[],
                   help="plain PGN with WhiteElo/BlackElo headers "
                        "(data/training-elo); repeatable")
    p.add_argument("--out", required=True)
    p.add_argument("--val-games", type=int, default=50,
                   help="number of games held out for validation "
                        "(game-disjoint)")
    args = p.parse_args()

    rows = []
    sources = []
    for spec in args.labels:
        path = Path(spec)
        rs = rows_from_labels(path)
        rows.extend(rs)
        sources.append({"kind": "labels", "file": str(path),
                        "rows": len(rs),
                        "sha256": hashlib.sha256(
                            path.read_bytes()).hexdigest()})
    for fi, spec in enumerate(args.pgn):
        path = Path(spec)
        rs = rows_from_pgn(path, id_base=-(fi + 1) * 1_000_000)
        rows.extend(rs)
        sources.append({"kind": "pgn", "file": str(path),
                        "rows": len(rs),
                        "sha256": hashlib.sha256(
                            path.read_bytes()).hexdigest()})
    if not rows:
        raise SystemExit("no rows")

    print(f"building arrays for {len(rows)} rows...", flush=True)
    arrays = build(rows)
    if arrays["bad"]:
        n_bad = arrays["bad"]
        valid = arrays["valid"]
        keep = int(valid.sum())
        # Compact by the actual valid-row mask, not a "trim the tail"
        # truncation -- a bad row anywhere but the very end otherwise left
        # its uninitialized np.empty garbage inside the retained slice
        # while discarding a genuinely good row from the tail instead. That
        # garbage (an engine/quality id with no real meaning) doesn't fail
        # loudly here -- it silently survives into the saved shards and
        # only surfaces later as a KeyError in the histogram step below,
        # by which point the corrupt shards are already on disk.
        arrays = {k: (v[valid] if isinstance(v, np.ndarray) else v)
                  for k, v in arrays.items() if k not in ("rows", "bad", "valid")}
        arrays["rows"] = keep
        print(f"dropped {n_bad} bad rows (kept {keep})")

    train_mask, val_game_ids = split_games(arrays, args.val_games,
                                           arrays["game_id"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_shard(out / "shard-train.npz", arrays, train_mask)
    save_shard(out / "shard-val.npz", arrays, ~train_mask)

    hist = {}
    for name, ids in (("engine", arrays["engine"]),
                      ("quality", arrays["quality"])):
        inv = {v: k for k, v in (ENGINE_IDS if name == "engine"
                                 else QUALITY_IDS).items()}
        for k in np.unique(ids):
            hist[f"{name}_{inv[int(k)]}"] = int((ids == k).sum())
    manifest = {
        "tool": "tools/pretrain_move_dataset.py",
        "sources": sources,
        "rows": int(arrays["rows"]),
        "games": int(len(np.unique(arrays["game_id"]))),
        "train_rows": int(train_mask.sum()),
        "val_rows": int((~train_mask).sum()),
        "val_games": val_game_ids,
        "split": "game-disjoint (last val_games game ids -> validation)",
        "board_encoding": ("STM-normalized 12 planes (mover P,N,B,R,Q,K "
                           "then opponent), a1=0 — unchessed-datagen "
                           "convention"),
        "action_space": "4096 moves x 5 promotion classes = 20480 "
                        "(move*5+promo; legal-only softmax at train time)",
        "histograms": hist,
    }
    manifest_path = out / "manifest.json"
    manifest_tmp = out / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
