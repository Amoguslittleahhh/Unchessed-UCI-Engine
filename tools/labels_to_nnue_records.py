#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
import chess

REC = struct.Struct('<12QhB5x')
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def transformed_square(square: int, flip: bool) -> int:
    return square ^ 56 if flip else square

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('labels')
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-abs-cp', type=int, default=2000)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    with Path(args.out).open('wb') as dst:
        for row in rows:
            board = chess.Board(row['fen'])
            us = board.turn
            them = not us
            flip = us == chess.BLACK
            bitboards = []
            for color in (us, them):
                for piece in PIECES:
                    bits = 0
                    for square in board.pieces(piece, color):
                        bits |= 1 << transformed_square(square, flip)
                    bitboards.append(bits)
            raw_score = int(row['stockfish_stm_cp'])
            if abs(raw_score) > args.max_abs_cp:
                continue
            score = max(-32768, min(32767, raw_score))
            wdl = 2 if score >= 80 else 0 if score <= -80 else 1
            dst.write(REC.pack(*bitboards, score, wdl))
    print(json.dumps({'input_records': len(rows), 'record_bytes': REC.size, 'out': str(args.out), 'max_abs_cp': args.max_abs_cp}))

if __name__ == '__main__':
    main()
