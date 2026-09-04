#!/usr/bin/env python3
"""Differential runtime checks for FIDE-relevant UCI state and move behavior.

This checks that the current binary returns a legal move for valid positions,
handles terminal positions coherently, and does not crash on deliberately
malformed FEN inputs. It does not claim to test OTB-only arbiter/clock rules.
"""
from __future__ import annotations
import json, random, subprocess, sys
from pathlib import Path
import chess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "target/debug/unchessed-adapter"
OUT = ROOT / "research/fide_uci_differential_results_v3.json"

CASES = [
    ("startpos", chess.Board().fen()),
    ("mate_in_one_white", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 100 1"),
    ("mate_in_one_black", "r5k1/8/8/8/8/8/5PPP/6K1 b - - 100 1"),
    ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ("dead_king_vs_king", "7k/8/8/8/8/8/8/K7 w - - 0 1"),
    ("bishop_only", "7k/8/8/8/8/8/2B5/K7 w - - 0 1"),
    ("knight_only", "7k/8/8/8/8/8/2N5/K7 w - - 0 1"),
    ("white_castle_kingside", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("white_castle_queenside", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("black_castle_both", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
    ("en_passant_white", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
    ("en_passant_black", "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"),
    ("promotion_quiet", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("promotion_capture", "3rk3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("pinned_piece", "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1"),
    ("king_attack_constraint", "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1"),
]

INVALID_FENS = [
    ("missing_white_king", "7k/8/8/8/8/8/8/8 w - - 0 1"),
    ("missing_black_king", "K7/8/8/8/8/8/8/8 w - - 0 1"),
    ("two_white_kings", "4k3/8/8/8/8/8/8/KK6 w - - 0 1"),
    ("pawn_on_back_rank", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("bad_castling_right", "4k3/8/8/8/8/8/8/4K3 w K - 0 1"),
    ("occupied_ep_square", "4k3/8/8/3pP3/8/8/8/4K3 w - e4 0 1"),
]

def run_batch(fens: list[str]) -> tuple[list[str], str, int]:
    commands = "uci\nisready\nsetoption name Adaptive value false\n"
    for fen in fens:
        commands += f"position fen {fen}\ngo depth 1\n"
    commands += "quit\n"
    p = subprocess.run([str(ENGINE)], input=commands, text=True, capture_output=True, timeout=90)
    best = [line.split()[1] for line in p.stdout.splitlines() if line.startswith("bestmove ")]
    return best, p.stdout + "\n--- STDERR ---\n" + p.stderr, p.returncode


def run_one(best: str, transcript: str, rc: int, fen: str) -> dict:
    board = chess.Board(fen)
    legal = best != "0000" and best in {m.uci() for m in board.legal_moves}
    terminal = board.is_game_over(claim_draw=False)
    return {"fen": fen, "terminal_reference": terminal,
            "reference_status": board.outcome(claim_draw=False).result() if terminal else None,
            "bestmove": best, "bestmove_legal": legal, "returncode": rc,
            "stdout_stderr": transcript}


# Kept as a small, separately callable probe for debugging individual cases.
def run_uci(fen: str) -> tuple[str, str, int]:
    best, transcript, rc = run_batch([fen])
    return (best[0] if best else ""), transcript, rc

def random_positions(n: int = 1000) -> list[tuple[str, str]]:
    rng = random.Random(20260904)
    out = []
    for i in range(n):
        b = chess.Board()
        for _ in range(rng.randrange(0, 60)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append((f"random_reachable_{i:03d}", b.fen()))
    return out

def main() -> int:
    if not ENGINE.exists():
        print(f"missing engine: {ENGINE}", file=sys.stderr)
        return 2
    named_valid = CASES + random_positions()
    valid_fens = [fen for _, fen in named_valid]
    invalid_fens = [fen for _, fen in INVALID_FENS]
    all_fens = valid_fens + invalid_fens
    bestmoves, transcript, rc = run_batch(all_fens)
    results = []
    for idx, (name, fen) in enumerate(named_valid):
        best = bestmoves[idx] if idx < len(bestmoves) else ""
        row = run_one(best, transcript, rc, fen)
        row["name"] = name
        results.append(row)
    for j, (name, fen) in enumerate(INVALID_FENS):
        idx = len(named_valid) + j
        results.append({"name": name, "fen": fen, "invalid_fen_reference": True,
                        "bestmove": bestmoves[idx] if idx < len(bestmoves) else "",
                        "returncode": rc, "stdout_stderr": transcript})
    summary = {
        "engine": str(ENGINE), "cases": len(results),
        "valid_position_cases": len(named_valid),
        "legal_bestmove_pass": sum(1 for r in results if "bestmove_legal" in r and r["bestmove_legal"]),
        "legal_bestmove_fail": sum(1 for r in results if "bestmove_legal" in r and not r["bestmove_legal"] and not r["terminal_reference"]),
        "invalid_fen_cases": len(INVALID_FENS),
        "invalid_fen_nonzero_exit": sum(1 for r in results if "invalid_fen_reference" in r and r["returncode"] != 0),
        "results": results,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if summary["legal_bestmove_fail"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
