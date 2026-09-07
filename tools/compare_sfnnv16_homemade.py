#!/usr/bin/env python3
"""Black-box comparison of official Stockfish 19 and Unchessed homemade evalbar.

This script does not import, copy, or inspect Stockfish implementation code or
weights. It only sends UCI commands and compares returned scores.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from pathlib import Path


def epd_to_fen(line: str) -> str:
    fields = line.split(" bm ", 1)[0].split()
    if len(fields) != 4:
        raise ValueError(f"unsupported EPD: {line}")
    return " ".join(fields + ["0", "1"])


def run_stockfish(binary: str, fen: str, depth: int) -> int:
    process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(f"uci\nisready\nposition fen {fen}\ngo depth {depth}\n".encode())
    process.stdin.flush()
    lines = []
    for line in process.stdout:
        lines.append(line)
        if line.startswith(b"bestmove "):
            break
    process.stdin.write(b"quit\n")
    process.stdin.flush()
    process.wait(timeout=10)
    out = b"".join(lines)
    scores = re.findall(rb"score cp (-?\d+)", out)
    mates = re.findall(rb"score mate (-?\d+)", out)
    if mates:
        mate = int(mates[-1])
        return (30000 if mate > 0 else -30000) + max(-100, min(100, mate))
    if not scores:
        raise RuntimeError(out.decode(errors="replace"))
    return int(scores[-1])


def run_homemade(binary: str, fen: str, mode: str, evalfile: str | None = None) -> int:
    option = f"setoption name EvalFile value {evalfile}\n" if evalfile else ""
    command = f"evalbar {mode}" if mode else "evalbar"
    commands = f"uci\n{option}isready\nposition fen {fen}\n{command}\nquit\n"
    out = subprocess.check_output([binary], input=commands.encode(), stderr=subprocess.STDOUT)
    matches = re.findall(rb"evalbar cp (-?\d+)", out)
    if not matches:
        raise RuntimeError(out.decode(errors="replace"))
    return int(matches[-1])


def white_score(stm_score: int, fen: str) -> int:
    return stm_score if fen.split()[1] == "w" else -stm_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--unchessed", required=True)
    parser.add_argument("--epd", required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="homemade")
    parser.add_argument("--evalfile", default=None)
    args = parser.parse_args()

    rows = []
    for raw in Path(args.epd).read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if " bm " in raw:
            fen = epd_to_fen(raw)
        else:
            fields = raw.split()
            if len(fields) != 6:
                raise ValueError(f"unsupported FEN/EPD line: {raw}")
            fen = " ".join(fields)
        sf_white = white_score(run_stockfish(args.stockfish, fen, args.depth), fen)
        homemade = run_homemade(args.unchessed, fen, args.mode, args.evalfile)
        delta = homemade - sf_white
        rows.append((fen, sf_white, homemade, delta))

    abs_errors = [abs(row[3]) for row in rows]
    with Path(args.out).open("w") as handle:
        handle.write("# Direct SFNNv16 black-box comparison\n\n")
        handle.write("The Stockfish value is obtained only through UCI from the official Stockfish 19 binary and published network; no Stockfish source or weights are imported into Unchessed. The homemade value is obtained through the selected Unchessed UCI mode.\n\n")
        handle.write(f"positions={len(rows)}\nstockfish_depth={args.depth}\n")
        handle.write(f"mae_cp={statistics.mean(abs_errors):.3f}\n")
        handle.write(f"median_abs_error_cp={statistics.median(abs_errors):.3f}\n")
        handle.write(f"max_abs_error_cp={max(abs_errors)}\n")
        handle.write("\n| FEN | SFNNv16/Stockfish white cp | Homemade white cp | Delta cp |\n|---|---:|---:|---:|\n")
        for fen, sf_white, homemade, delta in rows:
            handle.write(f"| `{fen}` | {sf_white} | {homemade} | {delta:+d} |\n")

    print(f"positions={len(rows)}")
    print(f"mae_cp={statistics.mean(abs_errors):.3f}")
    print(f"median_abs_error_cp={statistics.median(abs_errors):.3f}")
    print(f"max_abs_error_cp={max(abs_errors)}")


if __name__ == "__main__":
    main()
