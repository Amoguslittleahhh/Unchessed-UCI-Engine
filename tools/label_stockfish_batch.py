#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import queue
import re
import subprocess
import threading
from pathlib import Path

SCORE = re.compile(r"score cp (-?\d+)")
MATE = re.compile(r"score mate (-?\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    rows = []
    for line in Path(args.manifest).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= args.limit:
            break
    process = subprocess.Popen([args.stockfish], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdin is not None and process.stdout is not None
    def send(command: str) -> None:
        process.stdin.write(command + "\n")
        process.stdin.flush()
    def wait_bestmove() -> list[str]:
        lines = []
        for line in process.stdout:
            lines.append(line.rstrip())
            if line.startswith("bestmove "):
                return lines
        raise RuntimeError("Stockfish exited before bestmove")
    send("uci")
    while True:
        line = process.stdout.readline()
        if line.startswith("uciok"):
            break
    send("setoption name Threads value 1")
    send("setoption name Hash value 16")
    send("isready")
    while process.stdout.readline().strip() != "readyok":
        pass
    with Path(args.out).open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            send("position fen " + row["fen"])
            send(f"go depth {args.depth}")
            lines = wait_bestmove()
            matches = MATE.findall("\n".join(lines))
            if matches:
                mate = int(matches[-1])
                score = (30000 if mate > 0 else -30000) + max(-100, min(100, mate))
            else:
                scores = SCORE.findall("\n".join(lines))
                if not scores:
                    raise RuntimeError("no score in Stockfish output")
                score = int(scores[-1])
            record = dict(row)
            record["stockfish_stm_cp"] = score
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 50 == 0:
                print(f"labeled={index + 1}", flush=True)
    send("quit")
    process.wait(timeout=10)


if __name__ == "__main__":
    main()
