#!/usr/bin/env python3
"""Label a JSONL manifest through a persistent UCI engine process.

The teacher name is metadata only: Stockfish, lc0, and maia3 outputs are kept
separate and must not be averaged implicitly.
"""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path

CP = re.compile(r"score cp (-?\d+)")
MATE = re.compile(r"score mate (-?\d+)")
WDL = re.compile(r"wdl (-?\d+) (-?\d+) (-?\d+)")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True, choices=["stockfish", "lc0", "maia3"])
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--option", action="append", default=[], help="UCI option as NAME=VALUE")
    args = ap.parse_args()
    p = subprocess.Popen([args.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdin and p.stdout
    def send(s: str) -> None: p.stdin.write(s + "\n"); p.stdin.flush()
    send("uci")
    for line in p.stdout:
        if line.startswith("uciok"): break
    for opt in args.option:
        name, value = opt.split("=", 1); send(f"setoption name {name} value {value}")
    send("isready")
    for line in p.stdout:
        if line.startswith("readyok"): break
    with Path(args.out).open("w") as out:
        for raw in Path(args.manifest).read_text().splitlines():
            if not raw.strip(): continue
            row = json.loads(raw); send(f"position fen {row['fen']}")
            command = f"go nodes {args.nodes}" if args.nodes else f"go depth {args.depth}"
            send(command); last = ""; cp = None; mate = None; wdl = None; bestmove = None
            for line in p.stdout:
                if line.startswith("info "):
                    last = line
                    m = CP.search(line); cp = int(m.group(1)) if m else cp
                    m = MATE.search(line); mate = int(m.group(1)) if m else mate
                    m = WDL.search(line); wdl = list(map(int, m.groups())) if m else wdl
                if line.startswith("bestmove "):
                    bestmove = line.split()[1]; break
            result = dict(row, teacher=args.teacher, depth=args.depth, nodes=args.nodes, bestmove=bestmove, score_cp=cp, score_mate=mate, wdl=wdl)
            out.write(json.dumps(result, sort_keys=True) + "\n"); out.flush()
    send("quit"); p.wait(timeout=10)

if __name__ == "__main__":
    main()
