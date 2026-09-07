#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path
SCORE = re.compile(r"evalbar cp (-?\d+)")
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--evalfile", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    p = subprocess.Popen([args.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdin is not None and p.stdout is not None
    def send(s: str) -> None:
        p.stdin.write(s + "\n"); p.stdin.flush()
    def until(prefix: str) -> list[str]:
        lines=[]
        for line in p.stdout:
            lines.append(line.rstrip())
            if line.startswith(prefix): return lines
        raise RuntimeError(prefix)
    send("uci"); until("uciok")
    send("setoption name EvalFile value " + args.evalfile)
    send("isready"); until("readyok")
    with Path(args.out).open("w") as out:
        for i,row in enumerate(rows):
            send("position fen " + row["fen"]); send("evalbar")
            text="\n".join(until("info string [Unchessed] evalbar cp "))
            m=SCORE.search(text)
            if not m: raise RuntimeError(text)
            rec=dict(row); rec["loaded_eval_white_cp"]=int(m.group(1)); out.write(json.dumps(rec,separators=(",",":"))+"\n")
            if i%50==0: print(f"scored={i+1}",flush=True)
    send("quit"); p.wait(timeout=10)
if __name__ == "__main__": main()
