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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]
    process = subprocess.Popen([args.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdin is not None and process.stdout is not None
    def send(command: str) -> None:
        process.stdin.write(command + "\n")
        process.stdin.flush()
    def until(prefix: str) -> list[str]:
        lines = []
        for line in process.stdout:
            lines.append(line.rstrip())
            if line.startswith(prefix):
                return lines
        raise RuntimeError(f"process exited before {prefix}")
    send("uci")
    until("uciok")
    send("isready")
    until("readyok")
    with Path(args.out).open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            send("position fen " + row["fen"])
            send("evalbar homemade")
            lines = until("info string [Unchessed] evalbar cp ")
            text = "\n".join(lines)
            match = SCORE.search(text)
            if not match:
                raise RuntimeError(text)
            record = dict(row)
            record["unchessed_white_cp"] = int(match.group(1))
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 50 == 0:
                print(f"scored={index + 1}", flush=True)
    send("quit")
    process.wait(timeout=10)


if __name__ == "__main__":
    main()
