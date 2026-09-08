#!/usr/bin/env python3
"""Run the real Rust unchessed-adapter's loaded NNUE (via EvalFile) over a
label_uci.py-format reference JSONL, producing an evaluate_gates.py-ready
"student" JSONL with student_cp from the STM perspective -- so the eval
gate exercises the actual production inference path, not a Python re-
implementation of it."""
from __future__ import annotations
import argparse, json, queue, re, subprocess, sys, threading, time
from pathlib import Path

CP_RE = re.compile(r"evalbar cp (-?\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--nnue", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proc = subprocess.Popen(
        [args.adapter], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    q: "queue.Queue[str | None]" = queue.Queue()
    threading.Thread(
        target=lambda: [q.put(l) for l in iter(proc.stdout.readline, "")] or q.put(None),
        daemon=True,
    ).start()

    def send(s: str) -> None:
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    def wait_for(token: str, timeout: float = 20.0) -> list[str]:
        """Collect lines until one CONTAINS token (not startswith -- these
        are `info string [Unchessed] ...`-prefixed, not bare tokens)."""
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            try:
                line = q.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if line is None:
                break
            lines.append(line)
            if token in line:
                return lines
        return lines

    send("uci")
    wait_for("uciok")
    send(f"setoption name EvalFile value {args.nnue}")
    send("setoption name Adaptive value false")
    send("isready")
    wait_for("readyok")

    n_written = 0
    with open(args.out, "w") as out:
        for raw in Path(args.reference).read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            send(f"position fen {row['fen']}")
            send("evalbar")
            lines = wait_for("evalbar cp", timeout=10.0)
            cp_white = None
            for line in lines:
                m = CP_RE.search(line)
                if m:
                    cp_white = int(m.group(1))
                    break
            if cp_white is None:
                continue
            # evalbar reports White-relative; the reference/gate convention
            # (matching label_uci.py's UCI "score cp") is side-to-move-relative.
            white_to_move = " w " in row["fen"]
            student_cp = cp_white if white_to_move else -cp_white
            out.write(json.dumps({"position_id": row["position_id"], "student_cp": student_cp}) + "\n")
            n_written += 1

    send("quit")
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print(f"student rows written={n_written}", file=sys.stderr)


if __name__ == "__main__":
    main()
