#!/usr/bin/env python3
"""Calibrate achieved search depth against remaining clock time.

This is the "integrated depth/NPS across a real, larger position set" item
that round 6/7's status updates repeatedly flagged as still missing --
everything so far only measured the hint's *inference* cost in isolation,
never its effect on the depth the real UCI `go` loop actually reaches at
different points in a real game clock.

Drives the real compiled engine over stdin/stdout (no test harness, no
mocking) at a spread of `wtime`/`btime` values spanning the low-time-skip
threshold up through a genuine clock surplus, for both `UnarchitecturedHint`
off (baseline) and on, across several real positions (not just the start
position). For each `go`, records the deepest `info depth` line seen and the
wall time the engine actually took to return `bestmove`.

Usage:
    python tools/unarchitectured_v1_depth_time_calibration.py \
        --engine target/release/unchessed-adapter \
        --model artifacts/unarchitectured-v1-final.unarchv1
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import time

POSITIONS = [
    ("startpos", "startpos"),
    (
        "italian_middlegame",
        "fen r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6",
    ),
    (
        "endgame_rook",
        "fen 8/5pk1/6p1/7p/7P/6P1/5PK1/3r4 b - - 0 1",
    ),
]

# Milliseconds of remaining clock, spanning below and above the low-time
# skip gate (default 30000) and the aggressive one used in round 7's pilot
# (1000).
TIME_LEFT_MS = [500, 1000, 3000, 5000, 10000, 30000, 60000, 120000]

DEPTH_RE = re.compile(r"\binfo depth (\d+)\b")
BESTMOVE_RE = re.compile(r"^bestmove (\S+)")


def _pump_lines(stream, out_queue):
    """Runs in a background thread: forwards each line as it arrives, and a
    single None once the stream closes (process exited)."""
    for line in iter(stream.readline, ""):
        out_queue.put(line)
    out_queue.put(None)


def run_go(engine_path, model_path, hint_enabled, min_time_ms, position, time_left_ms, timeout_s):
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    # A direct, blocking proc.stdout.readline() defeats any deadline check
    # wrapped around it: the surrounding `while time.monotonic() < deadline`
    # only gets evaluated BETWEEN reads, so a silent or stalled engine that
    # never writes another line hangs this script indefinitely regardless
    # of the advertised timeout. Reading through a background thread and a
    # queue lets the deadline actually be enforced with queue.get(timeout=).
    line_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_pump_lines, args=(proc.stdout, line_queue), daemon=True)
    reader.start()

    def read_until(deadline):
        """One line, or None on timeout/EOF -- never blocks past `deadline`."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return line_queue.get(timeout=remaining)
        except queue.Empty:
            return None

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    send("uci")
    send("setoption name Threads value 1")
    send("setoption name Adaptive value false")
    send("setoption name OwnBook value false")
    send("setoption name Hash value 256")
    if hint_enabled:
        send(f"setoption name UnarchitecturedFile value {model_path}")
        send(f"setoption name UnarchitecturedMinTime value {min_time_ms}")
        send("setoption name UnarchitecturedHint value true")
    send("isready")

    deadline = time.monotonic() + 5.0
    while True:
        line = read_until(deadline)
        if line is None:  # timed out, or the process exited without readyok
            break
        if line.strip() == "readyok":
            break

    send("ucinewgame")
    send(f"position {position}")
    started = time.monotonic()
    send(
        f"go wtime {time_left_ms} btime {time_left_ms} winc 50 binc 50"
    )

    max_depth = 0
    charged_ms = None
    deadline = started + timeout_s
    bestmove = None
    while True:
        line = read_until(deadline)
        if line is None:  # advertised timeout reached, or engine went silent
            break
        depth_match = DEPTH_RE.search(line)
        if depth_match:
            max_depth = max(max_depth, int(depth_match.group(1)))
        if "Unarchitectured hint" in line and "charged=" in line:
            try:
                charged_ms = int(line.split("charged=")[1].split("ms")[0])
            except (IndexError, ValueError):
                pass
        if BESTMOVE_RE.match(line):
            bestmove = line.split()[1]
            break
    elapsed_ms = (time.monotonic() - started) * 1000.0

    send("quit")
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()

    return {
        "max_depth": max_depth,
        "bestmove": bestmove,
        "elapsed_ms": round(elapsed_ms, 1),
        "hint_charged_ms": charged_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-time", type=int, default=1000, help="UnarchitecturedMinTime to test")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    report = {"min_time_tested": args.min_time, "positions": {}}
    for pos_name, position in POSITIONS:
        report["positions"][pos_name] = {}
        for time_left in TIME_LEFT_MS:
            baseline = run_go(
                args.engine, args.model, False, args.min_time, position, time_left, args.timeout
            )
            hinted = run_go(
                args.engine, args.model, True, args.min_time, position, time_left, args.timeout
            )
            report["positions"][pos_name][str(time_left)] = {
                "baseline": baseline,
                "hint": hinted,
                "depth_delta": hinted["max_depth"] - baseline["max_depth"],
            }
            print(
                f"{pos_name:20s} time_left={time_left:7d}ms  "
                f"baseline depth={baseline['max_depth']:2d} elapsed={baseline['elapsed_ms']:7.1f}ms  "
                f"hint depth={hinted['max_depth']:2d} elapsed={hinted['elapsed_ms']:7.1f}ms "
                f"charged={hinted['hint_charged_ms']}  "
                f"delta={hinted['max_depth'] - baseline['max_depth']:+d}"
            )

    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
