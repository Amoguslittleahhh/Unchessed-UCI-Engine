#!/usr/bin/env python3
"""Smoke the default-off Unarchitectured Metal UCI root-hint candidate."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--engine", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--movetime", type=int, default=1000)
    result.add_argument("--timeout", type=float, default=15.0)
    return result


def send(process, command):
    process.stdin.write(command + "\n")
    process.stdin.flush()


def read_until(process, predicate, timeout):
    import time

    deadline = time.monotonic() + timeout
    lines = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("engine exited before expected UCI response")
        lines.append(line.rstrip())
        if predicate(lines[-1]):
            return lines
    raise RuntimeError("timed out waiting for UCI response")


def main():
    args = parser().parse_args()
    if not args.engine.is_file() or not args.model.is_file():
        raise SystemExit("--engine and --model must be existing files")
    process = subprocess.Popen(
        [str(args.engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        send(process, "uci")
        uci_lines = read_until(process, lambda line: line == "uciok", args.timeout)
        expected = "option name UnarchitecturedHint type check default false"
        if expected not in uci_lines:
            raise RuntimeError("candidate option missing or not default-off")
        send(process, "setoption name Adaptive value false")
        send(process, "setoption name OwnBook value false")
        send(process, f"setoption name UnarchitecturedFile value {args.model}")
        send(process, "setoption name UnarchitecturedHint value true")
        send(process, "isready")
        read_until(process, lambda line: line == "readyok", args.timeout)
        send(process, "position startpos")
        send(process, f"go movetime {args.movetime}")
        search_lines = read_until(process, lambda line: line.startswith("bestmove "), args.timeout)
        hint_lines = [line for line in search_lines if "Unarchitectured hint" in line]
        if not hint_lines or not any("exact" in line and "actions=20" in line for line in hint_lines):
            raise RuntimeError("candidate did not provide all 20 exact start-position hints")
        print(hint_lines[-1])
        print(search_lines[-1])
        print("Unarchitectured Metal UCI candidate smoke PASS")
    finally:
        if process.poll() is None:
            send(process, "quit")
            process.wait(timeout=args.timeout)


if __name__ == "__main__":
    main()
