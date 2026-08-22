#!/usr/bin/env python3
"""Adversarial UCI correctness regressions.

Usage: python tools/uci_edge_smoke.py <engine>
"""

import queue
import subprocess
import sys
import threading
import time


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.q = queue.Queue()
        self.log = []
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        self.until(lambda line: line == "uciok", 15)

    def _reader(self):
        for line in self.p.stdout:
            line = line.rstrip("\n")
            self.log.append(line)
            self.q.put(line)

    def send(self, command):
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()

    def until(self, predicate, timeout):
        output = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.q.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            output.append(line)
            if predicate(line):
                return output
        raise AssertionError(f"timeout; tail={self.log[-20:]}")

    def drain_for(self, seconds):
        output = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                output.append(self.q.get(timeout=max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                break
        return output

    def close(self):
        if self.p.poll() is None:
            self.send("quit")
            self.p.wait(timeout=5)


def bestmove(output):
    return next(line.split()[1] for line in reversed(output) if line.startswith("bestmove "))


def main():
    engine = Engine(sys.argv[1])

    malformed = [
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "4k3/8/8/8/8/8/4K3/8 w K - 0 1",
        "4k3/8/8/8/8/8/1P6/4K3 w - a1junk 0 1",
    ]
    for fen in malformed:
        engine.send("position fen " + fen)
        output = engine.until(lambda line: "could not parse:" in line, 3)
        assert any("could not parse:" in line for line in output)
    engine.send("isready")
    engine.until(lambda line: line == "readyok", 3)
    print("PASS malformed FEN rejection and liveness")

    engine.send("position startpos")
    engine.send("go nodes 1")
    output = engine.until(lambda line: line.startswith("bestmove "), 5)
    reported = []
    for line in output:
        fields = line.split()
        if "nodes" in fields:
            reported.append(int(fields[fields.index("nodes") + 1]))
    assert not reported or max(reported) <= 1, reported
    print("PASS exact tiny node limit")

    engine.send("setoption name Threads value 2")
    engine.send("position startpos")
    engine.send("go nodes 1000")
    output = engine.until(lambda line: line.startswith("bestmove "), 5)
    reported = []
    for line in output:
        fields = line.split()
        if "nodes" in fields:
            reported.append(int(fields[fields.index("nodes") + 1]))
    assert not reported or max(reported) <= 1000, reported
    engine.send("setoption name Threads value 1")
    print("PASS shared multi-thread node limit")

    engine.send("position startpos")
    engine.send("go depth 3 searchmoves a2a3")
    output = engine.until(lambda line: line.startswith("bestmove "), 5)
    assert bestmove(output) == "a2a3", output[-10:]
    print("PASS searchmoves restriction")

    engine.send("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    engine.send("go infinite")
    early = engine.drain_for(1.0)
    assert not any(line.startswith("bestmove ") for line in early), early[-10:]
    engine.send("stop")
    output = engine.until(lambda line: line.startswith("bestmove "), 5)
    assert bestmove(output) == "a1a8"
    print("PASS infinite waits for stop")

    engine.send("position startpos")
    engine.send("go ponder")
    time.sleep(0.1)
    engine.send("ponderhit")
    output = engine.until(lambda line: line.startswith("bestmove "), 5)
    assert bestmove(output) != "0000"
    print("PASS ponderhit termination")

    engine.close()
    print("ALL UCI EDGE TESTS PASSED")


if __name__ == "__main__":
    main()
