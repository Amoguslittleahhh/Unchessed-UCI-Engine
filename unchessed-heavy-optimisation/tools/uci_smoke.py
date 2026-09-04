#!/usr/bin/env python3
"""UCI protocol smoke test for the Unchessed engines.

Usage: python uci_smoke.py <path-to-engine>
Exercises the handshake, search, stop handling, options, and adapter logging.
"""
import subprocess
import sys
import time
import threading
import queue


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.q = queue.Queue()
        self.log = []
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        for line in self.p.stdout:
            line = line.rstrip("\n")
            self.log.append(line)
            self.q.put(line)

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def expect(self, predicate, timeout=10.0, desc=""):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.q.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                break
            if predicate(line):
                return line
        raise AssertionError(f"timeout waiting for: {desc}\nlast lines: {self.log[-15:]}")

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main():
    path = sys.argv[1]
    ok = 0

    e = Engine(path)

    # 1. handshake
    e.send("uci")
    idline = e.expect(lambda l: l.startswith("id name"), desc="id name")
    e.expect(lambda l: l == "uciok", desc="uciok")
    print(f"PASS handshake ({idline})")
    ok += 1

    # 2. isready
    e.send("isready")
    e.expect(lambda l: l == "readyok", desc="readyok")
    print("PASS isready")
    ok += 1

    # 3. options accepted silently
    e.send("setoption name Hash value 64")
    e.send("setoption name Troll value Off")
    e.send("setoption name UCI_Opponent value GM 3644 computer Stockfish 16")
    e.send("isready")
    e.expect(lambda l: l == "readyok", desc="readyok after setoption")
    print("PASS setoption (incl. UCI_Opponent)")
    ok += 1

    # 4. fixed-time search returns a legal-looking bestmove
    e.send("ucinewgame")
    e.send("position startpos")
    e.send("go movetime 1000")
    t0 = time.time()
    bm = e.expect(lambda l: l.startswith("bestmove"), timeout=6, desc="bestmove")
    dt = time.time() - t0
    mv = bm.split()[1]
    assert len(mv) in (4, 5) and mv != "0000", f"bad bestmove {bm}"
    assert dt < 4.0, f"movetime 1000 took {dt:.1f}s"
    print(f"PASS go movetime ({bm} in {dt:.2f}s)")
    ok += 1

    # 5. clock-based game search after some moves
    e.send("position startpos moves e2e4 e7e5 g1f3 b8c6")
    e.send("go wtime 60000 btime 60000 winc 1000 binc 1000")
    bm = e.expect(lambda l: l.startswith("bestmove"), timeout=15, desc="bestmove clocks")
    print(f"PASS go with clocks ({bm})")
    ok += 1

    # 6. go infinite + stop must yield bestmove promptly
    e.send("position startpos")
    e.send("go infinite")
    time.sleep(1.0)
    e.send("stop")
    bm = e.expect(lambda l: l.startswith("bestmove"), timeout=5, desc="bestmove after stop")
    print(f"PASS go infinite / stop ({bm})")
    ok += 1

    # 7. mate position: engine must find mate score
    e.send("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    e.send("go depth 6")
    e.expect(lambda l: "score mate 1" in l, timeout=10, desc="score mate 1")
    bm = e.expect(lambda l: l.startswith("bestmove"), timeout=5, desc="bestmove mate")
    assert bm.split()[1] == "a1a8", f"expected a1a8 got {bm}"
    print(f"PASS mate detection ({bm})")
    ok += 1

    # 8. stalemate/checkmate position handling: no legal moves -> bestmove 0000
    e.send("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # black is stalemated
    e.send("go movetime 200")
    bm = e.expect(lambda l: l.startswith("bestmove"), timeout=5, desc="bestmove stalemate")
    assert bm.split()[1] == "0000", f"expected 0000 got {bm}"
    print(f"PASS no-legal-moves handling ({bm})")
    ok += 1

    e.quit()

    # 9. adapter logging appeared somewhere along the way
    strings = [l for l in e.log if l.startswith("info string [Unchessed]")]
    assert strings, "expected [Unchessed] info string logs"
    print(f"PASS adapter logging ({len(strings)} [Unchessed] log lines)")
    ok += 1

    print(f"\nALL {ok} SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
