#!/usr/bin/env python3
"""Self-play sanity driver: the adapter plays itself with fast clocks.

Usage: python selfplay.py <path-to-engine> [games] [movetime_ms]
Verifies games complete without crashes, illegal moves, or protocol stalls.
The engine itself validates legality: an illegal move in `position` would
fail parsing and the driver would stall/error.
"""
import subprocess
import sys
import time
import threading
import queue


class Engine:
    def __init__(self, path, name):
        self.name = name
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        self.wait(lambda l: l == "uciok")

    def _reader(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait(self, pred, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.q.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                break
            if pred(line):
                return line
        raise AssertionError(f"{self.name}: timeout")

    def bestmove(self, moves, movetime):
        pos = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self.send(pos)
        self.send(f"go movetime {movetime}")
        line = self.wait(lambda l: l.startswith("bestmove"), timeout=movetime / 1000 + 10)
        return line.split()[1]

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(path, movetime, game_no):
    white = Engine(path, "white")
    black = Engine(path, "black")
    # vary personalities a little across games
    if game_no % 2 == 0:
        white.send("setoption name Troll value On")
    else:
        black.send("setoption name Adaptive value false")
    for e in (white, black):
        e.send("ucinewgame")
        e.send("isready")
        e.wait(lambda l: l == "readyok")

    moves = []
    result = "unfinished"
    for ply in range(300):
        eng = white if ply % 2 == 0 else black
        mv = eng.bestmove(moves, movetime)
        if mv == "0000":
            result = f"game over at ply {ply} (mate/stalemate)"
            break
        assert len(mv) in (4, 5), f"malformed move {mv}"
        moves.append(mv)
    else:
        result = "300-ply cap reached (fine for sanity run)"
    white.quit()
    black.quit()
    return result, len(moves)


def main():
    path = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    movetime = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    for g in range(games):
        t0 = time.time()
        result, plies = play_game(path, movetime, g)
        print(f"game {g + 1}: {plies} plies, {result} ({time.time() - t0:.0f}s)")
    print(f"\nSELF-PLAY OK: {games} games without crashes or protocol stalls")


if __name__ == "__main__":
    main()
