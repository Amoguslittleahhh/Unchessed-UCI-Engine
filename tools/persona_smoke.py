#!/usr/bin/env python3
"""Black-box regression tests for opponent identity and persona safety.

Usage: python tools/persona_smoke.py <path-to-adapter>
"""

import queue
import re
import subprocess
import sys
import threading
import time


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
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        handshake = self.until(lambda line: line == "uciok", 15)
        assert any(
            "option name UCI_Elo type spin default 2400 min 100 max 2600" in line
            for line in handshake
        ), handshake

    def _reader(self):
        for line in self.p.stdout:
            line = line.rstrip("\n")
            self.log.append(line)
            self.q.put(line)

    def send(self, command):
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()

    def until(self, predicate, timeout=10):
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

    def go(self, position, command, timeout=15):
        self.send(position)
        started = time.monotonic()
        self.send(command)
        output = self.until(lambda line: line.startswith("bestmove "), timeout)
        return output, time.monotonic() - started

    def close(self):
        if self.p.poll() is None:
            self.send("quit")
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()


def mode(output):
    line = next((line for line in reversed(output) if " mode=" in line), "")
    match = re.search(r"mode=([A-Z]+)", line)
    return match.group(1) if match else None


def test_identity_persistence(path):
    engine = Engine(path)
    engine.send("setoption name OwnBook value false")
    engine.send("setoption name UCI_Opponent value GM - computer Stockfish 16")
    engine.send("ucinewgame")
    output, _ = engine.go("position startpos", "go depth 3 wtime 5000 btime 5000")
    assert mode(output) == "FULL", output[-10:]
    engine.close()
    print("PASS identity survives ucinewgame")


def test_limited_engine_is_separate_from_identity(path):
    engine = Engine(path)
    engine.send("setoption name OwnBook value false")
    engine.send("setoption name UCI_Opponent value GM 1500 computer Stockfish 16")
    engine.send("ucinewgame")
    output, _ = engine.go("position startpos", "go depth 3 wtime 5000 btime 5000")
    persona = next(line for line in output if " mode=" in line)
    assert mode(output) == "MATCH", persona
    assert "type=known engine" in persona, persona
    engine.close()
    print("PASS limited engine identity/strength separation")


def test_contextual_persona_responses(path):
    human = Engine(path)
    human.send("setoption name OwnBook value false")
    human.send("setoption name UCI_Opponent value GM 2400 human TestGM")
    queen_rich = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 32"
    queenless = "position fen rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 32"
    clinch, _ = human.go(queen_rich, "go depth 3 wtime 5000 btime 5000")
    normal, _ = human.go(queenless, "go depth 3 wtime 5000 btime 5000")
    assert mode(clinch) == "CLINCH", clinch[-10:]
    assert mode(normal) == "MATCH", normal[-10:]
    human.close()

    engine = Engine(path)
    engine.send("setoption name OwnBook value false")
    engine.send("setoption name UCI_Opponent value - - computer Stockfish")
    losing = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 20"
    defense, _ = engine.go(losing, "go depth 3 wtime 5000 btime 5000")
    assert mode(defense) == "DEFEND", defense[-10:]
    engine.close()
    print("PASS contextual human/engine persona responses")


def test_fixed_strength_precedence(path):
    engine = Engine(path)
    for command in (
        "setoption name OwnBook value false",
        "setoption name UCI_LimitStrength value true",
        "setoption name UCI_Elo value 800",
    ):
        engine.send(command)
    losing = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 10"
    late = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 30"
    first, _ = engine.go(losing, "go depth 3 wtime 5000 btime 5000")
    second, _ = engine.go(late, "go depth 3 wtime 5000 btime 5000")
    assert mode(first) == mode(second) == "MATCH", (first[-5:], second[-5:])
    engine.close()
    print("PASS fixed strength overrides DEFEND/CLINCH")


def measurement_at(path, clock):
    engine = Engine(path)
    engine.send("setoption name OwnBook value false")
    # With complete ECO coverage every legal first move is book-known. Warm the
    # observation watermark, then use an intentionally non-book rook move.
    engine.go(
        "position startpos moves a2a3",
        f"go depth 1 wtime {clock} btime {clock}",
        15,
    )
    output, elapsed = engine.go(
        "position startpos moves a2a3 a7a6 a1a2",
        f"go depth 2 wtime {clock} btime {clock}",
        15,
    )
    observed = any("opponent move " in line for line in output)
    engine.close()
    return elapsed, observed


def test_smooth_measurement_budget(path):
    below = measurement_at(path, 9_999)
    above = measurement_at(path, 10_000)
    assert below[1] and above[1], (below, above)
    ratio = max(below[0], above[0]) / max(0.001, min(below[0], above[0]))
    assert ratio < 10.0 and abs(below[0] - above[0]) < 0.15, (below, above)
    print(f"PASS smooth measurement budget ({below[0]:.3f}s/{above[0]:.3f}s)")


def test_known_engine_auto_troll_lock(path):
    engine = Engine(path)
    engine.send("ucinewgame")
    engine.send("setoption name Troll value Auto")
    engine.send("setoption name UCI_Opponent value GM 1500 computer Stockfish 16")
    for _ in range(30):
        output, _ = engine.go("position startpos moves e2e4 e7e5", "go movetime 10")
        assert not any("[troll" in line for line in output), output[-10:]
    engine.close()
    print("PASS known engine Auto-troll lock (30/30)")


def test_short_movetime_does_not_run_measurement(path):
    engine = Engine(path)
    engine.send("setoption name OwnBook value false")
    output, elapsed = engine.go("position startpos moves a2a3", "go movetime 100")
    assert not any("opponent move " in line for line in output), output[-10:]
    assert elapsed < 0.35, elapsed
    engine.close()
    print(f"PASS short movetime safety ({elapsed:.3f}s)")


def main():
    path = sys.argv[1]
    test_identity_persistence(path)
    test_limited_engine_is_separate_from_identity(path)
    test_contextual_persona_responses(path)
    test_fixed_strength_precedence(path)
    test_smooth_measurement_budget(path)
    test_known_engine_auto_troll_lock(path)
    test_short_movetime_does_not_run_measurement(path)
    print("ALL PERSONA SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
