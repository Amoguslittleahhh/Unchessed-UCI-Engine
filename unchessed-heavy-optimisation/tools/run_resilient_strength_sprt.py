#!/usr/bin/env python3
"""Real paired-game strength SPRT for standard versus resilient detection.

Each pair uses two fresh Unchessed adapter processes with identical Unarchitectured
Metal settings; one has AcceleratedDetection off and the other on. Colors are
swapped in the second game of each pair. Outcomes are scored from the resilient
arm's perspective. The harness uses real UCI clocks and records telemetry.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

import chess

TELEMETRY_RE = re.compile(r"info string \[UnchessedTelemetry\] (.*)")

OPENINGS = [
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
    ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"],
    ["g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
]


class Uci:
    def __init__(self, command: list[str], log_path: Path):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.log = log_path.open("w", encoding="utf-8")

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, marker: str, timeout: float = 30.0) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            self.log.write(line + "\n")
            self.log.flush()
            lines.append(line)
            if marker in line:
                return lines
        raise TimeoutError(f"timeout waiting for {marker!r}: {lines[-5:]}")

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        finally:
            self.log.close()


def configure(engine: Uci, metal_file: Path, fusion: bool) -> None:
    engine.send("uci")
    engine.read_until("uciok")
    options = [
        "setoption name Threads value 1",
        "setoption name Hash value 64",
        "setoption name Adaptive value true",
        "setoption name OwnBook value false",
        "setoption name AdapterTelemetry value true",
        "setoption name EngineDetectV2 value false",
        f"setoption name AcceleratedDetection value {'true' if fusion else 'false'}",
        "setoption name UCI_Opponent value - - human UnknownOpponent",
        "setoption name UnarchitecturedHint value true",
        f"setoption name UnarchitecturedFile value {metal_file}",
    ]
    for option in options:
        engine.send(option)
    engine.send("isready")
    engine.read_until("readyok")


def score_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    # Three-outcome SPRT approximation from the real match WDL stream.
    n = wins + draws + losses
    if n == 0:
        return 0.0
    def probs(elo: float) -> tuple[float, float, float]:
        s = score_from_elo(elo)
        draw = 0.38
        win = (s - draw / 2.0) * 0.98
        loss = 1.0 - draw - win
        return max(win, 1e-9), draw, max(loss, 1e-9)
    p0 = probs(elo0)
    p1 = probs(elo1)
    return sum(c * math.log(a / b) for c, a, b in zip((wins, draws, losses), p1, p0))


def run_game(adapter: Path, output: Path, game_id: int, white_fusion: bool,
             metal_file: Path, clock_ms: int, increment_ms: int, max_plies: int) -> dict:
    game_dir = output / f"game_{game_id:02d}"
    game_dir.mkdir(parents=True, exist_ok=True)
    engines = {
        chess.WHITE: Uci([str(adapter)], game_dir / "white.log"),
        chess.BLACK: Uci([str(adapter)], game_dir / "black.log"),
    }
    fusion_by_color = {chess.WHITE: white_fusion, chess.BLACK: not white_fusion}
    for color, engine in engines.items():
        configure(engine, metal_file, fusion_by_color[color])
    board = chess.Board()
    opening = OPENINGS[(game_id - 1) % len(OPENINGS)]
    moves: list[str] = []
    clocks = {chess.WHITE: clock_ms, chess.BLACK: clock_ms}
    first_full: dict[bool, int | None] = {False: None, True: None}
    telemetry_counts = {False: 0, True: 0}
    skipped_low_time = {False: 0, True: 0}
    try:
        for move in opening:
            board.push_uci(move)
            moves.append(move)
        while len(moves) < max_plies and not board.is_game_over(claim_draw=False):
            color = board.turn
            engine = engines[color]
            engine.send(f"position startpos moves {' '.join(moves)}")
            started = time.monotonic()
            engine.send(f"go wtime {clocks[chess.WHITE]} btime {clocks[chess.BLACK]} winc {increment_ms} binc {increment_ms}")
            lines = engine.read_until("bestmove", timeout=60.0)
            clocks[color] = max(0, clocks[color] - int((time.monotonic() - started) * 1000) + increment_ms)
            for line in lines:
                match = TELEMETRY_RE.search(line)
                if not match:
                    continue
                payload = match.group(1)
                arm = fusion_by_color[color]
                if "event=opponent_observation" in payload:
                    telemetry_counts[arm] += 1
                if "event=observation_skipped" in payload and "reason=low_time" in payload:
                    skipped_low_time[arm] += 1
                if "event=persona_decision" in payload and "mode_after=FULL" in payload and first_full[arm] is None:
                    first_full[arm] = len(moves)
            bestmove = next((line.split()[1] for line in reversed(lines) if line.startswith("bestmove ")), None)
            if not bestmove or bestmove == "0000":
                break
            board.push_uci(bestmove)
            moves.append(bestmove)
        outcome = board.outcome(claim_draw=False)
        result = "*" if outcome is None else outcome.result()
        if result == "1-0":
            resilient_score = 1 if white_fusion else -1
        elif result == "0-1":
            resilient_score = -1 if white_fusion else 1
        else:
            resilient_score = 0
        return {
            "game": game_id,
            "white_fusion": white_fusion,
            "result": result,
            "resilient_score": resilient_score,
            "plies": len(moves),
            "moves": moves,
            "first_full_standard": first_full[False],
            "first_full_resilient": first_full[True],
            "observations_standard": telemetry_counts[False],
            "observations_resilient": telemetry_counts[True],
            "low_time_skips_standard": skipped_low_time[False],
            "low_time_skips_resilient": skipped_low_time[True],
        }
    finally:
        for engine in engines.values():
            engine.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--metal-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--clock-ms", type=int, default=60000)
    ap.add_argument("--increment-ms", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=80)
    args = ap.parse_args()
    if args.clock_ms < 10_000:
        ap.error("--clock-ms must be at least 10000")
    args.output.mkdir(parents=True, exist_ok=True)
    games: list[dict] = []
    for pair in range(args.pairs):
        games.append(run_game(args.adapter, args.output, 2 * pair + 1, True, args.metal_file, args.clock_ms, args.increment_ms, args.max_plies))
        games.append(run_game(args.adapter, args.output, 2 * pair + 2, False, args.metal_file, args.clock_ms, args.increment_ms, args.max_plies))
    wins = sum(g["resilient_score"] == 1 for g in games)
    draws = sum(g["resilient_score"] == 0 for g in games)
    losses = sum(g["resilient_score"] == -1 for g in games)
    upper = math.log((1 - 0.05) / 0.05)
    lower = math.log(0.05 / (1 - 0.05))
    out = {
        "protocol": {"pairs": args.pairs, "games": len(games), "clock_ms": args.clock_ms, "increment_ms": args.increment_ms, "max_plies": args.max_plies, "elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.05},
        "games": games,
        "wdl_resilient_perspective": {"wins": wins, "draws": draws, "losses": losses},
        "sprt": {"llr": llr(wins, draws, losses, 0, 5), "lower": lower, "upper": upper, "decision": "accept_h1" if llr(wins, draws, losses, 0, 5) >= upper else "reject_h0" if llr(wins, draws, losses, 0, 5) <= lower else "continue"},
    }
    (args.output / "sprt_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
