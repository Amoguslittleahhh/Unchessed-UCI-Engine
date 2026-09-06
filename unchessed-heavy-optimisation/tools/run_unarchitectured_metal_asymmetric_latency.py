#!/usr/bin/env python3
"""Real asymmetric latency probe for the unarchitectured-metal series.

Runs the checked-out Unchessed adapter against a distinct Stockfish binary through UCI. The
opponent is deliberately declared as an unknown human so the detector must
use live move-quality and clock evidence rather than the known-engine table.
Each arm receives the same opening and engine settings. The script records
all UCI output, move histories, first suspect reason, first Full-mode
telemetry event, observation coverage, and low-time skips; it does not simulate
observations. When `--clock-ms` is supplied, the driver uses a real UCI clock
and refuses a starting clock below the adapter's 10-second observation floor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import chess

TELEMETRY_RE = re.compile(r"info string \[UnchessedTelemetry\] (.*)")
MODE_RE = re.compile(r"\bmode_after=(\w+)")
REASON_RE = re.compile(r"\bsuspect_reason=([^ ]+)")
PLY_RE = re.compile(r"\bply=(\d+)")

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
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.log = log_path.open("w", encoding="utf-8")

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, marker: str, timeout: float = 20.0) -> list[str]:
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
        raise TimeoutError(f"timed out waiting for {marker!r}; last lines={lines[-5:]}")

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        finally:
            self.log.close()


def set_common(engine: Uci, unarchitectured_file: Path, fusion: bool) -> None:
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
    ]
    for option in options:
        engine.send(option)
    # The Metal runtime is explicitly enabled for this series-scoped probe.
    if unarchitectured_file.exists():
        engine.send("setoption name UnarchitecturedHint value true")
        engine.send(f"setoption name UnarchitecturedFile value {unarchitectured_file}")
    engine.send("isready")
    engine.read_until("readyok")


def run_game(
    adapter: Path,
    stockfish: Path,
    output_dir: Path,
    game_id: int,
    fusion: bool,
    unarchitectured_file: Path,
    max_plies: int,
    movetime_ms: int,
    clock_ms: int,
) -> dict:
    arm = "fusion" if fusion else "standard"
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    unchessed = Uci([str(adapter)], arm_dir / f"game_{game_id:02d}_unchessed.log")
    sf = Uci([str(stockfish)], arm_dir / f"game_{game_id:02d}_stockfish.log")
    board = chess.Board()
    opening = OPENINGS[(game_id - 1) % len(OPENINGS)]
    moves: list[str] = []
    telemetry: list[str] = []
    first_suspect: dict | None = None
    first_full: dict | None = None
    observations = 0
    skipped_low_time = 0
    skipped_other = 0
    clocks = {chess.WHITE: clock_ms, chess.BLACK: clock_ms}
    try:
        set_common(unchessed, unarchitectured_file, fusion)
        sf.send("uci")
        sf.read_until("uciok")
        sf.send("setoption name Threads value 1")
        sf.send("setoption name Hash value 64")
        sf.send("isready")
        sf.read_until("readyok")
        for engine in (unchessed, sf):
            engine.send("ucinewgame")
            engine.send("isready")
            engine.read_until("readyok")
        for move in opening:
            board.push_uci(move)
            moves.append(move)
        move_text = " ".join(moves)
        unchessed.send(f"position startpos moves {move_text}")
        sf.send(f"position startpos moves {move_text}")
        unchessed.send("isready")
        unchessed.read_until("readyok")
        sf.send("isready")
        sf.read_until("readyok")
        while len(moves) < max_plies and not board.is_game_over(claim_draw=False):
            white_to_move = board.turn == chess.WHITE
            engine = unchessed if white_to_move else sf
            engine.send(f"position startpos moves {' '.join(moves)}")
            started = time.monotonic()
            if clock_ms > 0:
                engine.send(
                    f"go wtime {clocks[chess.WHITE]} btime {clocks[chess.BLACK]} winc 0 binc 0"
                )
            else:
                engine.send(f"go movetime {movetime_ms}")
            lines = engine.read_until("bestmove", timeout=30.0)
            if clock_ms > 0:
                clocks[board.turn] = max(0, clocks[board.turn] - int((time.monotonic() - started) * 1000))
            bestmove = next((line.split()[1] for line in reversed(lines) if line.startswith("bestmove ")), None)
            if not bestmove or bestmove == "0000":
                break
            try:
                board.push_uci(bestmove)
            except ValueError:
                break
            moves.append(bestmove)
            for line in lines:
                match = TELEMETRY_RE.search(line)
                if not match:
                    continue
                payload = match.group(1)
                telemetry.append(payload)
                if "event=opponent_observation" in payload:
                    observations += 1
                elif "event=observation_skipped" in payload:
                    if "reason=low_time" in payload:
                        skipped_low_time += 1
                    else:
                        skipped_other += 1
                ply = int(PLY_RE.search(payload).group(1)) if PLY_RE.search(payload) else len(moves)
                reason_match = REASON_RE.search(payload)
                mode_match = MODE_RE.search(payload)
                if first_suspect is None and "suspect=1" in payload:
                    first_suspect = {"ply": ply, "reason": reason_match.group(1) if reason_match else None, "payload": payload}
                if first_full is None and mode_match and mode_match.group(1) == "FULL":
                    first_full = {"ply": ply, "payload": payload}
        return {
            "game": game_id,
            "arm": arm,
            "opening": opening,
            "plies": len(moves),
            "moves": moves,
            "first_suspect": first_suspect,
            "first_full": first_full,
            "telemetry_count": len(telemetry),
            "observations": observations,
            "skipped_low_time": skipped_low_time,
            "skipped_other": skipped_other,
            "clock_start_ms": clock_ms if clock_ms > 0 else None,
            "clock_remaining_ms": min(clocks.values()) if clock_ms > 0 else None,
        }
    finally:
        unchessed.close()
        sf.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/games/stockfish"))
    parser.add_argument("--metal-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-arm", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=40)
    parser.add_argument("--movetime-ms", type=int, default=80)
    parser.add_argument("--clock-ms", type=int, default=0, help="initial real clock per side; must be at least 10000 when set")
    args = parser.parse_args()
    if args.clock_ms and args.clock_ms < 10_000:
        parser.error("--clock-ms must be at least 10000 because low-time observation suppression begins below 10 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    game_id = 1
    for fusion in (False, True):
        for _ in range(args.games_per_arm):
            results.append(run_game(args.adapter, args.stockfish, args.output, game_id, fusion, args.metal_file, args.max_plies, args.movetime_ms, args.clock_ms))
            game_id += 1
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
