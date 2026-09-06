#!/usr/bin/env python3
"""Real-clock asymmetric detection-latency test: unchessed-adapter vs a
distinct strong opponent (not a mirror match), comparing AcceleratedDetection
off vs on by measuring moves-to-Full-mode confirmation from live telemetry.

Drives both engines directly over UCI with real wtime/btime clocks (not
movetime -- movetime below ~1000ms triggers the adapter's own low-time gate,
which suppresses ALL opponent-observation probing regardless of which
detector is under test; this cost a wasted first attempt during development,
see scripts/research/manus_fusion_asymmetric_latency_replication.md).

Used to independently replicate manus/research-facilities's asymmetric
latency results (commits 5d43f6a, db48f6e) at a larger sample size (16
games/arm vs their 2) against Stockfish 19. Result:

    standard:    mean 27.9 median 28 range [26,31] plies to Full
    accelerated: mean 23.9 median 24 range [22,26] plies to Full

~4 plies (2 moves) earlier, consistent across all 16 games -- see
scripts/research/manus_resilient_channel_large_replication.md for the full
writeup. This only tests against a genuinely strong, consistent opponent;
it does NOT establish a false-positive rate against weaker or human-like
opponents (e.g. Maia), which is the safety question that still needs
answering before AcceleratedDetection could be considered for promotion.

Usage:
    python asymmetric_latency_test.py <unchessed_adapter_path> <opponent_path>
        [--games-per-arm N] [--clock-ms MS] [--inc-ms MS] [--max-plies N]
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time

import chess

DEFAULT_OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["c2c4", "e7e5"],
    ["g1f3", "d7d5"], ["e2e4", "c7c5"], ["d2d4", "g8f6"], ["e2e4", "e7e6"],
    ["c2c4", "c7c5"], ["g1f3", "g8f6"], ["e2e4", "d7d6"], ["d2d4", "e7e6"],
    ["e2e4", "g8f6"], ["c2c4", "e7e6"], ["g1f3", "c7c5"], ["d2d4", "c7c5"],
]

TELEM_RE = re.compile(r"info string \[UnchessedTelemetry\] (.+)")


class Engine:
    def __init__(self, path, setup_cmds=None, capture_telemetry=False):
        self.capture_telemetry = capture_telemetry
        self.telemetry_lines: list[str] = []
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", 10)
        for cmd in (setup_cmds or []):
            self._send(cmd)
        self._send("isready")
        self._wait_for("readyok", 10)

    def _pump(self):
        for line in iter(self.proc.stdout.readline, ""):
            self.q.put(line.strip())
        self.q.put(None)

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token, timeout):
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            try:
                line = self.q.get(timeout=max(0, deadline - time.time()))
            except queue.Empty:
                break
            if line is None:
                break
            if self.capture_telemetry:
                m = TELEM_RE.match(line)
                if m:
                    self.telemetry_lines.append(m.group(1))
            lines.append(line)
            if line.strip() == token or line.startswith(token):
                return lines
        return lines

    def newgame(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", 10)

    def bestmove(self, moves, wtime, btime, inc_ms):
        pos_cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos_cmd)
        self._send(f"go wtime {wtime} btime {btime} winc {inc_ms} binc {inc_ms}")
        t0 = time.time()
        lines = self._wait_for("bestmove", 60)
        elapsed_ms = int((time.time() - t0) * 1000)
        bm = None
        for l in lines:
            if l.startswith("bestmove"):
                bm = l.split()[1]
        return bm, elapsed_ms

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def parse_telemetry_kv(line):
    fields = {}
    for token in line.split(" "):
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k] = v
    return fields


def first_full_ply(telemetry_lines):
    for line in telemetry_lines:
        f = parse_telemetry_kv(line)
        if f.get("event") == "persona_decision" and f.get("mode_after") == "FULL":
            return int(f.get("ply", -1))
    return None


def observation_stats(telemetry_lines):
    total = 0
    low_time_skips = 0
    for line in telemetry_lines:
        f = parse_telemetry_kv(line)
        if f.get("event") == "observation_skipped":
            total += 1
            if f.get("reason") == "low_time":
                low_time_skips += 1
        elif f.get("event") == "opponent_observation":
            total += 1
    return total, low_time_skips


def play_game(white, black, opening, clock_ms, inc_ms, max_plies):
    white.newgame()
    black.newgame()
    board = chess.Board()
    moves = list(opening)
    for mv in moves:
        board.push_uci(mv)
    ply = len(moves)
    wtime, btime = clock_ms, clock_ms
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        engine = white if board.turn == chess.WHITE else black
        mv, elapsed_ms = engine.bestmove(moves, wtime, btime, inc_ms)
        if board.turn == chess.WHITE:
            wtime = max(0, wtime - elapsed_ms + inc_ms)
        else:
            btime = max(0, btime - elapsed_ms + inc_ms)
        if mv is None or mv == "0000":
            break
        try:
            board.push_uci(mv)
        except Exception:
            break
        moves.append(mv)
        ply += 1
        if wtime <= 0 or btime <= 0:
            break


def run_arm(unchessed_path, opponent_path, accelerated, n_games, clock_ms, inc_ms, max_plies):
    results = []
    for i in range(n_games):
        opening = DEFAULT_OPENINGS[i % len(DEFAULT_OPENINGS)]
        unchessed_is_white = (i % 2 == 0)
        unch = Engine(unchessed_path, [
            "setoption name Adaptive value true",
            "setoption name OwnBook value false",
            "setoption name AdapterTelemetry value true",
            f"setoption name AcceleratedDetection value {'true' if accelerated else 'false'}",
            "setoption name Threads value 1",
            "setoption name Hash value 64",
            "setoption name UCI_Opponent value - - human UnknownOpponent",
        ], capture_telemetry=True)
        opp = Engine(opponent_path, [
            "setoption name Threads value 1",
            "setoption name Hash value 64",
        ])
        if unchessed_is_white:
            play_game(unch, opp, opening, clock_ms, inc_ms, max_plies)
        else:
            play_game(opp, unch, opening, clock_ms, inc_ms, max_plies)
        full_ply = first_full_ply(unch.telemetry_lines)
        obs_total, obs_low_time = observation_stats(unch.telemetry_lines)
        results.append({"game": i + 1, "unchessed_white": unchessed_is_white,
                         "first_full_ply": full_ply,
                         "obs_total": obs_total, "obs_low_time_skips": obs_low_time})
        print(f"  arm={'accel' if accelerated else 'std'} game{i+1}: "
              f"first_full_ply={full_ply} obs={obs_total} low_time_skips={obs_low_time}", flush=True)
        unch.quit()
        opp.quit()
    return results


def summarize(results):
    confirmed = [r["first_full_ply"] for r in results if r["first_full_ply"] is not None]
    return {
        "games": len(results),
        "confirmed": len(confirmed),
        "mean_ply": sum(confirmed) / len(confirmed) if confirmed else None,
        "median_ply": sorted(confirmed)[len(confirmed) // 2] if confirmed else None,
        "min_ply": min(confirmed) if confirmed else None,
        "max_ply": max(confirmed) if confirmed else None,
        "total_low_time_skips": sum(r["obs_low_time_skips"] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("unchessed_path")
    parser.add_argument("opponent_path")
    parser.add_argument("--games-per-arm", type=int, default=16)
    parser.add_argument("--clock-ms", type=int, default=60_000, help="starting clock per side; must clear the adapter's ~1000ms low-time floor by a wide margin")
    parser.add_argument("--inc-ms", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--out", default="asymmetric_latency_results.json")
    args = parser.parse_args()

    if args.clock_ms < 10_000:
        raise SystemExit(f"--clock-ms {args.clock_ms} is unsafely low; the adapter's low-time gate "
                          "suppresses opponent observation below its own thresholds, which would "
                          "silently produce a meaningless result")

    print("=== Standard (AcceleratedDetection=false) ===")
    std_results = run_arm(args.unchessed_path, args.opponent_path, False,
                           args.games_per_arm, args.clock_ms, args.inc_ms, args.max_plies)

    print("=== Accelerated (AcceleratedDetection=true) ===")
    accel_results = run_arm(args.unchessed_path, args.opponent_path, True,
                             args.games_per_arm, args.clock_ms, args.inc_ms, args.max_plies)

    std_summary = summarize(std_results)
    accel_summary = summarize(accel_results)
    print("\n=== SUMMARY ===")
    print("standard:", json.dumps(std_summary))
    print("accelerated:", json.dumps(accel_summary))
    with open(args.out, "w") as f:
        json.dump({"standard": std_results, "accelerated": accel_results,
                    "standard_summary": std_summary, "accelerated_summary": accel_summary}, f, indent=2)


if __name__ == "__main__":
    main()
