#!/usr/bin/env python3
"""Real Maia-3 false-positive test using the OFFICIAL maia3-uci binary
(CSSLab/maia3, https://github.com/CSSLab/maia3), not the simplified
ONNX export tools/maia3_false_positive_test.py used.

Both sides are real UCI subprocesses this time: unchessed-adapter and
the official maia3-5m engine, driven with real wtime/btime clocks so
the adapter's own low-time gate behaves normally. Maia's SelfElo/
OppoElo UCI options set the elo band, and Temperature=1.0 enables the
same human-like stochastic sampling documented for the engine (its
default is Temperature=0.0 / argmax, which would make it play
deterministically and not represent human-like inconsistency at all).

This exists to resolve a real discrepancy: the simplified-ONNX 40-game
run (scripts/research/manus_resilient_only_maia3_recheck.md) found the
resilient-only AcceleratedDetection restructure did not reduce the
Maia-3 false-confirmation rate (14/20, 70%), contradicting Manus's own
5-game official-binary result (1/5, 20%). Manus's own reproducibility
doc (manus/research-facilities@7025872) flagged the ONNX export's lack
of one-ply opponent-response ranking as the likely explanation and
asked for a same-scale run on the official binary -- this is that run.

Usage:
    python maia3_uci_false_positive_test.py <unchessed_adapter_path> \\
        [--maia-cmd "maia3-5m"] [--nnue-path PATH]
        [--games-per-arm-per-band N] [--elo-bands E1,E2,...]
        [--clock-ms MS] [--inc-ms MS] [--max-plies N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import shlex
import subprocess
import sys
import threading
import time

try:
    import chess
except ImportError:
    sys.exit("requires python-chess: pip install chess")

DEFAULT_OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["c2c4", "e7e5"],
    ["g1f3", "d7d5"], ["e2e4", "c7c5"], ["d2d4", "g8f6"], ["e2e4", "e7e6"],
    ["c2c4", "c7c5"], ["g1f3", "g8f6"],
]

TELEM_RE = re.compile(r"info string \[UnchessedTelemetry\] (.+)")


class UciEngine:
    def __init__(self, cmd, setup_cmds=None, capture_telemetry=False, cwd=None):
        self.capture_telemetry = capture_telemetry
        self.telemetry_lines: list[str] = []
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=cwd,
        )
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", 15)
        for cmd_ in (setup_cmds or []):
            self._send(cmd_)
        self._send("isready")
        self._wait_for("readyok", 60)  # Maia3 loads its checkpoint here

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
        self._wait_for("readyok", 30)

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
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def parse_kv(line):
    fields = {}
    for token in line.split(" "):
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k] = v
    return fields


def full_confirmations(telemetry_lines):
    out = []
    for line in telemetry_lines:
        f = parse_kv(line)
        if f.get("event") == "persona_decision" and f.get("mode_after") == "FULL":
            out.append(int(f.get("ply", -1)))
    return out


def suspect_reasons(telemetry_lines):
    reasons = {}
    for line in telemetry_lines:
        f = parse_kv(line)
        if f.get("suspect") == "1" and f.get("suspect_reason"):
            r = f["suspect_reason"]
            reasons[r] = reasons.get(r, 0) + 1
    return reasons


def observation_stats(telemetry_lines):
    total = 0
    low_time_skips = 0
    for line in telemetry_lines:
        f = parse_kv(line)
        if f.get("event") == "observation_skipped":
            total += 1
            if f.get("reason") == "low_time":
                low_time_skips += 1
        elif f.get("event") == "opponent_observation":
            total += 1
    return total, low_time_skips


def play_game(adapter, maia, opening, adapter_is_white, clock_ms, inc_ms, max_plies):
    adapter.newgame()
    maia.newgame()
    board = chess.Board()
    moves = list(opening)
    for mv in moves:
        board.push_uci(mv)
    ply = len(moves)
    wtime, btime = clock_ms, clock_ms
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        adapter_turn = (board.turn == chess.WHITE) == adapter_is_white
        engine = adapter if adapter_turn else maia
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
    return board.result(claim_draw=True), ply


def run_arm(adapter_path, nnue_path, maia_cmd, accelerated, elo_band, n_games,
            clock_ms, inc_ms, max_plies):
    adapter_setup = []
    if nnue_path:
        adapter_setup.append(f"setoption name EvalFile value {nnue_path}")
    adapter_setup += [
        "setoption name Adaptive value true",
        "setoption name OwnBook value false",
        "setoption name AdapterTelemetry value true",
        f"setoption name AcceleratedDetection value {'true' if accelerated else 'false'}",
        "setoption name Threads value 1",
        "setoption name Hash value 64",
        "setoption name UCI_Opponent value - - human UnknownOpponent",
    ]
    maia_setup = [
        f"setoption name SelfElo value {elo_band}",
        f"setoption name OppoElo value {elo_band}",
        "setoption name Temperature value 1.0",
        "setoption name MultiPV value 1",
    ]

    results = []
    for i in range(n_games):
        opening = DEFAULT_OPENINGS[i % len(DEFAULT_OPENINGS)]
        adapter_is_white = (i % 2 == 0)
        adapter = UciEngine([adapter_path], adapter_setup, capture_telemetry=True)
        maia = UciEngine(shlex.split(maia_cmd), maia_setup, capture_telemetry=False)
        result, plies = play_game(adapter, maia, opening, adapter_is_white,
                                   clock_ms, inc_ms, max_plies)
        confirms = full_confirmations(adapter.telemetry_lines)
        reasons = suspect_reasons(adapter.telemetry_lines)
        obs_total, obs_low_time = observation_stats(adapter.telemetry_lines)
        rec = {
            "game": i + 1, "adapter_white": adapter_is_white, "elo_band": elo_band,
            "result": result, "plies": plies,
            "first_full_ply": confirms[0] if confirms else None,
            "suspect_reason_counts": reasons,
            "obs_total": obs_total, "obs_low_time_skips": obs_low_time,
        }
        results.append(rec)
        print(f"  arm={'accel' if accelerated else 'std'} elo={elo_band} game{i+1}: "
              f"result={result} plies={plies} first_full={rec['first_full_ply']} "
              f"obs={obs_total} low_time_skips={obs_low_time} reasons={reasons}", flush=True)
        adapter.quit()
        maia.quit()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("unchessed_path")
    parser.add_argument("--maia-cmd", default="maia3-5m", help="command to launch the official maia3-uci engine, e.g. 'maia3-5m' or 'maia3-uci --model maia3-5m'")
    parser.add_argument("--nnue-path", default=None)
    parser.add_argument("--games-per-arm-per-band", type=int, default=10)
    parser.add_argument("--elo-bands", default="1500,2000")
    parser.add_argument("--clock-ms", type=int, default=60_000)
    parser.add_argument("--inc-ms", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=48)
    parser.add_argument("--out", default="maia3_uci_false_positive_results.json")
    args = parser.parse_args()

    if args.clock_ms < 10_000:
        raise SystemExit(f"--clock-ms {args.clock_ms} is unsafely low; the adapter's low-time gate "
                          "suppresses opponent observation below its own thresholds")

    elo_bands = [int(x) for x in args.elo_bands.split(",")]

    all_results = {}
    for accelerated in (False, True):
        arm = "accelerated" if accelerated else "standard"
        all_results[arm] = []
        for band in elo_bands:
            print(f"=== {arm} arm, maia elo {band} ===", flush=True)
            res = run_arm(args.unchessed_path, args.nnue_path, args.maia_cmd, accelerated,
                          band, args.games_per_arm_per_band, args.clock_ms, args.inc_ms,
                          args.max_plies)
            all_results[arm].extend(res)

    print("\n=== SUMMARY ===")
    for arm, results in all_results.items():
        total = len(results)
        confirmed = [r for r in results if r["first_full_ply"] is not None]
        legacy_clock_games = sum(
            1 for r in results if r["suspect_reason_counts"].get("legacy_clock", 0) > 0)
        resilient_games = sum(
            1 for r in results if r["suspect_reason_counts"].get("legacy_accelerated_resilient", 0) > 0)
        fusion_games = sum(
            1 for r in results if r["suspect_reason_counts"].get("legacy_accelerated_fusion", 0) > 0)
        total_low_time = sum(r["obs_low_time_skips"] for r in results)
        print(f"{arm}: games={total} full_confirmed={len(confirmed)} "
              f"legacy_clock_games={legacy_clock_games} "
              f"resilient_games={resilient_games} fusion_games={fusion_games} "
              f"total_low_time_skips={total_low_time}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
