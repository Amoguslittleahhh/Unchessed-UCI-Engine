#!/usr/bin/env python3
"""Real Maia-3 false-positive test for AcceleratedDetection.

Plays real games between unchessed-adapter (real UCI subprocess, real
wtime/btime clock so its own low-time gate behaves normally) and the
real Maia-3 model (mcognetta/simple-maia3-inference ONNX export --
Maia-3 has no UCI binary, so its moves are sampled directly from the
model's elo-conditioned policy via onnxruntime, temperature=1, exactly
as tools/selfplay_elo_mixer.py does; see that file's docstring for the
model interface and dependency setup, including `fetch-model`).

Compares standard (AcceleratedDetection=false) against accelerated
(=true, with the clock-corroboration fix from commit f3cf694 / ported
from manus/research-facilities@f23b915) across two elo bands, watching
for false Full-mode confirmations against an opponent that is
human-like by construction, not a strong engine.

Result (40 games: 2 elo bands x 10 games/arm/band, 60s/0inc clocks,
48-ply cap, NNUE eval, scripts/research/manus_maia3_false_positive_result.md
has the full writeup):

    standard:    20/20 (100%) false-confirmed Full, mean ply 10.0 (range 6-19)
    accelerated: 13/20 ( 65%) false-confirmed Full, mean ply 24.6 (range 19-36)

The fix is a real, measurable improvement (100%->65% false-positive
rate, confirmation delayed ~2.5x) but does not make AcceleratedDetection
false-positive-safe against Maia-3: the corroboration gate
(estimate_elo>=2450 etc.) is satisfiable often enough that Maia-3's
actual played move quality, per Unchessed's own cp-loss estimator,
clears the bar in most games. The dedicated resilient-channel reason
(legacy_accelerated_resilient) never independently fired in any of the
40 games -- confirmations came through the now-gated legacy_clock path
or legacy_accelerated_fusion instead.

Usage:
    python maia3_false_positive_test.py <unchessed_adapter_path> \\
        <maia3_onnx_model_path> [--nnue-path PATH]
        [--games-per-arm-per-band N] [--elo-bands E1,E2,...]
        [--clock-ms MS] [--inc-ms MS] [--max-plies N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import chess
except ImportError:
    sys.exit("requires python-chess: pip install chess")

try:
    from selfplay_elo_mixer import Maia3
except ImportError:
    sys.exit(
        "requires tools/selfplay_elo_mixer.py's Maia3 class on the path; "
        "run this alongside a checkout that has it, e.g.\n"
        "  PYTHONPATH=/path/to/unchessed-heavy-optimisation/tools "
        "python maia3_false_positive_test.py ...\n"
        "and fetch the ONNX model first with:\n"
        "  python selfplay_elo_mixer.py fetch-model --out /path/to/model"
    )

DEFAULT_OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["c2c4", "e7e5"],
    ["g1f3", "d7d5"], ["e2e4", "c7c5"], ["d2d4", "g8f6"], ["e2e4", "e7e6"],
    ["c2c4", "c7c5"], ["g1f3", "g8f6"],
]

TELEM_RE = re.compile(r"info string \[UnchessedTelemetry\] (.+)")


class Adapter:
    def __init__(self, adapter_path, accelerated, nnue_path=None, cwd=None):
        self.telemetry_lines: list[str] = []
        self.proc = subprocess.Popen(
            [adapter_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=cwd,
        )
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", 10)
        setup = []
        if nnue_path:
            setup.append(f"setoption name EvalFile value {nnue_path}")
        setup += [
            "setoption name Adaptive value true",
            "setoption name OwnBook value false",
            "setoption name AdapterTelemetry value true",
            f"setoption name AcceleratedDetection value {'true' if accelerated else 'false'}",
            "setoption name Threads value 1",
            "setoption name Hash value 64",
            "setoption name UCI_Opponent value - - human UnknownOpponent",
        ]
        for cmd in setup:
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


def play_game(adapter, model, rng, opening, adapter_is_white, maia_elo,
               clock_ms, inc_ms, max_plies):
    adapter.newgame()
    board = chess.Board()
    moves = list(opening)
    for mv in moves:
        board.push_uci(mv)
    ply = len(moves)
    wtime, btime = clock_ms, clock_ms
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        adapter_turn = (board.turn == chess.WHITE) == adapter_is_white
        if adapter_turn:
            mv, elapsed_ms = adapter.bestmove(moves, wtime, btime, inc_ms)
        else:
            move_probs, _ldw, _top1 = model.probs(board.fen(), maia_elo, maia_elo)
            t0 = time.time()
            mv = rng.choices(list(move_probs), weights=list(move_probs.values()))[0]
            elapsed_ms = max(1, int((time.time() - t0) * 1000))
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


def run_arm(adapter_path, nnue_path, model, accelerated, elo_band, n_games,
            seed, clock_ms, inc_ms, max_plies):
    rng = random.Random(seed)
    adapter = Adapter(adapter_path, accelerated, nnue_path)
    results = []
    for i in range(n_games):
        opening = DEFAULT_OPENINGS[i % len(DEFAULT_OPENINGS)]
        adapter_is_white = (i % 2 == 0)
        result, plies = play_game(adapter, model, rng, opening, adapter_is_white,
                                   elo_band, clock_ms, inc_ms, max_plies)
        confirms = full_confirmations(adapter.telemetry_lines)
        reasons = suspect_reasons(adapter.telemetry_lines)
        rec = {
            "game": i + 1, "adapter_white": adapter_is_white, "elo_band": elo_band,
            "result": result, "plies": plies,
            "first_full_ply": confirms[0] if confirms else None,
            "suspect_reason_counts": reasons,
        }
        results.append(rec)
        print(f"  arm={'accel' if accelerated else 'std'} elo={elo_band} game{i+1}: "
              f"result={result} plies={plies} first_full={rec['first_full_ply']} "
              f"reasons={reasons}", flush=True)
        adapter.telemetry_lines.clear()
    adapter.quit()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("unchessed_path")
    parser.add_argument("maia3_model_path")
    parser.add_argument("--nnue-path", default=None)
    parser.add_argument("--games-per-arm-per-band", type=int, default=10)
    parser.add_argument("--elo-bands", default="1500,2000")
    parser.add_argument("--clock-ms", type=int, default=60_000)
    parser.add_argument("--inc-ms", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=48)
    parser.add_argument("--out", default="maia3_false_positive_results.json")
    args = parser.parse_args()

    elo_bands = [int(x) for x in args.elo_bands.split(",")]

    print("loading Maia-3 model...", flush=True)
    model = Maia3(Path(args.maia3_model_path))

    all_results = {}
    for accelerated in (False, True):
        arm = "accelerated" if accelerated else "standard"
        all_results[arm] = []
        for band in elo_bands:
            print(f"=== {arm} arm, maia elo {band} ===", flush=True)
            res = run_arm(args.unchessed_path, args.nnue_path, model, accelerated,
                          band, args.games_per_arm_per_band, seed=1000 + band,
                          clock_ms=args.clock_ms, inc_ms=args.inc_ms,
                          max_plies=args.max_plies)
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
        print(f"{arm}: games={total} full_confirmed={len(confirmed)} "
              f"legacy_clock_games={legacy_clock_games} "
              f"resilient_games={resilient_games} fusion_games={fusion_games}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
