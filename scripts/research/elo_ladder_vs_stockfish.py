"""Absolute Elo estimate for the adapter, anchored against real Stockfish.

Everything else in this project's SPRT history measures relative strength
(engine A vs engine B). This script anchors an absolute estimate instead,
by playing the adapter (Adaptive=false, pure engine strength -- no persona
behavior) against Stockfish with UCI_LimitStrength/UCI_Elo set to a series
of calibrated levels, and finding where the score crosses roughly 50%.

Requires a real Stockfish binary (UCI_Elo range 1320-3190) -- not bundled
here; point --stockfish at whatever copy is available locally (this was
run against the Stockfish 18 binary bundled with the En Croissant chess
GUI, org.encroissant.app/engines/stockfish/).

No cutechess-cli needed: drives both engines directly over UCI via a
small reader-thread + queue harness, matching the pattern used elsewhere
in this project's ad hoc match runners.

Known result (2026-09-05, movetime=1000ms, 8 games/level, this reviewer's
Windows machine): 8/8 at UCI_Elo 2200, 8/8 at UCI_Elo 2500, 5.5/8 at UCI_Elo
2800 (first competitive level) -> implied absolute Elo ~2940 at this fast
time control. n=8 at the one informative level is a small sample (wide
CI, easily +/-150-200 Elo) and the result is specific to this movetime,
not a slower reference control the way CCRL/CEGT lists use. See
scripts/research/elo_ladder_vs_stockfish_result_20260905.md for the full
writeup.
"""
import subprocess, threading, queue, time, sys, argparse

MOVETIME_MS = 1000
MAX_PLIES = 200
GAMES_PER_LEVEL = 8
ELO_LEVELS = [2200, 2500, 2800]
OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["c2c4", "e7e5"],
    ["g1f3", "d7d5"], ["e2e4", "c7c5"], ["d2d4", "g8f6"], ["e2e4", "e7e6"],
]


class Engine:
    def __init__(self, path, name, setup_cmds=None):
        self.name = name
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.q = queue.Queue()
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

    def _wait_for(self, token, timeout, collect=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.q.get(timeout=max(0, deadline - time.time()))
            except queue.Empty:
                break
            if line is None:
                break
            if collect is not None:
                collect.append(line)
            if line.strip() == token or line.startswith(token):
                return line
        return None

    def newgame(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", 10)

    def bestmove(self, moves):
        pos_cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos_cmd)
        self._send(f"go movetime {MOVETIME_MS}")
        line = self._wait_for("bestmove", 20)
        if line is None:
            return None
        parts = line.split()
        return parts[1] if len(parts) > 1 else None

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def play_game(white, black, opening):
    import chess
    white.newgame()
    black.newgame()
    board = chess.Board()
    moves = list(opening)
    for mv in moves:
        board.push_uci(mv)
    ply = len(moves)
    while not board.is_game_over(claim_draw=True) and ply < MAX_PLIES:
        engine = white if board.turn == chess.WHITE else black
        mv = engine.bestmove(moves)
        if mv is None or mv == "0000":
            return board.result(claim_draw=True) or "*"
        try:
            board.push_uci(mv)
        except Exception:
            # illegal move -> adjudicate as a loss for the mover
            return "0-1" if board.turn == chess.WHITE else "1-0"
        moves.append(mv)
        ply += 1
    if ply >= MAX_PLIES:
        return "1/2-1/2"
    return board.result(claim_draw=True)


def elo_from_score(score_fraction, opp_elo):
    import math
    s = min(max(score_fraction, 0.001), 0.999)
    diff = -400 * math.log10(1 / s - 1)
    return opp_elo + diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unchessed_path")
    parser.add_argument("stockfish_path")
    parser.add_argument("--levels", type=int, nargs="+", default=ELO_LEVELS)
    parser.add_argument("--games-per-level", type=int, default=GAMES_PER_LEVEL)
    args = parser.parse_args()

    results = []
    for elo in args.levels:
        eng_us = Engine(args.unchessed_path, "unchessed", [
            "setoption name Adaptive value false",
            "setoption name OwnBook value false",
            "setoption name Threads value 1",
            "setoption name Hash value 64",
        ])
        eng_sf = Engine(args.stockfish_path, "stockfish", [
            "setoption name UCI_LimitStrength value true",
            f"setoption name UCI_Elo value {elo}",
            "setoption name Threads value 1",
            "setoption name Hash value 64",
        ])
        score = 0.0
        n = 0
        for i in range(args.games_per_level):
            opening = OPENINGS[i % len(OPENINGS)]
            if i % 2 == 0:
                white, black = eng_us, eng_sf
                us_is_white = True
            else:
                white, black = eng_sf, eng_us
                us_is_white = False
            result = play_game(white, black, opening)
            n += 1
            if result == "1/2-1/2":
                score += 0.5
            elif (result == "1-0" and us_is_white) or (result == "0-1" and not us_is_white):
                score += 1.0
            print(f"  elo={elo} game{i+1}: us={'White' if us_is_white else 'Black'} -> {result}", flush=True)
        eng_us.quit()
        eng_sf.quit()
        frac = score / n
        est = elo_from_score(frac, elo)
        print(f"ELO {elo}: score {score}/{n} = {frac:.3f}  -> implied unchessed Elo ~{est:.0f}", flush=True)
        results.append((elo, score, n, est))

    print("\n=== SUMMARY ===")
    for elo, score, n, est in results:
        print(f"vs SF UCI_Elo={elo}: {score}/{n}  implied ~{est:.0f}")


if __name__ == "__main__":
    main()
