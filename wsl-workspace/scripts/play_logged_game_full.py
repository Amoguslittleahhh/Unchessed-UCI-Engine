import subprocess
import threading
import queue
import time
import sys
import chess

ADAPTER = "/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter"
RUBI = "/home/amogusontheterminal/unchessed-ai/data/rubichess_bin/RubiChess-20240817/linux/RubiChess-20240817_x86-64-avx2"
OUTDIR = "/home/amogusontheterminal/unchessed-ai/results/tenmin_game_logged_full"
import os
os.makedirs(OUTDIR, exist_ok=True)

START_MS = 10 * 60 * 1000  # 10 minutes
INC_MS = 0

class Engine:
    def __init__(self, name, cmd, logpath):
        self.name = name
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.q = queue.Queue()
        self.logfile = open(logpath, "w", encoding="utf-8")
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.q.put(line)
            self.logfile.write(f"<< {line}\n")
            self.logfile.flush()

    def send(self, cmd):
        self.logfile.write(f">> {cmd}\n")
        self.logfile.flush()
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, marker, timeout):
        out = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            out.append(line)
            if marker in line:
                break
        return out

    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self.logfile.close()


def setup(engine, options):
    engine.send("uci")
    engine.read_until("uciok", 10)
    for opt, val in options.items():
        engine.send(f"setoption name {opt} value {val}")
    engine.send("isready")
    engine.read_until("readyok", 10)
    engine.send("ucinewgame")
    engine.send("isready")
    engine.read_until("readyok", 10)


adapter = Engine("Adapter", [ADAPTER], f"{OUTDIR}/adapter_log.txt")
rubi = Engine("RubiChess", [RUBI], f"{OUTDIR}/rubichess_log.txt")

setup(adapter, {"Threads": 1, "Adaptive": "false", "OwnBook": "true", "Hash": 128})
setup(rubi, {"Threads": 1, "Hash": 128})

moves = []
san_log = []
wtime = START_MS
btime = START_MS
white_is_adapter = True

move_num_file = open(f"{OUTDIR}/moves.txt", "w")

ply = 0
MAX_PLIES = 300
result = None
board = chess.Board()

while ply < MAX_PLIES:
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        print(f"ply {ply}: game over, result={result} ({board.outcome(claim_draw=True)})")
        break
    is_white = (ply % 2 == 0)
    engine = adapter if (is_white == white_is_adapter) else rubi
    pos_cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
    engine.send(pos_cmd)
    t0 = time.time()
    engine.send(f"go wtime {wtime} btime {btime} winc {INC_MS} binc {INC_MS}")
    # generous real-wall-clock timeout: engine's own clock should self-limit,
    # this is just a safety net against a true hang
    out = engine.read_until("bestmove", timeout=(wtime if is_white else btime) / 1000.0 + 30)
    elapsed_ms = int((time.time() - t0) * 1000)
    if is_white:
        wtime = max(0, wtime - elapsed_ms) + INC_MS
    else:
        btime = max(0, btime - elapsed_ms) + INC_MS
    bm_line = next((l for l in out if l.startswith("bestmove")), None)
    if not bm_line:
        print(f"ply {ply}: NO BESTMOVE from {engine.name} (possible flag/hang)")
        result = "no_bestmove"
        break
    mv = bm_line.split()[1]
    if mv in ("0000", "(none)"):
        print(f"ply {ply}: {engine.name} returned null move -- game over")
        result = board.result(claim_draw=True)
        break
    try:
        move_obj = chess.Move.from_uci(mv)
        if move_obj not in board.legal_moves:
            print(f"ply {ply}: {engine.name} played ILLEGAL move {mv} in position {board.fen()}")
            result = f"illegal_move_by_{engine.name}"
            break
        board.push(move_obj)
    except Exception as e:
        print(f"ply {ply}: couldn't parse move {mv} from {engine.name}: {e}")
        result = f"bad_move_by_{engine.name}"
        break
    moves.append(mv)
    move_num_file.write(f"{ply} {'W' if is_white else 'B'} {engine.name} {mv} wtime={wtime} btime={btime} elapsed={elapsed_ms}ms\n")
    move_num_file.flush()
    print(f"ply {ply} ({'W' if is_white else 'B'}, {engine.name}): {mv}  [wtime={wtime}ms btime={btime}ms]", flush=True)
    if wtime <= 0:
        print("White flagged"); result = "white_flag"; break
    if btime <= 0:
        print("Black flagged"); result = "black_flag"; break
    ply += 1

move_num_file.write(f"RESULT: {result}\n")
move_num_file.close()

with open(f"{OUTDIR}/final_moves.txt", "w") as f:
    f.write(" ".join(moves) + "\n")
    f.write(f"result: {result}\n")

import chess.pgn
game = chess.pgn.Game()
game.headers["White"] = "UnchessedAdapter"
game.headers["Black"] = "RubiChessFull"
game.headers["Result"] = result if result in ("1-0", "0-1", "1/2-1/2") else "*"
game.headers["TimeControl"] = "600"
node = game
b2 = chess.Board()
for mv in moves:
    m = chess.Move.from_uci(mv)
    node = node.add_variation(m)
    b2.push(m)
with open(f"{OUTDIR}/game.pgn", "w") as f:
    print(game, file=f)

adapter.quit()
rubi.quit()
print("DONE, result:", result)
