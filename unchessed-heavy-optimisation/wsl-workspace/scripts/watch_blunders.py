import subprocess, time, threading, queue

ENGINE = "/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter"
proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
q = queue.Queue()
threading.Thread(target=lambda: [q.put(l.rstrip("\n")) for l in proc.stdout], daemon=True).start()

def send(cmd):
    proc.stdin.write(cmd + "\n"); proc.stdin.flush()

def read_until(marker, timeout=15.0):
    out = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = q.get(timeout=max(0.01, deadline - time.time()))
        except queue.Empty:
            break
        out.append(line)
        if marker in line:
            break
    return out

send("uci"); read_until("uciok")
send("isready"); read_until("readyok")
send("setoption name Adaptive value false")
send("setoption name UCI_LimitStrength value true")
send("setoption name UCI_Elo value 500")
send("setoption name OwnBook value false")
send("ucinewgame")
send("isready"); read_until("readyok")

# self-play: have this SAME weak-target engine play both sides so we can
# watch its own move-selection reasoning move by move without needing an
# opponent process at all
moves = []
for ply in range(40):
    pos_cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
    send(pos_cmd)
    send("go movetime 200")
    out = read_until("bestmove", timeout=10.0)
    bestmove_line = next((l for l in out if l.startswith("bestmove")), None)
    for l in out:
        print(f"ply {ply}: {l}", flush=True)
    if not bestmove_line:
        print(f"ply {ply}: NO BESTMOVE, stopping"); break
    mv = bestmove_line.split()[1]
    if mv == "0000":
        print(f"ply {ply}: game over"); break
    moves.append(mv)

send("quit")
try:
    proc.wait(timeout=5)
except Exception:
    proc.kill()
