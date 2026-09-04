import subprocess, time, threading, queue, sys

ADAPTER = "/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter"
SF = "/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2"
SF_ELO = sys.argv[1] if len(sys.argv) > 1 else None  # None = full strength

def spawn(cmd):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.rstrip("\n")) for l in p.stdout], daemon=True).start()
    return p, q

def send(p, cmd):
    p.stdin.write(cmd + "\n"); p.stdin.flush()

def read_until(q, marker, timeout=15.0):
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

adapter, aq = spawn([ADAPTER])
sf, sq = spawn([SF])

send(adapter, "uci"); read_until(aq, "uciok")
send(adapter, "isready"); read_until(aq, "readyok")
send(adapter, "setoption name Adaptive value true")
send(adapter, "setoption name OwnBook value true")
send(adapter, "ucinewgame")
send(adapter, "isready"); read_until(aq, "readyok")

send(sf, "uci"); read_until(sq, "uciok")
if SF_ELO:
    send(sf, "setoption name UCI_LimitStrength value true")
    send(sf, f"setoption name UCI_Elo value {SF_ELO}")
send(sf, "isready"); read_until(sq, "readyok")
send(sf, "ucinewgame")

moves = []
adapter_white = True  # Adapter plays White
for ply in range(50):
    is_adapter_turn = (ply % 2 == 0) == adapter_white
    pos_cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
    if is_adapter_turn:
        send(adapter, pos_cmd)
        send(adapter, "go movetime 300")
        out = read_until(aq, "bestmove", timeout=10.0)
        for l in out:
            if "mode=" in l or "-> " in l or "engine suspect" in l.lower() or "estimate" in l:
                print(f"[ply {ply}] ADAPTER: {l}", flush=True)
        bm = next((l for l in out if l.startswith("bestmove")), None)
    else:
        send(sf, pos_cmd)
        send(sf, "go movetime 300")
        out = read_until(sq, "bestmove", timeout=10.0)
        bm = next((l for l in out if l.startswith("bestmove")), None)
    if not bm:
        print(f"ply {ply}: no bestmove, stopping"); break
    mv = bm.split()[1]
    if mv == "0000":
        print(f"ply {ply}: game over"); break
    moves.append(mv)

for p in (adapter, sf):
    send(p, "quit")
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
