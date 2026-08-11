import subprocess, time

ENGINE = "/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix2/release/unchessed-adapter"
proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def send(cmd):
    proc.stdin.write(cmd + "\n"); proc.stdin.flush()

def read_until(marker, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        if marker in line:
            return line
    return None

send("uci"); read_until("uciok")
send("isready"); read_until("readyok")
send("setoption name Adaptive value false")
send("setoption name UCI_LimitStrength value true")
send("setoption name UCI_Elo value 500")
send("setoption name OwnBook value false")
send("ucinewgame")
send("isready"); read_until("readyok")

# a real midgame-ish position with many legal moves, out of any book
FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
send(f"position fen {FEN}")

t0 = time.time()
send("go wtime 5000 winc 50 btime 5000 binc 50")
line = read_until("bestmove", timeout=20.0)
elapsed = time.time() - t0
print(f"Move took {elapsed:.2f}s -> {line}")

send("quit")
try:
    proc.wait(timeout=5)
except Exception:
    proc.kill()
