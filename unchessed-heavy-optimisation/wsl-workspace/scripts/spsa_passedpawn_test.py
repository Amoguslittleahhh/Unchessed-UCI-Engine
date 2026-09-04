import subprocess
import random
import re
import time
import json
import os

CUTECHESS = "/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli"
BOOK = "/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn"
ENGINE = "/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-tunablepp/release/unchessed-adapter"
OUTDIR = "/home/amogusontheterminal/unchessed-ai/results/spsa_pp"
os.makedirs(OUTDIR, exist_ok=True)
CKPT = f"{OUTDIR}/checkpoint_test.json"

GAMES_PER_ITER = 4
TC = "3+0.03"
MAX_ITERS = 2

# Spall gain schedule
ALPHA = 0.602
GAMMA = 0.101
A = MAX_ITERS * 0.1
C0 = 25.0   # initial perturbation size, in raw 0-200 units
A0 = 18.0   # initial step size

LO, HI = 0.0, 200.0

def clip(v):
    return max(LO, min(HI, v))

def load_checkpoint():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            d = json.load(f)
        return d["theta"], d["k"]
    return [50.0, 50.0], 0

def save_checkpoint(theta, k):
    with open(CKPT, "w") as f:
        json.dump({"theta": theta, "k": k}, f)

def play_match(mg_plus, eg_plus, mg_minus, eg_minus, games):
    pgn_out = f"{OUTDIR}/iter.pgn"
    log_out = f"{OUTDIR}/iter.log"
    cmd = [
        CUTECHESS,
        "-engine", f"cmd={ENGINE}", "name=Plus",
        "option.Threads=1", "option.Adaptive=false", "option.OwnBook=false", "option.Hash=32",
        f"option.PassedPawnMgPct={int(round(mg_plus))}", f"option.PassedPawnEgPct={int(round(eg_plus))}",
        "-engine", f"cmd={ENGINE}", "name=Minus",
        "option.Threads=1", "option.Adaptive=false", "option.OwnBook=false", "option.Hash=32",
        f"option.PassedPawnMgPct={int(round(mg_minus))}", f"option.PassedPawnEgPct={int(round(eg_minus))}",
        "-each", "proto=uci", f"tc={TC}",
        "-openings", f"file={BOOK}", "format=pgn", "order=random", "plies=12",
        "-repeat", "-rounds", str(games // 2), "-games", "2", "-concurrency", "13",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=900",
        "-pgnout", pgn_out,
    ]
    with open(log_out, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
    with open(log_out) as lf:
        content = lf.read()
    m = re.findall(r"Score of Plus vs Minus: (\d+) - (\d+) - (\d+)", content)
    if not m:
        return 0.5, 0
    w, l, d = map(int, m[-1])
    n = w + l + d
    if n == 0:
        return 0.5, 0
    score = (w + 0.5 * d) / n
    return score, n


theta, k = load_checkpoint()
print(f"Starting/resuming at k={k}, theta={theta}", flush=True)

while k < MAX_ITERS:
    c_k = C0 / ((k + 1) ** GAMMA)
    a_k = A0 / ((k + 1 + A) ** ALPHA)
    delta = [random.choice([-1, 1]), random.choice([-1, 1])]

    theta_plus = [clip(theta[i] + c_k * delta[i]) for i in range(2)]
    theta_minus = [clip(theta[i] - c_k * delta[i]) for i in range(2)]

    t0 = time.time()
    score, n = play_match(theta_plus[0], theta_plus[1], theta_minus[0], theta_minus[1], GAMES_PER_ITER)
    elapsed = time.time() - t0

    y = (score - 0.5) * 2.0  # in [-1, 1], positive = plus side better
    ghat = [y / (2 * c_k * delta[i]) for i in range(2)]
    theta = [clip(theta[i] + a_k * ghat[i]) for i in range(2)]

    k += 1
    save_checkpoint(theta, k)
    print(
        f"iter {k}/{MAX_ITERS} c_k={c_k:.2f} a_k={a_k:.2f} delta={delta} "
        f"plus={[round(x,1) for x in theta_plus]} minus={[round(x,1) for x in theta_minus]} "
        f"score={score:.3f} (n={n}) -> theta={[round(x,1) for x in theta]} [{elapsed:.0f}s]",
        flush=True,
    )

print("SPSA DONE, final theta:", theta, flush=True)
with open(f"{OUTDIR}/final_theta.json", "w") as f:
    json.dump({"theta": theta}, f)
