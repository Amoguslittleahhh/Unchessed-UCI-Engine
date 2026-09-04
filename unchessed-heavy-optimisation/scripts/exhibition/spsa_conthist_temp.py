# TEMPORARY single-run tuning tool for ContHistPct only. Not a permanent
# part of the project -- run once, read final_theta.json, then discard.
# Single-parameter SPSA is far more tractable than this project's earlier
# 2-7 parameter campaigns (which were flat/unreliable) since there's only
# one dimension to find a gradient in.
import subprocess
import random
import re
import time
import json
import os

CUTECHESS = "/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli"
BOOK = "/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn"
ENGINE = "/home/amogusontheterminal/unchessed-kingsafety-src/target/release/unchessed-adapter"
OUTDIR = "/home/amogusontheterminal/unchessed-ai/results/spsa_conthist_temp"
os.makedirs(OUTDIR, exist_ok=True)
CKPT = f"{OUTDIR}/checkpoint.json"

GAMES_PER_ITER = 12
TC = "3+0.03"
MAX_ITERS = 40

ALPHA = 0.602
GAMMA = 0.101
A = MAX_ITERS * 0.1
C0 = 25.0
A0 = 18.0

LO, HI = 0.0, 200.0


def clip(v):
    return max(LO, min(HI, v))


def load_checkpoint():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            d = json.load(f)
        return d["theta"], d["k"]
    return [50.0], 0


def save_checkpoint(theta, k):
    with open(CKPT, "w") as f:
        json.dump({"theta": theta, "k": k}, f)


def play_match(plus, minus, games):
    pgn_out = f"{OUTDIR}/iter.pgn"
    log_out = f"{OUTDIR}/iter.log"
    cmd = [
        CUTECHESS,
        "-engine", f"cmd={ENGINE}", "name=Plus",
        "option.Threads=1", "option.Adaptive=false", "option.OwnBook=false", "option.Hash=32",
        f"option.ContHistPct={int(round(plus))}",
        "-engine", f"cmd={ENGINE}", "name=Minus",
        "option.Threads=1", "option.Adaptive=false", "option.OwnBook=false", "option.Hash=32",
        f"option.ContHistPct={int(round(minus))}",
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
    return (w + 0.5 * d) / n, n


theta, k = load_checkpoint()
print(f"Starting/resuming ContHistPct SPSA at k={k}, theta={theta}", flush=True)

while k < MAX_ITERS:
    c_k = C0 / ((k + 1) ** GAMMA)
    a_k = A0 / ((k + 1 + A) ** ALPHA)
    delta = random.choice([-1, 1])

    theta_plus = clip(theta[0] + c_k * delta)
    theta_minus = clip(theta[0] - c_k * delta)

    t0 = time.time()
    score, n = play_match(theta_plus, theta_minus, GAMES_PER_ITER)
    elapsed = time.time() - t0

    y = (score - 0.5) * 2.0
    ghat = y / (2 * c_k * delta)
    theta[0] = clip(theta[0] + a_k * ghat)

    k += 1
    save_checkpoint(theta, k)
    print(
        f"iter {k}/{MAX_ITERS} c_k={c_k:.2f} a_k={a_k:.2f} delta={delta} "
        f"plus={theta_plus:.1f} minus={theta_minus:.1f} score={score:.3f} (n={n}) "
        f"-> theta={theta[0]:.1f} [{elapsed:.0f}s]",
        flush=True,
    )

print("SPSA DONE, final theta:", theta, flush=True)
with open(f"{OUTDIR}/final_theta.json", "w") as f:
    json.dump({"theta": theta}, f)
