import chess
import chess.pgn
import chess.engine
import glob
import re
import os

SF_PATH = "/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2"
OUTDIR = "/home/amogusontheterminal/unchessed-ai/results/elo_ladder"
CSV_PATH = "/home/amogusontheterminal/unchessed-ai/results/elo_ladder/results.csv"
LIMIT = chess.engine.Limit(depth=14)

done_levels = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        for line in f:
            parts = line.strip().split(",")
            if parts and parts[0].isdigit():
                done_levels.add(int(parts[0]))

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write("elo,avg_cp_loss,blunders,n_moves,mate_misses,result\n")


def start_engine():
    eng = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    eng.configure({"Threads": 1, "Hash": 64})
    return eng


def cp(score):
    if score.is_mate():
        mv = score.mate()
        return 100000 if mv is None else (10000 - abs(mv) * 10) * (1 if mv > 0 else -1)
    return score.score()


engine = start_engine()

pgn_files = sorted(
    glob.glob(f"{OUTDIR}/elo_*.pgn"),
    key=lambda p: int(re.search(r"elo_(\d+)\.pgn", p).group(1)),
)

for path in pgn_files:
    m = re.search(r"elo_(\d+)\.pgn", path)
    elo = int(m.group(1))
    if elo in done_levels:
        continue

    with open(path, encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        print(f"ELO {elo}: no game found, skipping", flush=True)
        continue

    white = game.headers.get("White", "")
    adapter_is_white = white.startswith("Adapter")
    board = game.board()
    cp_losses = []
    blunders = 0
    mate_misses = 0

    for move in game.mainline_moves():
        is_adapter_move = (board.turn == chess.WHITE) == adapter_is_white
        if is_adapter_move:
            for attempt in range(3):
                try:
                    info_before = engine.analyse(board, LIMIT)
                    best_score = info_before["score"].pov(board.turn)
                    best_move = info_before.get("pv", [None])[0]
                    had_mate = best_score.is_mate() and best_score.mate() > 0
                    played_move = move
                    board.push(move)
                    info_after = engine.analyse(board, LIMIT)
                    after_score = info_after["score"].pov(not board.turn)
                    board.pop()
                    break
                except chess.engine.EngineTerminatedError:
                    print(f"  engine died mid-analysis at ELO {elo}, restarting (attempt {attempt+1})", flush=True)
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = start_engine()
            else:
                raise RuntimeError(f"engine kept dying at ELO {elo}")

            best_cp = cp(best_score)
            played_cp = cp(after_score)
            loss = max(0, best_cp - played_cp)
            cp_losses.append(loss)
            if loss >= 300:
                blunders += 1
            if had_mate and played_move != best_move:
                mate_misses += 1
            board.push(move)
        else:
            board.push(move)

    avg_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0
    result = game.headers.get("Result", "?")
    n = len(cp_losses)
    print(
        f"ELO {elo}: avg_cp_loss={avg_loss:.0f} blunders={blunders}/{n} "
        f"mate_misses={mate_misses} result={result}",
        flush=True,
    )
    with open(CSV_PATH, "a") as f:
        f.write(f"{elo},{avg_loss:.1f},{blunders},{n},{mate_misses},{result}\n")

engine.quit()
print("ANALYSIS COMPLETE", flush=True)
