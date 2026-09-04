import chess
import chess.pgn
import chess.engine
import glob
import re
import os

SF_PATH = "/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2"
OUTDIR = "/home/amogusontheterminal/unchessed-ai/results/elo_points"
CSV_PATH = f"{OUTDIR}/per_move_results.csv"
LIMIT = chess.engine.Limit(depth=14)


def start_engine():
    eng = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    eng.configure({"Threads": 1, "Hash": 64})
    return eng


def cp(score):
    if score.is_mate():
        mv = score.mate()
        return 100000 if mv is None else (10000 - abs(mv) * 10) * (1 if mv > 0 else -1)
    return score.score()


done_games = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0].isdigit():
                done_games.add((int(parts[0]), parts[1]))
else:
    with open(CSV_PATH, "w") as f:
        f.write("elo,game_idx,avg_cp_loss,blunders,n_moves,result\n")

engine = start_engine()

pgn_files = sorted(
    glob.glob(f"{OUTDIR}/elo_*.pgn"),
    key=lambda p: int(re.search(r"elo_(\d+)\.pgn", p).group(1)),
)

for path in pgn_files:
    m = re.search(r"elo_(\d+)\.pgn", path)
    elo = int(m.group(1))
    with open(path, encoding="utf-8", errors="replace") as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            key = (elo, str(game_idx))
            game_idx += 1
            if key in done_games:
                continue

            white = game.headers.get("White", "")
            adapter_is_white = white.startswith("Adapter")
            board = game.board()
            cp_losses = []
            blunders = 0
            parse_ok = True

            for move in game.mainline_moves():
                if not board.is_legal(move):
                    print(f"  ELO {elo} game {game_idx}: illegal move {move} encountered, skipping rest of game", flush=True)
                    parse_ok = False
                    break
                is_adapter_move = (board.turn == chess.WHITE) == adapter_is_white
                if is_adapter_move:
                    for attempt in range(3):
                        try:
                            info_before = engine.analyse(board, LIMIT)
                            best_score = info_before["score"].pov(board.turn)
                            board.push(move)
                            info_after = engine.analyse(board, LIMIT)
                            after_score = info_after["score"].pov(not board.turn)
                            board.pop()
                            break
                        except chess.engine.EngineTerminatedError:
                            try:
                                engine.quit()
                            except Exception:
                                pass
                            engine = start_engine()
                    else:
                        raise RuntimeError(f"engine kept dying at ELO {elo} game {game_idx}")

                    loss = max(0, cp(best_score) - cp(after_score))
                    cp_losses.append(loss)
                    if loss >= 300:
                        blunders += 1
                    board.push(move)
                else:
                    board.push(move)

            avg_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0
            n = len(cp_losses)
            result = game.headers.get("Result", "?")
            print(
                f"ELO {elo} game {game_idx}: avg_cp_loss={avg_loss:.0f} blunders={blunders}/{n} result={result}",
                flush=True,
            )
            with open(CSV_PATH, "a") as f2:
                f2.write(f"{elo},{game_idx-1},{avg_loss:.1f},{blunders},{n},{result}\n")

engine.quit()
print("ANALYSIS COMPLETE", flush=True)
