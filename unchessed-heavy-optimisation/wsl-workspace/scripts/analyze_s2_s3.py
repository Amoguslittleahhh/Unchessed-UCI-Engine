import chess
import chess.pgn
import chess.engine
import statistics

SF_PATH = "/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2"
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


def analyze(path, label):
    engine = start_engine()
    with open(path, encoding="utf-8", errors="replace") as f:
        game_idx = 0
        all_losses = []
        all_blunders = 0
        all_moves = 0
        results = []
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_idx += 1
            white = game.headers.get("White", "")
            adapter_is_white = "Adapt" in white
            board = game.board()
            cp_losses = []
            blunders = 0
            for move in game.mainline_moves():
                if not board.is_legal(move):
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
                        raise RuntimeError("engine kept dying")
                    loss = max(0, cp(best_score) - cp(after_score))
                    cp_losses.append(loss)
                    if loss >= 300:
                        blunders += 1
                board.push(move)
            avg = sum(cp_losses) / len(cp_losses) if cp_losses else 0
            result = game.headers.get("Result", "?")
            results.append(result)
            all_losses.extend(cp_losses)
            all_blunders += blunders
            all_moves += len(cp_losses)
            print(f"{label} game {game_idx}: avg_cp_loss={avg:.0f} blunders={blunders}/{len(cp_losses)} result={result}", flush=True)
    engine.quit()
    if all_losses:
        print(f"\n{label} AGGREGATE: n_games={game_idx} mean_cp_loss={statistics.mean(all_losses):.1f} median={statistics.median(all_losses):.1f} blunder_rate={all_blunders/all_moves*100:.1f}% ({all_blunders}/{all_moves} moves)")
    return results


print("=== S2: Adaptive vs SF@2800 ===")
analyze("/home/amogusontheterminal/unchessed-ai/results/feature_matrix/s2_adaptive_vs_strong.pgn", "S2")
print("\n=== S3: Adaptive vs full-strength SF ===")
analyze("/home/amogusontheterminal/unchessed-ai/results/feature_matrix/s3_adaptive_vs_full.pgn", "S3")
