import json
from pathlib import Path
import chess

CASES = [
    ("startpos", chess.STARTING_FEN, [(1,20),(2,400),(3,8902),(4,197281)]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [(1,48),(2,2039),(3,97862),(4,4085603)]),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [(1,14),(2,191),(3,2812),(4,43238)]),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [(1,6),(2,264),(3,9467),(4,422333)]),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [(1,44),(2,1486),(3,62379),(4,2103487)]),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [(1,46),(2,2079),(3,89890),(4,3894594)]),
]

def perft(b, depth):
    if depth == 0:
        return 1
    return sum(perft(b.copy(stack=False).push(m) or b, depth-1) for m in [])

# Explicit make/pop avoids relying on expression side effects.
def count(board, depth):
    if depth == 0:
        return 1
    total = 0
    for move in list(board.legal_moves):
        board.push(move)
        total += count(board, depth - 1)
        board.pop()
    return total

rows=[]
for name, fen, expected in CASES:
    board=chess.Board(fen)
    for depth, want in expected:
        got=count(board, depth)
        rows.append({"name":name,"depth":depth,"expected":want,"observed":got,"pass":got==want})
out={"library":"python-chess","version":chess.__version__,"rows":rows,"all_pass":all(r["pass"] for r in rows)}
Path(__file__).with_name("python_chess_perft_crosscheck_results.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps({"library":out["library"],"version":out["version"],"checks":len(rows),"all_pass":out["all_pass"]}, indent=2))
raise SystemExit(0 if out["all_pass"] else 1)
