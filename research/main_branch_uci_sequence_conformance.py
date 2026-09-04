#!/usr/bin/env python3
import json, random, subprocess
from pathlib import Path
import chess

ENGINE=Path('/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter')
OUT=Path('/home/ubuntu/unchessed_audit/repo/research/main_branch_uci_sequence_results.json')

def make_sequences(n=250):
    rng=random.Random(20260904); out=[]
    for i in range(n):
        b=chess.Board(); moves=[]
        for _ in range(rng.randrange(0,81)):
            legal=list(b.legal_moves)
            if not legal: break
            m=rng.choice(legal); moves.append(m.uci()); b.push(m)
        out.append((f'game_{i:04d}',moves,b.fen()))
    out += [
      ('ep_sequence',['e2e4','a7a6','e4e5','d7d5'],'4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'),
      ('castle_sequence',['e2e4','e7e5','g1f3','b8c6','f1e2','g8f6','e1g1'],'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPPBPPP/RNBQ1RK1 b kq - 3 4'),
    ]
    return out

def run(cmds):
    p=subprocess.run([str(ENGINE)],cwd=str(ENGINE.parent),input='\n'.join(cmds+['quit'])+'\n',text=True,capture_output=True,timeout=240)
    return p.returncode,p.stdout+p.stderr

def main():
    seq=make_sequences(); cmds=['uci','setoption name OwnBook value false','setoption name Adaptive value false','setoption name Troll value Off','isready']
    for _,moves,_ in seq:
        cmds += [('position startpos moves '+' '.join(moves)).rstrip(),'go depth 1']
    rc,text=run(cmds); bms=[x.split()[1] for x in text.splitlines() if x.startswith('bestmove ')]
    rows=[]
    for i,(label,moves,fen) in enumerate(seq):
        b=chess.Board();
        for m in moves: b.push_uci(m)
        bm=bms[i] if i<len(bms) else ''
        terminal=b.is_checkmate() or b.is_stalemate() or b.is_insufficient_material() or b.is_seventyfive_moves()
        ok=(bm=='0000') if terminal else False
        if not terminal and bm:
            try: ok=chess.Move.from_uci(bm) in b.legal_moves
            except ValueError: ok=False
        rows.append({'label':label,'plies':len(moves),'fen':fen,'bestmove':bm,'terminal':terminal,'legal':ok})
    invalid_fen='4k3/8/8/8/8/8/8/4K3 w K - 0 1'
    irc,itext=run(['uci','setoption name OwnBook value false','setoption name Adaptive value false','isready','position startpos','go depth 1','position fen '+invalid_fen,'go depth 1'])
    ibm=[x.split()[1] for x in itext.splitlines() if x.startswith('bestmove ')]
    summary={'sequence_cases':len(seq),'legal_or_terminal_pass':sum(x['legal'] for x in rows),'fail':sum(not x['legal'] for x in rows),'uciok': 'uciok' in text,'readyok':'readyok' in text,'invalid_parser_error':'could not parse' in itext.lower(),'invalid_bestmoves':len(ibm),'engine_returncode':rc,'invalid_returncode':irc}
    OUT.write_text(json.dumps({'summary':summary,'rows':rows,'invalid_transcript':itext},indent=2)+'\n'); print(json.dumps(summary,indent=2)); raise SystemExit(0 if summary['fail']==0 else 1)
if __name__=='__main__': main()
