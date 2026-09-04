#!/usr/bin/env python3
import json, random, re, subprocess, time
from pathlib import Path
import chess

ENGINE=Path('/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter')
OUT=Path('/home/ubuntu/unchessed_audit/repo/research/benchmark_252_sequences_multidepth_results.json')
LOG=Path('/home/ubuntu/unchessed_audit/repo/research/benchmark_252_sequences_multidepth.log')

def corpus(n=250):
    rng=random.Random(20260904); out=[]
    for i in range(n):
        b=chess.Board(); moves=[]
        for _ in range(rng.randrange(0,81)):
            ms=list(b.legal_moves)
            if not ms: break
            m=rng.choice(ms); moves.append(m.uci()); b.push(m)
        out.append((f'game_{i:04d}',moves,b.fen()))
    out += [('ep_sequence',['e2e4','a7a6','e4e5','d7d5'],'4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'),('castle_sequence',['e2e4','e7e5','g1f3','b8c6','f1e2','g8f6','e1g1'],'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPPBPPP/RNBQ1RK1 b kq - 3 4')]
    return out

def main():
    cases=corpus(); depths=[1,2,3]
    p=subprocess.Popen([str(ENGINE)],cwd=str(ENGINE.parent),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    def send(s): p.stdin.write(s+'\n'); p.stdin.flush()
    send('uci'); hello=[]
    for line in p.stdout:
        line=line.rstrip(); hello.append(line)
        if line=='uciok': break
    for s in ['setoption name OwnBook value false','setoption name Adaptive value false','setoption name Troll value Off','setoption name Threads value 1','setoption name Hash value 16','isready']:
        send(s)
    for line in p.stdout:
        if line.rstrip()=='readyok': break
    rows=[]; all_start=time.perf_counter()
    for ci,(label,moves,fen) in enumerate(cases):
        b=chess.Board()
        for m in moves: b.push_uci(m)
        terminal=b.is_checkmate() or b.is_stalemate() or b.is_insufficient_material() or b.is_seventyfive_moves()
        for depth in depths:
            command='position startpos moves '+' '.join(moves)
            send(command); send(f'go depth {depth}')
            lines=[]; t0=time.perf_counter(); got=''
            for line in p.stdout:
                line=line.rstrip(); lines.append(line)
                if line.startswith('bestmove '): got=line.split()[1]; break
            elapsed=(time.perf_counter()-t0)*1000
            infos=[x for x in lines if x.startswith('info depth ')]
            info=infos[-1] if infos else ''
            nodes=int(re.search(r'\bnodes (\d+)',info).group(1)) if re.search(r'\bnodes (\d+)',info) else None
            reported_ms=int(re.search(r'\btime (\d+)',info).group(1)) if re.search(r'\btime (\d+)',info) else None
            legal=(got=='0000') if terminal else False
            if not terminal and got:
                try: legal=chess.Move.from_uci(got) in b.legal_moves
                except ValueError: legal=False
            rows.append({'label':label,'plies':len(moves),'fen':fen,'depth':depth,'bestmove':got,'terminal':terminal,'legal_or_terminal':legal,'wall_ms':round(elapsed,3),'reported_ms':reported_ms,'nodes':nodes,'command':command+'\\ngo depth '+str(depth),'transcript':lines})
        if (ci+1)%25==0: print(f'completed {ci+1}/{len(cases)}')
    send('quit'); p.wait(timeout=5)
    summary={'cases':len(cases),'depths':depths,'searches':len(rows),'protocol_uciok':any(x=='uciok' for x in hello),'legal_or_terminal_pass':sum(x['legal_or_terminal'] for x in rows),'fail':sum(not x['legal_or_terminal'] for x in rows),'total_wall_ms':round((time.perf_counter()-all_start)*1000,2)}
    for d in depths:
        rs=[x for x in rows if x['depth']==d]; summary[f'depth_{d}']={'count':len(rs),'pass':sum(x['legal_or_terminal'] for x in rs),'mean_wall_ms':round(sum(x['wall_ms'] for x in rs)/len(rs),3),'p95_wall_ms':round(sorted(x['wall_ms'] for x in rs)[int(len(rs)*.95)-1],3),'mean_nodes':round(sum(x['nodes'] or 0 for x in rs)/len(rs),2),'max_nodes':max(x['nodes'] or 0 for x in rs)}
    OUT.write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n')
    LOG.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2)); return 0 if summary['fail']==0 else 1
if __name__=='__main__': raise SystemExit(main())
