#!/usr/bin/env python3
import json, re, subprocess
from pathlib import Path
import chess

ENGINE=Path('/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter')
EPD=Path('/home/ubuntu/unchessed_audit/main_repo/benchmarks/matetrack.epd')
OUT=Path('/home/ubuntu/unchessed_audit/repo/research/main_branch_matetrack_runtime_results.json')

def run_case(fen, expected_san):
    b=chess.Board(fen); expected=b.parse_san(expected_san).uci()
    inp='uci\nsetoption name Adaptive value false\nsetoption name OwnBook value false\nsetoption name Troll value Off\nisready\nposition fen '+fen+'\ngo depth 10\n'
    try:
        p=subprocess.Popen([str(ENGINE)],cwd=str(ENGINE.parent),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        p.stdin.write(inp); p.stdin.flush(); lines=[]; got=''
        for line in p.stdout:
            line=line.rstrip('\n'); lines.append(line)
            if line.startswith('bestmove '):
                got=line.split()[1]; break
        p.stdin.write('quit\n'); p.stdin.flush(); p.wait(timeout=3)
        return {'expected_uci':expected,'expected_san':expected_san,'observed_uci':got,'pass':got==expected,'returncode':p.returncode,'transcript':lines}
    except subprocess.TimeoutExpired:
        p.kill(); return {'expected_uci':expected,'expected_san':expected_san,'observed_uci':'','pass':False,'timeout':True,'transcript':lines}

def main():
    rows=[]
    for line in EPD.read_text().splitlines():
        if not line or line.startswith('#'): continue
        left=line.split(' bm ',1); fen=' '.join(left[0].split()+['0','1'])
        expected=left[1].split(';',1)[0].strip()
        ident=re.search(r'id "([^"]+)"',left[1]).group(1)
        r=run_case(fen,expected); r.update({'id':ident,'fen':fen}); rows.append(r)
    out={'engine':str(ENGINE),'cases':len(rows),'passed':sum(r['pass'] for r in rows),'failed':sum(not r['pass'] for r in rows),'rows':rows}
    OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ('engine','cases','passed','failed')},indent=2)); raise SystemExit(0 if out['failed']==0 else 1)
if __name__=='__main__': main()
