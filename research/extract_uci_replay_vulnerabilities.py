#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

ENGINE=Path('/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter')
OUT=Path('/home/ubuntu/unchessed_audit/repo/research/uci_replay_and_malformed_fen_exact.md')
BASE=['uci','setoption name OwnBook value false','setoption name Adaptive value false','setoption name Troll value Off','isready']
REPLAYS=[
 ('en_passant_replay',['e2e4','a7a6','e4e5','d7d5']),
 ('castle_replay',['e2e4','e7e5','g1f3','b8c6','f1e2','g8f6','e1g1']),
]
INVALID=[
 ('missing_black_king','4k3/8/8/8/8/8/8/4K3 w - - 0 1'),
 ('two_white_kings','4k3/8/8/8/8/8/4K3/4K3 w - - 0 1'),
 ('bad_castling_rights','4k3/8/8/8/8/8/8/4K3 w K - 0 1'),
 ('occupied_en_passant','4k3/8/8/4p3/4P3/8/8/4K3 w - e5 0 1'),
 ('back_rank_pawn','4k3/P7/8/8/8/8/8/4K3 w - - 0 1'),
]

def run(cmds):
 p=subprocess.Popen([str(ENGINE)],cwd=str(ENGINE.parent),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 p.stdin.write('\n'.join(cmds)+'\n'); p.stdin.flush(); lines=[]
 for line in p.stdout:
  lines.append(line.rstrip('\n'))
  if line.startswith('bestmove '): break
 p.stdin.write('quit\n'); p.stdin.flush(); p.wait(timeout=3)
 return lines

def section(name,cmds):
 lines=run(cmds)
 out=[f'## {name}','', '### Exact command stream','```text']+cmds+['```','', '### Exact engine output','```text']+lines+['```','']
 return '\n'.join(out)

def main():
 chunks=['# Exact UCI Replay and Malformed-FEN Evidence','',f'Engine: `{ENGINE}`','', 'Each section was run in an isolated process; the extractor waited for `bestmove` before sending `quit`.']
 for name,moves in REPLAYS:
  chunks.append(section(name,BASE+["position startpos moves "+' '.join(moves),'go depth 1']))
 chunks.append('## Malformed-FEN probes\n')
 for name,fen in INVALID:
  chunks.append(section(name,BASE+['position startpos','go depth 1','position fen '+fen,'go depth 1']))
 OUT.write_text('\n'.join(chunks)+'\n')
 print(OUT)
if __name__=='__main__': main()
