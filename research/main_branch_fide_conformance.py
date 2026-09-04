#!/usr/bin/env python3
"""Fresh main-branch-only UCI/FIDE conformance harness.
The tested binary is built from /home/ubuntu/unchessed_audit/main_repo (remote main).
"""
import json, random, subprocess
from pathlib import Path
import chess

ROOT=Path(__file__).resolve().parents[1]
ENGINE=Path('/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter')
OUT=ROOT/'research/main_branch_fide_conformance_results.json'
FIXED=[
('startpos',chess.STARTING_FEN),('castle_both','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
('castle_blocked','r3k2r/8/8/8/8/8/8/R2QK2R w KQkq - 0 1'),('en_passant','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'),
('promotion_q','4k3/P7/8/8/8/8/8/4K3 w - - 0 1'),('promotion_capture','2r1k3/P7/8/8/8/8/8/4K3 w - - 0 1'),
('in_check','4k3/8/8/8/8/8/4r3/4K3 w - - 0 1'),('pin','4k3/8/8/8/8/8/4r3/3QK3 w - - 0 1'),
('stalemate','7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'),('dead_kk','8/8/8/8/8/8/4k3/4K3 w - - 0 1'),
('dead_kb','8/8/8/8/8/8/4k3/2B1K3 w - - 0 1'),('dead_kn','8/8/8/8/8/8/4k3/2N1K3 w - - 0 1'),
('mate_hmc_100','6k1/5ppp/8/8/8/8/8/R5K1 w - - 100 1'),('quiet_hmc_100','7k/8/8/8/8/8/8/K7 w - - 100 1')]
INVALID=[('missing_black_king','4k3/8/8/8/8/8/8/4K3 w - - 0 1'),('two_white_kings','4k3/8/8/8/8/8/4K3/4K3 w - - 0 1'),('bad_castle','4k3/8/8/8/8/8/8/4K3 w K - 0 1'),('occupied_ep','4k3/8/8/4p3/4P3/8/8/4K3 w - e5 0 1'),('back_rank_pawn','4k3/P7/8/8/8/8/8/4K3 w - - 0 1')]

def reachable(n=1000):
 r=random.Random(20260904); out=[]
 for i in range(n):
  b=chess.Board()
  for _ in range(r.randrange(61)):
   ms=list(b.legal_moves)
   if not ms: break
   b.push(r.choice(ms))
  out.append((f'reachable_{i:04d}',b.fen()))
 return out

def transcript(cmds, timeout=300):
 p=subprocess.run([str(ENGINE)],cwd=str(ENGINE.parent),input='\n'.join(cmds)+'\nquit\n',text=True,capture_output=True,timeout=timeout)
 return p.returncode,p.stdout+p.stderr

def parse_bestmoves(text): return [x.split()[1] for x in text.splitlines() if x.startswith('bestmove ') and len(x.split())>=2]
def run():
 assert ENGINE.exists()
 configs=['uci','isready','setoption name OwnBook value false','setoption name Adaptive value false','setoption name UnarchitecturedHint value false','setoption name Troll value Off','isready']
 cases=FIXED+reachable(); cmds=configs[:]
 for _,fen in cases: cmds += ['position fen '+fen,'go depth 1']
 rc,text=transcript(cmds); bms=parse_bestmoves(text)
 results=[]
 for i,(label,fen) in enumerate(cases):
  b=chess.Board(fen); bm=bms[i] if i<len(bms) else ''
  terminal=b.is_checkmate() or b.is_stalemate() or b.is_insufficient_material() or b.is_seventyfive_moves(); ok=(bm=='0000') if terminal else False
  if not terminal and bm:
   try: ok=chess.Move.from_uci(bm) in b.legal_moves
   except ValueError: ok=False
  results.append({'label':label,'fen':fen,'bestmove':bm,'terminal_reference':terminal,'legal_bestmove':ok})
 icmd=configs[:]
 for _,fen in INVALID: icmd += ['position fen '+fen,'go depth 1']
 irc,itext=transcript(icmd); ibms=parse_bestmoves(itext)
 invalid=[]
 for i,(label,fen) in enumerate(INVALID):
  invalid.append({'label':label,'fen':fen,'bestmove_after_rejection':ibms[i] if i<len(ibms) else '', 'parser_error': 'could not parse' in itext.lower()})
 protocol={'uciok':'uciok' in text,'readyok':text.count('readyok')>=2,'returncode':rc}
 summary={'engine':str(ENGINE),'fixed_cases':len(FIXED),'reachable_cases':1000,'protocol':protocol,'valid_cases':len(results),'nonterminal_cases':sum(not x['terminal_reference'] for x in results),'nonterminal_legal_pass':sum(x['legal_bestmove'] for x in results if not x['terminal_reference']),'nonterminal_legal_fail':sum(not x['legal_bestmove'] for x in results if not x['terminal_reference']),'terminal_cases':sum(x['terminal_reference'] for x in results),'terminal_0000':sum(x['terminal_reference'] and x['bestmove']=='0000' for x in results),'invalid_cases':len(invalid),'invalid_parser_error':sum(x['parser_error'] for x in invalid),'invalid_bestmove_emitted':sum(bool(x['bestmove_after_rejection']) for x in invalid),'engine_returncode_invalid_run':irc}
 OUT.write_text(json.dumps({'summary':summary,'fixed_and_reachable':results,'invalid':invalid,'uci_transcript':text,'invalid_transcript':itext},indent=2)+'\n')
 print(json.dumps(summary,indent=2)); return 0 if protocol['uciok'] and protocol['readyok'] and summary['nonterminal_legal_fail']==0 else 1
if __name__=='__main__': raise SystemExit(run())
