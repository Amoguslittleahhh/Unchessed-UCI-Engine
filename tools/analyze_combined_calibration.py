#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import chess,numpy as np
from fit_homemade_features import feature_vector

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('labels'); ap.add_argument('--out',required=True); ap.add_argument('--max-abs-cp',type=int,default=0); a=ap.parse_args()
 rows=[json.loads(x) for x in Path(a.labels).read_text().splitlines() if x.strip()]
 if a.max_abs_cp:
  rows=[r for r in rows if abs(int(r['stockfish_stm_cp'])) <= a.max_abs_cp]
 y=np.array([r['stockfish_stm_cp'] if chess.Board(r['fen']).turn==chess.WHITE else -r['stockfish_stm_cp'] for r in rows],float)
 x=[]
 for r in rows:
  b=chess.Board(r['fen']); side=1 if b.turn==chess.WHITE else -1; f=feature_vector(r['fen']); *s,p=f
  z=[1.0,r['loaded_eval_white_cp']/1000.0,p/30.0]
  z += [v/10.0 for v in s]
  z += [s[i]*(p/30.0)/10.0 for i in (1,2,6,8,9)]
  z += [side*p/30.0, s[1]*s[2]/100.0, s[6]*s[8]/10.0]
  x.append(z)
 x=np.array(x); test=np.arange(len(rows))%5==0
 best=None
 for lam in np.logspace(-3,5,65):
  reg=np.eye(x.shape[1])*lam; reg[0,0]=0
  w=np.linalg.solve(x[~test].T@x[~test]+reg,x[~test].T@y[~test]); pred=x@w
  val=float(np.mean(abs(pred[test]-y[test])))
  if best is None or val<best[0]: best=(val,lam,w,float(np.mean(abs(pred[~test]-y[~test]))))
 val,lam,w,tr=best; base=np.array([r['loaded_eval_white_cp'] for r in rows]); out={'positions':len(rows),'test':int(test.sum()),'loaded_test_mae_cp':float(np.mean(abs(base[test]-y[test]))),'calibrated_test_mae_cp':val,'calibrated_train_mae_cp':tr,'calibrated_all_mae_cp':float(np.mean(abs(x@w-y))),'lambda':float(lam),'weights':[float(v) for v in w]}
 Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
