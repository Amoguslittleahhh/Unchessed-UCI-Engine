#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import chess, numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('labels'); ap.add_argument('--out',required=True); ap.add_argument('--max-abs-cp',type=int,default=2000); a=ap.parse_args()
 rows=[json.loads(x) for x in Path(a.labels).read_text().splitlines() if x.strip()]
 rows=[r for r in rows if abs(int(r['stockfish_stm_cp']))<=a.max_abs_cp]
 y=np.asarray([r['stockfish_stm_cp'] if chess.Board(r['fen']).turn==chess.WHITE else -r['stockfish_stm_cp'] for r in rows],float)
 p=np.asarray([r['loaded_eval_white_cp'] for r in rows],float); test=np.arange(len(rows))%5==0
 d={'positions':len(rows),'test':int(test.sum()),'max_abs_target_cp':a.max_abs_cp,'all_mae_cp':float(np.mean(abs(p-y))),'test_mae_cp':float(np.mean(abs(p[test]-y[test]))),'median_test_abs_cp':float(np.median(abs(p[test]-y[test]))),'max_test_abs_cp':int(max(abs(p[test]-y[test])))}
 Path(a.out).write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d,indent=2))
if __name__=='__main__': main()
