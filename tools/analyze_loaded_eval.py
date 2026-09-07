#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import chess
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('labels'); ap.add_argument('--out',required=True); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.labels).read_text().splitlines() if x.strip()]
    y=np.asarray([r['stockfish_stm_cp'] if chess.Board(r['fen']).turn==chess.WHITE else -r['stockfish_stm_cp'] for r in rows],float)
    pred=np.asarray([r['loaded_eval_white_cp'] for r in rows],float)
    test=np.arange(len(rows))%5==0
    d={'positions':len(rows),'test':int(test.sum()),'all_mae_cp':float(np.mean(abs(pred-y))),'test_mae_cp':float(np.mean(abs(pred[test]-y[test]))),'median_test_abs_cp':float(np.median(abs(pred[test]-y[test]))),'max_test_abs_cp':int(max(abs(pred[test]-y[test])))}
    Path(a.out).write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d,indent=2))
if __name__=='__main__': main()
