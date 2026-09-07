#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comparison')
    args = parser.parse_args()
    rows=[]
    pat=re.compile(r"\| `([^`]+)` \| (-?\d+) \| (-?\d+) \| ([+-]?\d+) \|")
    for line in Path(args.comparison).read_text().splitlines():
        m=pat.match(line)
        if m:
            _, sf, home, _=m.groups(); rows.append((int(sf),int(home)))
    y=np.array([r[0] for r in rows],dtype=float); x=np.array([r[1] for r in rows],dtype=float)
    mask=np.arange(len(rows))%5!=0
    A=np.column_stack([x[mask],np.ones(mask.sum())])
    w=np.linalg.lstsq(A,y[mask],rcond=None)[0]
    pred=A@w
    test=np.arange(len(rows))%5==0
    test_pred=w[0]*x[test]+w[1]
    print(f'positions={len(rows)}')
    print(f'baseline_mae={np.mean(np.abs(x[test]-y[test])):.3f}')
    print(f'fit_train_mae={np.mean(np.abs(pred-y[mask])):.3f}')
    print(f'fit_test_mae={np.mean(np.abs(test_pred-y[test])):.3f}')
    print(f'slope={w[0]:.9f} intercept={w[1]:.9f}')
if __name__=='__main__': main()
