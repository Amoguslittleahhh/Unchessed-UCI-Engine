#!/usr/bin/env python3
"""Evaluate a student JSONL against reference score fields.

Student rows must preserve position_id and contain student_cp; reference rows
must contain score_cp or score_mate. This intentionally reports failures rather
than inventing a parity score for missing terminal labels.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def score(row):
    if row.get("score_mate") is not None:
        m = int(row["score_mate"]); return (30000 if m > 0 else -30000) + max(-100, min(100, m))
    return row.get("score_cp")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reference",required=True); ap.add_argument("--student",required=True); ap.add_argument("--mae-limit",type=float,default=100.0); args=ap.parse_args()
    ref={json.loads(x)["position_id"]:json.loads(x) for x in Path(args.reference).read_text().splitlines() if x.strip()}
    errors=[]
    for raw in Path(args.student).read_text().splitlines():
        if not raw.strip(): continue
        row=json.loads(raw); r=ref.get(row["position_id"]); a=score(r) if r else None; b=row.get("student_cp")
        if a is not None and b is not None: errors.append(abs(float(a)-float(b)))
    if not errors: raise SystemExit("no aligned score rows")
    mae=statistics.mean(errors); print(json.dumps({"positions":len(errors),"mae_cp":mae,"median_abs_error_cp":statistics.median(errors),"max_abs_error_cp":max(errors),"mae_gate_pass":mae<=args.mae_limit},sort_keys=True))
    raise SystemExit(0 if mae<=args.mae_limit else 1)
if __name__ == "__main__": main()
