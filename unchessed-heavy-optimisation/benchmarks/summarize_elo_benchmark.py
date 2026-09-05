#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path

src = Path(sys.argv[1])
data = json.loads(src.read_text())
rows = []
for elo in sorted({g['sf_elo'] for g in data['games']}):
    games = [g for g in data['games'] if g['sf_elo'] == elo]
    score = sum(g['unchessed_score'] for g in games) / len(games)
    wins = sum(g['unchessed_score'] == 1.0 for g in games)
    draws = sum(g['unchessed_score'] == 0.5 for g in games)
    losses = sum(g['unchessed_score'] == 0.0 for g in games)
    # Wilson interval for a Bernoulli win-equivalent score is only a rough
    # descriptive interval here; no SPRT claim is made for these tiny samples.
    n = len(games)
    z = 1.96
    denom = 1 + z*z/n
    center = (score + z*z/(2*n)) / denom
    half = z * math.sqrt(score*(1-score)/n + z*z/(4*n*n)) / denom
    rows.append({'sf_elo': elo, 'games': n, 'wins': wins, 'draws': draws, 'losses': losses, 'score': score, 'wilson95_low': max(0.0, center-half), 'wilson95_high': min(1.0, center+half), 'avg_plies': sum(g['plies'] for g in games)/n})
result = {'source': str(src), 'rows': rows, 'overall_score': sum(g['unchessed_score'] for g in data['games'])/len(data['games']), 'games': len(data['games'])}
Path(src.with_suffix('.summary.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
