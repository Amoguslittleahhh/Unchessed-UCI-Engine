#!/usr/bin/env python3
import json
import math
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
data = json.load(open(path))
groups = defaultdict(list)
for g in data["results"]:
    opponent = "Stockfish" if "Stockfish" in (g["white"], g["black"]) else "Maia-3"
    score = 1.0 if (g["result"] == "1-0" and g["white"] == "Unchessed Game Adapter") or (g["result"] == "0-1" and g["black"] == "Unchessed Game Adapter") else 0.0
    if g["result"] == "1/2-1/2": score = 0.5
    groups[opponent].append((score, g))

def ci95(scores):
    n = len(scores)
    p = sum(scores) / n
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / (1 + z * z / n)
    return p, max(0.0, center - half), min(1.0, center + half)

summary = {}
for opponent, items in groups.items():
    scores = [x[0] for x in items]
    p, lo, hi = ci95(scores)
    summary[opponent] = {
        "games": len(items),
        "score_points": sum(scores),
        "score_percent": round(100 * p, 2),
        "wilson_95_ci_percent": [round(100 * lo, 2), round(100 * hi, 2)],
        "wins": sum(x == 1.0 for x in scores),
        "draws": sum(x == 0.5 for x in scores),
        "losses": sum(x == 0.0 for x in scores),
        "mean_plies": round(statistics.mean(x[1]["plies"] for x in items), 2),
        "mean_elapsed_seconds": round(statistics.mean(x[1]["elapsed_s"] for x in items), 2),
        "telemetry_decisions": sum(x[1]["persona_decisions"] for x in items),
        "mode_switches": sum(x[1]["mode_switches"] for x in items),
    }
print(json.dumps({"summary": summary, "overall": {"games": len(data["results"]), "score_points": sum(x[0] for xs in groups.values() for x in xs)}}, indent=2))
