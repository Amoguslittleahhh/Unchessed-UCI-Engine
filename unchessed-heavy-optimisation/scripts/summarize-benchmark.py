#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path), delimiter="\t"))
for row in rows:
    for key in ("nodes", "time_ms", "nps", "max_rss_kb", "hash_mb"):
        row[key] = int(row[key])
groups = defaultdict(list)
for row in rows:
    groups[(row["build"], row["hash_mb"])].append(row)
print("build\thash_mb\tmean_nps\tmedian_nps\tmean_rss_kb\tmean_nodes")
for key in sorted(groups):
    vals = groups[key]
    print(*key, round(statistics.mean(r["nps"] for r in vals)), round(statistics.median(r["nps"] for r in vals)), round(statistics.mean(r["max_rss_kb"] for r in vals)), round(statistics.mean(r["nodes"] for r in vals)), sep="\t")
for h in sorted({r["hash_mb"] for r in rows}):
    p = statistics.mean(r["nps"] for r in groups[("portable", h)])
    v = statistics.mean(r["nps"] for r in groups[("v3", h)])
    print(f"delta\t{h}\t{v-p:+.0f} nps\t{(v/p-1)*100:+.2f}%")
