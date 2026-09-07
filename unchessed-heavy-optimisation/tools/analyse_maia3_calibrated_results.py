#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
for arm, rows in data.items():
    confirmed = [r for r in rows if r.get('first_full_ply') is not None]
    plies = [r['first_full_ply'] for r in confirmed]
    skips = sum(r.get('obs_low_time_skips', 0) for r in rows)
    reasons = {}
    for row in rows:
        for reason, count in row.get('suspect_reason_counts', {}).items():
            reasons[reason] = reasons.get(reason, 0) + count
    print(json.dumps({
        'arm': arm,
        'games': len(rows),
        'full_confirmed': len(confirmed),
        'confirmation_rate': len(confirmed) / len(rows) if rows else None,
        'mean_first_full_ply': statistics.mean(plies) if plies else None,
        'median_first_full_ply': statistics.median(plies) if plies else None,
        'range_first_full_ply': [min(plies), max(plies)] if plies else None,
        'total_low_time_skips': skips,
        'reason_counts': reasons,
        'all_confirmations_resilient': bool(confirmed) and all(set(r.get('suspect_reason_counts', {})) == {'legacy_accelerated_resilient'} for r in confirmed) if arm == 'accelerated' else None,
    }, indent=2))
