# Real-engine gauntlet

This directory contains 48 paired UCI games played by the full-strength
`unchessed-reviewer` against tagged releases of Ethereal, Berserk, and
Stockfish. The external binaries are not redistributed. `provenance.json`
records their source commits, build modes, licenses, binary hashes, harness,
host, openings, and adjudication settings.

The PGNs include per-move node and NPS metadata and are the auditable source for
`report.json` and `result.md`. Regenerate the aggregate with:

```bash
python3 tools/summarize_engine_gauntlet.py \
  --candidate Unchessed \
  --provenance benchmarks/real-engines/provenance.json \
  --pgn benchmarks/real-engines/games/*.pgn \
  --json benchmarks/real-engines/report.json \
  --markdown benchmarks/real-engines/result.md --check
```

This is a node-limited interoperability/strength smoke test. Eight opening
pairs are far too few for an Elo claim; the paired bootstrap intervals in the
report make that uncertainty explicit.
