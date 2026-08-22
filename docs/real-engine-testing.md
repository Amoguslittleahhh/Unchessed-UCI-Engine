# Real-engine testing

The full-strength reviewer has completed actual UCI games against three
independent, tagged open-source engines:

- Ethereal 14.00, source-default HCE build;
- Berserk 4.7.0, the HCE release before Berserk adopted NNUE;
- Stockfish 12 with `Use NNUE=false` and neutral contempt.

The committed result is `benchmarks/real-engines/result.md`, with all 48 PGNs
and exact build provenance beside it.

## Controlled conditions

Each match uses eight opening pairs from `tools/classics.pgn`, reverses colors,
and gives both engines exactly 20,000 nodes per move, one thread, 64 MiB hash,
and MultiPV 1. Fastchess runs one game at a time on the one-physical-core host.
No engine receives a wall-clock or core-count advantage. Books stop after eight
plies and tablebases are disabled.

The reviewer scored:

| Opponent | W-L-D | Score |
|---|---:|---:|
| Ethereal 14 HCE | 6-9-1 | 40.62% |
| Berserk 4.7 | 5-10-1 | 34.38% |
| Stockfish 12 classical | 4-9-3 | 34.38% |
| Combined | 15-28-5 | 36.46% |

This establishes real-engine interoperability and an initial baseline, not an
Elo estimate. There are only eight independent opening pairs per opponent,
most games use conservative Fastchess adjudication, and the paired confidence
intervals are correspondingly wide.

## Why Stockfish 18 is not in this local result

Stockfish 18 source compiled successfully, but its two official NNUE network
assets could not be retrieved from the network-asset host in this sandbox, so
it correctly refused to search. Substituting an unrelated network or reporting
a non-running binary would be invalid. Stockfish 12 was therefore built with
its supported classical evaluator and is labeled accordingly.

A current-engine follow-up should use official binaries/networks for Stockfish
18, current Berserk, current RubiChess, and at least one MCTS engine such as
Lc0. Those assets and GPU/high-core runs remain external; the match format and
provenance requirements are now fixed.

## Reproduction requirements

- use `unchessed-reviewer`, never the adaptive player binary;
- pin engine tags/commits and record executable SHA-256 values;
- use one thread, equal hash, MultiPV 1, and equal node budgets;
- pair every opening with colors reversed;
- retain complete PGNs;
- report W/L/D, points, pentanomial counts, and pair-level uncertainty;
- do not infer Elo from this smoke-sized sample.
