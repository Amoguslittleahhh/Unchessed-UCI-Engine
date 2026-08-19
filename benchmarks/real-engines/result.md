# Real-engine gauntlet

These are actual paired UCI games, not synthetic positions or self-play.

## Conditions

- 20,000 nodes per move for both engines
- 1 thread and 64 MiB hash per engine
- 8 opening pairs with colors reversed
- opening source: `tools/classics.pgn` (8 plies)
- no tablebases; one physical host core; concurrency 1
- results include adjudication under the recorded Fastchess settings

## Results from the Unchessed perspective

| Opponent | W-L-D | Score | Pair-bootstrap 95% | Median Unchessed NPS | Median opponent NPS |
|---|---:|---:|---:|---:|---:|
| Ethereal14-HCE | 6-9-1 | 40.62% | [21.88%, 62.50%] | 911,391 | 1,762,000 |
| Berserk4.7 | 5-10-1 | 34.38% | [6.25%, 65.62%] | 886,937 | 1,926,833 |
| Stockfish12-Classical | 4-9-3 | 34.38% | [15.62%, 56.25%] | 888,250 | 1,540,000 |

Combined descriptive score: **17.5/48 (36.46%)**.

## Interpretation

The sample is deliberately small and its intervals are wide. It proves UCI interoperability and provides a real-engine smoke baseline; it does not establish an Elo rating or a statistically significant ordering. Stockfish 12 was compiled in classical-evaluation mode because current Stockfish 18 network assets were unavailable on this runner. Ethereal 14 and Berserk 4.7 are also real tagged releases using their HCE paths.

Full PGNs and exact source/build provenance are committed beside this report.
