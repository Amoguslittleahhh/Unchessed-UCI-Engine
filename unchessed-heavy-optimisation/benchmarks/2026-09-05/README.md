# Real Elo-capped benchmark — 2026-09-05

This benchmark compares the release-built `unchessed-adapter` against the official Stockfish 19 universal Linux binary using Stockfish's native `UCI_LimitStrength=true` and `UCI_Elo` controls.

## Fixed configuration

| Setting | Value |
|---|---|
| Unchessed binary | `target/release/unchessed-adapter` |
| Opponent | Stockfish 19 universal x86-64 binary |
| Stockfish Elo caps | 1320, 1800, 2400 |
| Games | 2 per Elo cap, 6 total |
| Colors | Alternated by game |
| Threads | 1 for both engines |
| Hash | 16 MiB for both engines |
| Time control | 30 ms per move, implemented by the headless UCI runner |
| Maximum game length | 120 plies; unfinished games scored as draws at the cap |
| Unchessed mode | `Adaptive=true`, `PersonaSmooth=false`, `EngineDetectV2=false` |
| Raw records | `elo_matrix_real.json` |
| Summary | `elo_matrix_real.summary.json` |

The runner is `../run_elo_benchmark.py`. It validates every returned UCI move against `python-chess` legal move generation, alternates colors, records engine output tails, and stores the complete move list and telemetry in JSON.

## Results

| Stockfish UCI_Elo | Games | Unchessed wins | Draws | Unchessed losses | Score | Average plies |
|---:|---:|---:|---:|---:|---:|---:|
| 1320 | 2 | 0 | 2 | 0 | 0.500 | 120.0 |
| 1800 | 2 | 0 | 1 | 1 | 0.250 | 115.5 |
| 2400 | 2 | 0 | 0 | 2 | 0.000 | 103.5 |
| **Overall** | **6** | **0** | **3** | **3** | **0.250** | — |

These are real game results, but they are **not statistically sufficient for an Elo estimate or a strength claim**. The sample is only two games per cap, the time control is intentionally short, and three games reached the 120-ply ceiling and were recorded as draws. The Wilson intervals in the machine-readable summary are descriptive only and should not be treated as an SPRT.

The results do provide a first sanity check: the engine completed six games without an illegal-move failure, and performance degraded as the opponent cap increased under this fixed low-time setup. A production claim requires substantially more games, a real opening suite, longer controls, and a pre-registered SPRT.
