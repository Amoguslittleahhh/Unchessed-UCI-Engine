# Eval-bar breakthrough benchmark v4

Date: 2026-09-07. Branch: `manus/research-facilities`.

The benchmark uses the repository’s real `benchmarks/matetrack.epd` corpus, containing seven legal mate-track positions. The WDL benchmark performs 10,000 rounds over all seven positions after warming the material-indexed tensor. The exact reference uses the Stockfish 19 logistic equations with `exp`; the fast path uses the precomputed `[material_index][display_cp]` tensor.

| Measurement | Result |
|---|---:|
| Positions | 7 |
| Rounds | 10,000 |
| Exact reference elapsed | 4,964,388 ns |
| Fast tensor elapsed | 1,770,786 ns |
| Steady-state speedup | 2.80x |
| Maximum WDL delta | 0 per mille |
| Exact checksum | 62,020,000 |
| Fast checksum | 62,020,000 |

The speedup is approximately 2.80x on this sandbox CPU for the WDL projection alone. It does not claim a 3.10x whole-engine speedup because board evaluation and search dominate normal engine operation.

The focused eval-bar suite passed 6/6 tests. The workspace compiled successfully. The reviewer binary was exercised through UCI with `uci`, `isready`, `position`, and `evalbar`. The start position returned `cp 14`, `wdl 32 957 11`, and `bar 0.510`; a legal queen-versus-king position returned `cp 1164`, `wdl 1000 0 0`, and `bar 1.000`. Both outputs explicitly reported `provenance=proxy`.

The implemented breakthroughs are: (1) material-indexed calibration caching; (2) a precomputed integer WDL tensor that removes runtime logistic exponentials in the normal ±4000 cp range; (3) confidence-adaptive, bounded smoothing that preserves raw scores while preventing single-update visual shocks; and (4) a provenance-safe UCI integration that cannot silently claim exact Stockfish SFNNv16 parity.
