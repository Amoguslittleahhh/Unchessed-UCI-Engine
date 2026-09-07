# Stockfish 19 calibration tournament summary

This is a small fixed-control research tournament, not a statistically significant Elo estimate. Each variant played two games per control with colors reversed.

| Variant | Movetime (ms) | Games | Candidate score | Wins | Draws | Losses | Median plies |
|---|---:|---:|---:|---:|---:|---:|---:|
| hce-default | 30 | 2 | 0.250 | 0 | 1 | 1 | 108.0 |
| hce-default | 100 | 2 | 0.000 | 0 | 0 | 2 | 86.5 |
| hce-default | 300 | 2 | 0.250 | 0 | 1 | 1 | 84.5 |
| hce-mobility-110 | 30 | 2 | 0.000 | 0 | 0 | 2 | 55.5 |
| hce-mobility-110 | 100 | 2 | 0.250 | 0 | 1 | 1 | 96.0 |
| hce-mobility-110 | 300 | 2 | 0.000 | 0 | 0 | 2 | 59.5 |
| hce-passed-110 | 30 | 2 | 0.000 | 0 | 0 | 2 | 74.5 |
| hce-passed-110 | 100 | 2 | 0.000 | 0 | 0 | 2 | 90.5 |
| hce-passed-110 | 300 | 2 | 0.250 | 0 | 1 | 1 | 115.5 |
| homemade-nonlinear | 30 | 2 | 0.000 | 0 | 0 | 2 | 40.5 |
| homemade-nonlinear | 100 | 2 | 0.250 | 0 | 1 | 1 | 85.5 |
| homemade-nonlinear | 300 | 2 | 0.250 | 0 | 1 | 1 | 103.5 |

Aggregate candidate scores are averages of 0/0.5/1 game points. The default HCE and nonlinear mode tied on the small sample; the default HCE is retained because it has no nonlinear overhead and the sample is too small to justify a strength change.
