# Unarchitectured Metal asymmetric latency summary

This report is derived from real UCI games against Stockfish 16, not simulated observations.

## normal_80ms

| Arm | Games | Full rate | Mean first-Full ply | Median first-Full ply | Fusion-trigger rate |
|---|---:|---:|---:|---:|---:|
| standard | 10 | 1.000 | 30 | 30.0 | 0.000 |
| fusion | 10 | 1.000 | 30.6 | 31.0 | 0.400 |

Paired complete games: 10; fusion-minus-standard latency mean: 0.6 plies; earlier/same/later pairs: 2/4/4.

## fast_30ms

| Arm | Games | Full rate | Mean first-Full ply | Median first-Full ply | Fusion-trigger rate |
|---|---:|---:|---:|---:|---:|
| standard | 6 | 1.000 | 30 | 31.0 | 0.000 |
| fusion | 6 | 1.000 | 29.666666666666668 | 31.0 | 0.500 |

Paired complete games: 6; fusion-minus-standard latency mean: -0.3333333333333333 plies; earlier/same/later pairs: 2/3/1.
