# Homemade breakthrough ablation

Scores are computed from the direct black-box Stockfish 19/SFNNv16 report. No Stockfish implementation or weights are used. `base` is the existing HCE score; every other row adds only the named original Unchessed feature family.

| Variant | MAE cp | Median absolute error cp |
|---|---:|---:|
| base | 180.442 | 100.000 |
| center_pressure | 183.850 | 115.000 |
| pawn_geometry | 177.633 | 103.000 |
| king_shelter | 179.875 | 104.000 |
| activity_space | 182.767 | 109.000 |
| coordination | 183.250 | 106.000 |
| outpost | 181.000 | 108.000 |
| bishop_rook | 180.408 | 103.000 |
| tempo | 180.700 | 101.000 |
| full_stack | 182.083 | 111.000 |
| selected_stack | 177.183 | 101.000 |

The final calibrated runtime applies the learned shrinkage and bias after the full stack; this table intentionally isolates feature-family contribution before that calibration.
