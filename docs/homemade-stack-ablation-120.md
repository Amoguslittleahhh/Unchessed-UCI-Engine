# Homemade breakthrough ablation

Scores are computed from the direct black-box Stockfish 19/SFNNv16 report. No Stockfish implementation or weights are used. `base` is the existing HCE score; every other row adds only the named original Unchessed feature family.

| Variant | MAE cp | Median absolute error cp |
|---|---:|---:|
| base | 187.808 | 107.000 |
| center_pressure | 190.167 | 114.000 |
| pawn_geometry | 182.417 | 101.000 |
| king_shelter | 186.642 | 106.000 |
| activity_space | 189.550 | 115.000 |
| coordination | 189.533 | 106.000 |
| outpost | 188.467 | 110.000 |
| bishop_rook | 188.275 | 108.000 |
| tempo | 187.683 | 105.000 |
| full_stack | 189.158 | 118.000 |

The final calibrated runtime applies the learned shrinkage and bias after the full stack; this table intentionally isolates feature-family contribution before that calibration.
