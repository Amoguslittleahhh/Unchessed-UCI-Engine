# Homemade multi-breakthrough fit against black-box SFNNv16 reference

Only UCI score observations and original Unchessed bitboard signals are used. No Stockfish source, weights, feature rows, or formulas are imported. Positions with index divisible by five are held out.

positions=30
train=24
test=6
baseline_test_mae_cp=420.167
fit_train_mae_cp=34.990
fit_test_mae_cp=846.519

| Feature | Coefficient |
|---|---:|
| hce | 0.565087608 |
| center | -136.179545210 |
| pawn_structure | -313.637428520 |
| king_safety | -946.323700428 |
| activity | 1228.648635315 |
| coordination | -1232.965391526 |
| space | 188.600730567 |
| passed_pawns | 1432.632282118 |
| outposts | 1465.580747981 |
| bishop_pair | -6748.222632474 |
| rook_files | 215.180743743 |
| tempo | -397.695212966 |
| center_phase | 490.861277825 |
| pawn_phase | 342.362555014 |
| king_phase | 1054.279650477 |
| activity_phase | -1872.654854884 |
| coordination_phase | 1447.933733903 |
| space_phase | -118.134448940 |
| passed_phase | -1856.448618990 |
| outposts_phase | -1108.732413403 |
| bishop_pair_phase | 8556.935382254 |
| rook_files_phase | 143.453829162 |
| tempo_phase | 687.261979422 |
