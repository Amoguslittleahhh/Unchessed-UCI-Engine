# Homemade multi-breakthrough fit against black-box SFNNv16 reference

Only UCI score observations and original Unchessed bitboard signals are used. No Stockfish source, weights, feature rows, or formulas are imported. Positions with index divisible by five are held out.

positions=120
train=96
test=24
baseline_test_mae_cp=257.833
fit_train_mae_cp=140.748
fit_test_mae_cp=250.003

| Feature | Coefficient |
|---|---:|
| hce | 0.482647796 |
| center | -29.813706553 |
| pawn_structure | 81.046347203 |
| king_safety | 225.288447303 |
| activity | -23.238408027 |
| coordination | -16.290204244 |
| space | 27.060655381 |
| passed_pawns | -117.109679269 |
| outposts | 277.303724855 |
| bishop_pair | 299.000640448 |
| rook_files | 186.661862089 |
| tempo | 115.708817049 |
| center_phase | 61.710333308 |
| pawn_phase | -104.951293590 |
| king_phase | -252.166994806 |
| activity_phase | 26.998510009 |
| coordination_phase | 6.020941228 |
| space_phase | -43.571207508 |
| passed_phase | 256.555067594 |
| outposts_phase | -294.930955671 |
| bishop_pair_phase | -407.663233869 |
| rook_files_phase | -24.057180708 |
| tempo_phase | -136.024157977 |
