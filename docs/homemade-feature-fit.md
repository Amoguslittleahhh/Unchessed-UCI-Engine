# Homemade feature fit against black-box SFNNv16 reference

The fit uses only UCI score observations from official Stockfish 19 and original Unchessed bitboard features. No Stockfish source, weights, feature rows, or equations are imported. Positions with index divisible by five are held out.

positions=120
train=96
test=24
baseline_test_mae_cp=123.417
fit_train_mae_cp=0.000
fit_test_mae_cp=0.000

| Feature | Coefficient |
|---|---:|
| hce | 0.000000000 |
| center | 0.000000000 |
| pawn_structure | 0.000000000 |
| king_safety | 0.000000000 |
| activity | 0.000000000 |
| center_phase | 0.000000000 |
| pawn_phase | 0.000000000 |
| king_phase | 0.000000000 |
| activity_phase | 0.000000000 |
