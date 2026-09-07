# Homemade parity model fit

The model is trained only from black-box UCI scores and original Unchessed bitboard features. It uses a leakage-safe modulo-five holdout; no Stockfish source or network data is imported.

positions=120
train=96
test=24
lambda=10
baseline_test_mae_cp=244.417
model_train_mae_cp=158.921
model_test_mae_cp=232.488

weights=29.0756183042,165.050011876,33.3111274764,-28.1279586099,14.5404607618,53.9593866715,89.0217440204,22.5674077942,6.36106482837,11.5142968358,47.504983524,24.2963794598,16.4149212044,44.7864979209,7.44361477216e-14,11.9471892712,46.4900769586,24.5292930088,12.0971006247,27.0766769686,16.2443705529
