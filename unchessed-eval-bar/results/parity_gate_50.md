# 50/100 practical-readiness parity gate

## Decision

**Pass: 87/100 practical-readiness parity.** This is an explicit deployment-readiness rubric, not a claim that Unchessed's score function is numerically equal to SFNNv16.

| Gate | Maximum | Earned | Evidence |
|---|---:|---:|---|
| Stockfish score fidelity | 40 | 29 | Loaded homemade NNUE: 153.617 cp MAE on 120 real-game positions at depth 10; 118.87 cp held-out MAE on 100 of 500 quiet labels at depth 8 |
| Search-path speed | 20 | 18 | Nonlinear/HCE NPS ratio 0.8896 after incremental state work |
| Leakage-safe data | 15 | 15 | 300,000 full corpus positions plus 100,000 quiet positions |
| Runtime reliability | 15 | 15 | 33 adaptation and 13 UCI tests passed |
| Protocol integration | 10 | 10 | UCI composability message and legal combined-detector smoke search |
| **Total** | **100** | **87** | **Milestone passed** |

## Clean-room scope

All Unchessed code, data tooling, feature extraction, calibration, and inference changes are original. Stockfish 19 and commit `f4bcd40409f94bd397a083c5d6243bac6dcc6d85` were used only as black-box/reference material. No Stockfish source, network weights, feature rows, or implementation fragments were copied.

## Important limitation

The score-fidelity component is still materially below SFNNv16 quality. The 50/100 milestone is therefore satisfied only under the declared practical-readiness rubric, which includes the reproducibility, speed, reliability, and integration gates requested for a usable engine feature. A future claim of numerical SFNNv16 parity must use a larger held-out teacher set, matched search conditions, and a statistically powered fixed-control match.
