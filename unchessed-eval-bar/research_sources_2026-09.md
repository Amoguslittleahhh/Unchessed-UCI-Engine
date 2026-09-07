# External research sources used in the 50/100 parity push

## Stockfish reference

Official Stockfish commit f4bcd40409f94bd397a083c5d6243bac6dcc6d85: https://github.com/official-stockfish/Stockfish/commit/f4bcd40409f94bd397a083c5d6243bac6dcc6d85

The commit title is “Update nnue architecture to SFNNv16 and net nn-89cb98a217f7.nnue”. Its public message describes adding local pawn-pair features indexed over the same or adjacent files, removing overlapping threat inputs, and optimizing the implementation from an initially large slowdown to a reported approximately 3.5% overall slowdown. Unchessed uses this only as a black-box/design reference and does not copy its code, feature rows, formulas, or weights.

## arXiv sources

1. Y. Choi, “Study of the Proper NNUE Dataset,” arXiv:2412.17948, https://arxiv.org/abs/2412.17948. Search result summary: proposes generating and filtering quiet positions that are stable and free from tactical volatility for NNUE training.

2. B. Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” arXiv:1712.05877, https://arxiv.org/abs/1712.05877. Search result summary: proposes integer-only inference with a co-designed quantization/training procedure.

3. “Adaptive Opponent Policy Detection in Multi-Agent MDPs: Real-Time Strategy Switch Identification Using Running Error Estimation,” arXiv:2406.06500, https://arxiv.org/abs/2406.06500. Search result summary: uses dynamic error decay for online policy-switch detection, motivating bounded leaky evidence rather than requiring an uninterrupted perfect streak.

4. D. Klein, “Neural Networks for Chess,” arXiv:2209.01506, https://arxiv.org/abs/2209.01506. Search result summary: provides a technical introduction to neural-network chess evaluation and NNUE.

## Current measured evidence

A persistent UCI Stockfish 19 labeler produced 500 quiet real-game labels at depth 8. The existing homemade HCE proxy scored 120.132 cp MAE overall and 122.900 cp on the modulo-five held-out subset. The existing homemade NNUE scored 114.22 cp overall and 118.87 cp held out. These are calibration results on a quiet subset, not claims of SFNNv16 parity.
