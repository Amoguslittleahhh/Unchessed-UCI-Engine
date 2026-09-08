# Homemade NNUE Reference Breakthrough

## Measured improvement

The authorized `unchessed-nnue.bin` handoff is already the exact network loaded by the branch. Its SHA-256 is `38845a16d73a6fe0bd4ac95c86c017c65c97bc82c7ce2f6dce2f1b3fbe8577b5`. Loader and accumulator validation passed all 15 NNUE tests, including color-mirror symmetry, feature dimensions, SIMD/scalar agreement, special-move incremental refresh, and non-degenerate output.

Against the exact Stockfish reference executable built from commit `edb0d9db6731067ec50ce619ff372b463bc4dd5d`, the supplied network achieved the best direct result measured so far on the fixed 120-position corpus at depth 8:

| Evaluator | MAE (cp) | Median absolute error (cp) | Maximum error (cp) |
|---|---:|---:|---:|
| Supplied homemade NNUE | **148.683** | **80.5** | 1,352 |
| Calibrated research mode | 173.033 | 99.0 | 1,137 |
| Homemade mode | 174.592 | 103.0 | 1,152 |
| Tactical multi-head | 181.767 | 113.0 | 1,389 |

The supplied model therefore reduces MAE by 14.35% relative to the previous best calibrated mode. This is a meaningful architecture/data result, but it remains above the strict 100 cp all-position gate.

## Mathematical analysis

Five-fold cross-validation was used to test whether a simple post-hoc correction could explain the remaining error. A feature matrix contained the raw network score, material imbalance, absolute material imbalance, non-pawn phase, halfmove clock, empty-square count, phase square, and material-phase interaction. The supplied NNUE had raw cross-validated MAE 148.683 cp; adding these low-dimensional corrections produced 153.545 cp, while a pure affine correction produced 166.586 cp. Thus the raw learned output is already better than these simple corrections, and applying them would be harmful.

The result identifies a useful direction: the remaining gap is not a simple global scale/bias defect. It is more likely caused by teacher-target mismatch, depth/position distribution mismatch, terminal and tactical tails, and missing Stockfish-style feature families or output calibration. The shipped network was trained on self-generated HCE search labels rather than Stockfish/SFNNv16 labels, so the direct comparison is a domain-shift measurement rather than a training-quality failure.

## Architecture implication

The supplied network is a valid original Unchessed NNUE, but it is deliberately smaller and structurally different from the audited SFNNv16 reference: it uses 22,528 HalfKAv2_hm inputs, 256 accumulator channels per perspective, SCReLU, one 512-to-1 output, f32 inference, and no threat/pawn-pair inputs, PSQT split, hidden layers, or quantized integer path. The strongest next experiment is therefore not more affine calibration. It is training an original teacher-labeled model with a richer feature state and separate terminal/tactical targets, while preserving the current network as the stable fallback.

## Conclusion

The network handoff materially improved the measured parity result, but it did not establish stable 100/100 parity. The repository must continue to report the 148.683 cp result honestly until a leakage-safe Stockfish-labeled corpus, independent retraining, and untouched-data validation pass the 100 cp and terminal gates.
