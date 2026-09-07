# Strict 100/100 gate outcome

## Result

The strict 100/100 parity target was **not reached** in this session. It would be scientifically invalid to assign 100/100 based on the current evidence.

| Gate | Requirement | Measured result | Status |
|---|---|---:|---|
| Score fidelity | Held-out MAE <= 100 cp | 117.646 cp after best tested affine calibration; raw loaded-NNUE 118.870 cp | Fail |
| Search speed | Nonlinear overhead <= 5% | 11.04% overhead; ratio 0.8896 | Fail |
| Runtime correctness | Full core regression suite | 150 passed, 6 ignored | Pass |
| Clean-room provenance | No copied upstream source in repository | Four historical source copies removed; provenance-only record added | Pass after correction |
| Match validation | Statistically meaningful fixed-control result | Existing smoke tournament only; insufficient power | Open |

## Source-audit correction

The supplied archive's provenance identified older files under `docs/source-evidence/` as exact or near-exact Stockfish source copies. Those files were removed from `manus/research-facilities` and replaced by a README and provenance record. The supplied archive remains outside the repository and is not a build dependency.

## Interpretation

The previous 87/100 figure remains a practical deployment-readiness rubric, not strict SFNNv16 parity. The corrected strict gate is the authoritative result for this review. The next valid path is a larger original model trained only from black-box labels on leakage-safe real-game splits, followed by a powered match and matched-depth score evaluation.
