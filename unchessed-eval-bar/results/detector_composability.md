# Detector composability and practical-parity update

## Problem reproduced from the real-game report

The previous `OpponentModel::suspect_reason()` returned immediately to `suspect_reason_v2()` whenever `EngineDetectV2=true`. As a result, `AcceleratedDetection=true` was silently ignored whenever both options were enabled. This matched the attached report: a real 60+1 game against Stockfish 19 never reached `Mode::Full` despite strong evidence, while AcceleratedDetection alone reached Full.

## Clean-room fix

When both options are enabled, V2 evidence remains active and the accelerated resilient channel can independently promote. A new telemetry reason, `composed_accelerated_resilient`, distinguishes this path. The UCI handler emits an explicit informational message whenever the combination is active. V2 also accepts a bounded resilient-mass fallback for strong but noisy play; twelve uninterrupted low-loss observations are no longer the only anonymous-ceiling route.

## Validation

| Check | Result |
|---|---:|
| Adaptation tests | 33 passed |
| UCI tests | 13 passed |
| Composition regression | Passed |
| Noisy strong V2 regression | Passed |
| Release UCI warning smoke test | Passed |
| Quiet real-game manifest | 100,000 positions |

The detector change improves practical adapter behavior and removes a real configuration failure. It does not, by itself, establish SFNNv16 evaluation parity. The existing direct score comparison remains 178.925 cp MAE on 120 real-game positions, and the requested 50/100 parity milestone is not claimed.
