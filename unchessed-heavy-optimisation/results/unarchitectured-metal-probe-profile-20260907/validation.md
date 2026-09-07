# Observation-probe saturation validation

The final validation used the release adapter built from `manus/research-facilities`, Stockfish 19 Linux universal, the committed `unarchitectured-metal-final.unmetal` artifact, one thread, 64 MiB hash, `Adaptive=true`, `OwnBook=false`, `AdapterTelemetry=true`, `EngineDetectV2=false`, and `UCI_Opponent=- - human UnknownOpponent`. The driver used two fixed-opening games per arm, a 48-ply horizon, and a 60,000 ms real UCI clock per side. The standard arm used `AcceleratedDetection=false`; the fusion arm used `AcceleratedDetection=true`.

Both arms produced 23 observations per game and zero low-time skips. Standard first-Full plies were 30 and 30, for a mean of 30. The fusion first-Full plies were 26 and 24, for a mean of 25. Both fusion confirmations used `legacy_accelerated_resilient`. One fusion game recorded three `probe_saturated` observations after the held-verdict counter reached the saturation threshold. The remaining observations used `probe_high_fidelity`; book observations were reported separately.

The implementation was additionally exercised by an 80-ply one-game-per-arm real Stockfish 19 run. The standard arm recorded nine saturated observations after first Full; the fusion arm did not reach ten consecutive public suspect observations in that trajectory. This is expected and is useful negative evidence: the throttle engages only when its explicit held-verdict invariant is met.

The validation is not a strength test and does not establish a universal speedup. It establishes that the throttle is real, auditable, clock-safe, compatible with strict resilient promotion, and capable of engaging on a long game without suppressing observation coverage.
