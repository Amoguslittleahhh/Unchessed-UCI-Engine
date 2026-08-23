# Unarchitectured v1 runtime benchmarks

Unarchitectured v1 is the sole canonical architecture name. Hydra and Apex
names are experimental history only.

- `runtime-forward-2026-08-22.{json,md}` records the retained-int8 matrix
  backend, dequantized comparison, and all-exit parity/drift gates.
- `runtime-forward-2026-08-23.{json,md}` records the cache/reduction round using
  alternating measurements against the merged round-two `main` worktree.
- `integration-trial-2026-08-23.json` records the fixture-disjoint calibration
  smoke and precharged shallow-hint search trial.
- `uci-candidate-2026-08-23.json` records the real default-off UCI wiring,
  short-clock gate, exact-hint smoke, safety tests, and unrun SPRT launcher.

All figures are host-specific instrumentation. The candidate remains default-off
and is not deployment calibration, Elo, or SPRT evidence.
