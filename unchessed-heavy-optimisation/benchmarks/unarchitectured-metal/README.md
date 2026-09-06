# Unarchitectured Metal runtime benchmarks

Unarchitectured Metal is the sole canonical architecture name. Hydra and Apex
names are experimental history only.

- `runtime-forward-2026-08-22.{json,md}` records the retained-int8 matrix
  backend, dequantized comparison, and all-exit parity/drift gates.
- `runtime-forward-2026-08-23.{json,md}` records the cache/reduction round using
  alternating measurements against the merged round-two `main` worktree.
- `integration-trial-2026-08-23.json` records the fixture-disjoint calibration
  smoke and precharged shallow-hint search trial.
- `uci-candidate-2026-08-23.json` records the real default-off UCI wiring,
  short-clock gate, exact-hint smoke, safety tests, and unrun SPRT launcher.
- `pretrain-probe-2026-08-28.json` records the sandbox move-prediction
  pretrain probe (13,076 committed self-play rows, 256-wide NumPy MLP):
  the 0/200 rating-conditioning sweep inverts to 118/200 with dual-elo
  conditioning, top-1 accuracy 0.1687 vs 0.0879 baseline
  (`docs/move-prediction-pretrain-plan.md`).

All figures are host-specific instrumentation. The candidate remains default-off
and is not deployment calibration, Elo, or SPRT evidence.
