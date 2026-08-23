# Unarchitectured v1 runtime benchmarks

Unarchitectured v1 is the sole canonical architecture name. Hydra and Apex
names are experimental history only.

- `runtime-forward-2026-08-22.{json,md}` records the retained-int8 matrix
  backend, dequantized comparison, and all-exit parity/drift gates.
- `runtime-forward-2026-08-23.{json,md}` records the cache/reduction round using
  alternating measurements against the merged round-two `main` worktree.

These files contain host-specific standalone latency measurements. They are not
integrated search NPS, clock-safety, Elo, or SPRT evidence.
