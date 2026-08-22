# Unarchitectured v1 calculated architecture budget

Unarchitectured v1 is the sole canonical architecture name. Hydra v1-v5 and the
Apex v1 naming candidate remain experimental history.

```bash
python tools/unarchitectured_v1_architecture_report.py \
  --json benchmarks/unarchitectured-v1/report.json \
  --markdown benchmarks/unarchitectured-v1/result.md --check
```

The report also freezes autonomous-safety, efficient-epoch, and feature-schema
contracts. `runtime-forward-2026-08-21.{json,md}` records the first f32/SIMD
round; `runtime-forward-2026-08-22.{json,md}` records the retained-int8 matrix
backend, controlled dequantized comparison, and all-exit parity/drift gates.
Architecture figures are calculations; runtime files are standalone
measurements, not NPS or Elo.
