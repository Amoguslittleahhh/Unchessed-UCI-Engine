# Hydra Aegis v3 calculated architecture budget

These files are deterministic calculations from
`config/unchessed_hydra_v3.json`, not measured latency, accuracy, NPS, Elo, or
SPRT evidence.

```bash
python tools/hydra_v3_architecture_report.py \
  --config config/unchessed_hydra_v3.json \
  --json benchmarks/hydra-v3/report.json \
  --markdown benchmarks/hydra-v3/result.md --check
```

`feature-microbenchmark.{json,md}` separately records a measured Rust
full-refresh extraction plus synthetic-accumulation result. It is a standalone
CPU microbenchmark, not an integrated NPS or Elo result.

The architecture, data ABI, implemented status, equations, and promotion gates
are documented in `docs/unchessed-hydra-v3-mathematics.md`.
