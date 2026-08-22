# Hydra Aegis v2 calculated architecture budget

These files are deterministic calculations from
`config/unchessed_hydra_v2.json`, not measured latency, accuracy, or Elo.

```bash
python tools/hydra_v2_architecture_report.py \
  --config config/unchessed_hydra_v2.json \
  --json benchmarks/hydra-v2/report.json \
  --markdown benchmarks/hydra-v2/result.md --check
```

The full architecture and equations are documented in
`docs/unchessed-hydra-v2-mathematics.md`.
