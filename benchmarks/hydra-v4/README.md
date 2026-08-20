# Hydra Aegis v4 calculated architecture budget

The report is generated from `config/unchessed_hydra_v4.json`:

```bash
python tools/hydra_v4_architecture_report.py \
  --config config/unchessed_hydra_v4.json \
  --json benchmarks/hydra-v4/report.json \
  --markdown benchmarks/hydra-v4/result.md --check
```

These are deterministic parameter, storage, and operation-count calculations.
They are not measured latency, model accuracy, integrated NPS, Elo, or SPRT
evidence.
