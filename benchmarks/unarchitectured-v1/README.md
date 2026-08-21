# Unarchitectured v1 calculated architecture budget

Unarchitectured v1 is the sole canonical architecture name. Hydra v1-v5 and the
Apex v1 naming candidate remain experimental history.

```bash
python tools/unarchitectured_v1_architecture_report.py \
  --json benchmarks/unarchitectured-v1/report.json \
  --markdown benchmarks/unarchitectured-v1/result.md --check
```

The report also freezes autonomous-safety, efficient-epoch, and feature-schema
contracts. These are architecture calculations, not trained-model accuracy,
throughput, NPS, Elo, or SPRT evidence.
