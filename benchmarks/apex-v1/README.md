# Unchessed Apex v1 calculated architecture budget

Apex v1 is the canonical name for the architecture previously developed under
the experimental Hydra v1-v5 lineage.

```bash
python tools/apex_v1_architecture_report.py \
  --json benchmarks/apex-v1/report.json \
  --markdown benchmarks/apex-v1/result.md --check
```

The figures are calculated budgets, not trained-model accuracy, hardware
throughput, NPS, Elo, or SPRT evidence.
