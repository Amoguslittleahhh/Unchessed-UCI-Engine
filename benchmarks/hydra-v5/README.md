# Experimental Hydra Apex v5 calculated architecture budget

Reproduce the committed parameter, memory, CPU-layout, and operation counts:

```bash
python tools/hydra_v5_architecture_report.py \
  --config config/unchessed_hydra_v5.json \
  --student-config config/unchessed_hydra_v4.json \
  --json benchmarks/hydra-v5/report.json \
  --markdown benchmarks/hydra-v5/result.md --check
```

The base 58.4M Oracle and the resolved 29.1M-878.1M Verda GPU profiles are
training-only and are distilled into the compact runtime student. These are
calculations, not measured GPU/4-360-vCPU throughput, model accuracy, integrated
NPS, Elo, or SPRT evidence.
