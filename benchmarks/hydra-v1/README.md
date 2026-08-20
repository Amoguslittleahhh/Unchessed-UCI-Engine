# Hydra v1 calculated architecture budget

`report.json` and `result.md` are deterministic calculations from
`config/unchessed_hydra_v1.json`. They are not measured latency, accuracy, or
Elo results.

Regenerate and verify:

```bash
python tools/hydra_architecture_report.py \
  --config config/unchessed_hydra_v1.json \
  --json benchmarks/hydra-v1/report.json \
  --markdown benchmarks/hydra-v1/result.md --check
```

The mathematical specification and acceptance criteria are in
`docs/unchessed-hydra-mathematics.md`.
