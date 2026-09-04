# Evaluator Provenance and Fair Benchmark Setup

The optimization copy now carries the canonical Unchessed NNUE asset at `unchessed-nnue.bin`. It is the same 23 MiB file present in the working main snapshot and has SHA-256:

`38845a16d73a6fe0bd4ac95c86c017c65c97bc82c7ce2f6dce2f1b3fbe8577b5`

The asset is intentionally force-added despite the repository's generic `*.bin` ignore rule so the optimization copy is self-contained for reproducible evaluation. No Unchessed policy-net file is present. Adaptive policy behavior therefore uses the documented heuristic-prior fallback unless a compatible policy file is supplied explicitly.

## Mandatory benchmark rule

A strength or speed measurement must explicitly load the NNUE file. The portable-versus-x86-64-v3 harness now requires:

```bash
scripts/benchmark-portable-v3.sh "$(pwd)/unchessed-nnue.bin"
```

The rapid match harness now requires:

```bash
python3 scripts/rapid-benchmark.py \
  --unchessed ./target/release/unchessed-adapter \
  --stockfish /usr/games/stockfish \
  --maia '<Maia-3 UCI command>' \
  --eval-file "$(pwd)/unchessed-nnue.bin" \
  --games 4 --initial-ms 180000 --inc-ms 2000 \
  --max-plies 240 --out benchmarks/results/rapid-nnue.json
```

Both harnesses fail before running a measurement when the file is missing or empty. They send `setoption name EvalFile value <absolute-path>` after the UCI handshake and record the evaluator path and SHA-256 in outputs where applicable. `PolicyFile` remains optional because the current tests disable Adaptive or use the engine's heuristic prior fallback.

## Historical artifacts

The earlier portable-v3 table is renamed `portable-v3-20260904-113843-hce.tsv`. The earlier rapid match is renamed `rapid-elo-uncertainty-20260904-hce.json`. Both were run before the explicit-evaluator correction and must be interpreted as HCE-only. They are not silently relabeled as NNUE measurements.
