# Unarchitectured Metal benchmark campaign — 2026-09-06

This campaign compares the scalar fallback against the runtime-dispatched AVX2/FMA path using the migrated Unarchitectured Metal release binary. It also validates the AArch64 Linux and Apple Silicon compilation targets.

## Controls

Each condition used one search thread, Adaptive disabled, opening book disabled, a deterministic valid `UNCHNNUE` v1 benchmark fixture, 500,000 requested nodes, four fixed positions, hash sizes of 4/16/64 MiB, and three repeats. The harness waits for `bestmove` and records only completed UCI searches. The scalar condition sets `UNCHESSED_DISABLE_SIMD=1`; dispatch uses the cached CPU capability registry.

## Results

| Build | Scalar median NPS | Dispatched median NPS | Difference |
|---|---:|---:|---:|
| Native x86-64 | 742,908 | 1,404,292 | +89.03% |
| x86-64-v3 | 777,416 | 1,357,775 | +74.65% |

The native matrix contains 36 scalar and 36 dispatched rows. The x86-64-v3 matrix contains the same 36 + 36 structure. The detailed raw CSV files and generated summaries are in this directory.

The native x86-64 median results by position are 782,627 versus 1,466,451 NPS for start position, 666,220 versus 1,175,468 for Kiwipete, 722,773 versus 1,306,969 for the middlegame, and 892,960 versus 1,485,827 for the endgame. These figures are throughput measurements under the deterministic fixture, not Elo results.

## ARM targets

`aarch64-unknown-linux-gnu` and `aarch64-apple-darwin` both passed workspace `cargo check`. The benchmark host is x86-64, has no QEMU AArch64 runner, and has no Apple Silicon hardware. Consequently, ARM NEON NPS is not claimed here. Native AArch64 execution remains a required follow-up.

## Reproduction

```bash
python3 scripts/make-benchmark-nnue.py benchmarks/artifacts/benchmark-v1.unchnnue
cargo build --workspace --release
python3 scripts/benchmark-dispatch.py --binary target/release/unchessed-adapter \
  --output benchmarks/results/2026-09-06/native-dispatch.csv \
  --label native-avx2-fma-dispatch --nodes 500000 --repeats 3 --hash 4 16 64
python3 scripts/benchmark-dispatch.py --binary target/release/unchessed-adapter \
  --output benchmarks/results/2026-09-06/native-scalar.csv \
  --label native-scalar-fallback --scalar --nodes 500000 --repeats 3 --hash 4 16 64
```
