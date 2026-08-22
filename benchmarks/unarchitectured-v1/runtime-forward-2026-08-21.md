# Unarchitectured v1 runtime-forward optimization

Measured on the sandbox's two-visible-CPU Intel Xeon 2.60GHz host with Rust
1.97.1 and the real exported package whose SHA-256 is
`5fd9fc3fbf47bd2620c2e832e24c98525b59feeea791abf1c7ae32b9d311b16d`.

| Path | Latency | Speedup vs naive full |
|---|---:|---:|
| Naive scalar 8-layer/256 baseline | 208.61 ms | 1.00x |
| Optimized full 8-layer/256 | 14.92 ms | 13.98x |
| Optimized middle 4-layer/192 | 6.37 ms | 32.77x |
| Optimized shallow 2-layer/128 | 2.66 ms | 78.37x |

Optimizations:

- runtime-selected AVX2/FMA dot and scaled-add kernels with scalar fallback;
- four-token SIMD matrix microkernel reusing each weight row;
- cache-blocked token/output loop ordering;
- scoped two-way QKV, FFN-up, and attention-head parallelism;
- contiguous attention value accumulation;
- SIMD GAB template accumulation;
- removal of an intermediate 32,768-float GAB copy; and
- SIMD policy, regret, history, value, RMSNorm, and adapter projections.

The original full-exit Python parity tests still pass within `5e-3`, including
identical best moves, and the real `Position -> PositionInput` test passes.

The middle and shallow paths use the already-trained Matryoshka prefixes and
pass shape/finiteness tests, but do not yet have independent Python reference
vectors. They are not wired into search and require parity, calibration, and
SPRT gates first.

These are standalone latency measurements on one shared host, not integrated
search NPS or Elo.
