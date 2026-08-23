# Unarchitectured v1 retained-int8 runtime — round 2

Measured on the sandbox's two-visible-CPU Intel Xeon 2.60GHz host with Rust
1.97.1 and the real exported package whose SHA-256 is
`5fd9fc3fbf47bd2620c2e832e24c98525b59feeea791abf1c7ae32b9d311b16d`.
Runtime source SHA-256:
`a7823ba4f2a587faa1d78331a2e8a88eba65c0763aff783bfe4b761e0e6df5d0`.

## Backend comparison

The controlled benchmark alternates four 50-call rounds per path in one process
after five warmups. The comparison therefore measures the arithmetic backend,
not two builds run under unrelated host load.

| Inference threads | Dequantized f32 matrices | Retained-int8 matrices | Speedup |
|---:|---:|---:|---:|
| 1 | 21.487748 ms | 15.454061 ms | 1.3904x |
| 2 | 16.007506 ms | 13.007211 ms | 1.2307x |

The deployment backend retains the package's symmetric int8 matrix weights,
dynamically quantizes each activation row to int16, accumulates i16×i8 products
into i32, then applies both scales once per output. AVX2 kernels process four
tokens and two or three output rows together; non-AVX2 targets use a scalar
integer fallback. Activation max-reduction and quantization are also vectorized.

## Two-thread exit ladder

| Exit | Latency |
|---|---:|
| Shallow 2/128 | 2.428812 ms |
| Middle 4/192 | 5.213683 ms |
| Full 8/256 | 12.775529 ms |

These are standalone calls. They do not include or predict integrated search
NPS, achieved depth, clock behavior, or game strength.

## Numerical gates

`integer_matrix_path_stays_close_to_dequantized_path` compares every exit's
complete policy logits, regret means, regret log-scales, evidence, and active
representation. The enforced maximum absolute drift is `5e-4`.

| Exit | Logits | Regret mean | Regret log-scale | Evidence | Representation |
|---|---:|---:|---:|---:|---:|
| 2/128 | 0.00012088 | 0.00000900 | 0.00001705 | 0.00001311 | 0.00000736 |
| 4/192 | 0.00006807 | 0.00002086 | 0.00001538 | 0.00000811 | 0.00001090 |
| 8/256 | 0.00009680 | 0.00003862 | 0.00005136 | 0.00002122 | 0.00002832 |

The independent Python parity tests remain unchanged: full logits/evidence and
narrow logits use `5e-3`; narrow pooled evidence/representation use the
reviewed `2e-2` reduction-order tolerance. Best moves match at all exits, and
the live `Position → PositionInput` conversion test passes.

## Scope and remaining gates

This round does **not** wire the model into search. Before any strength-changing
use, the backend and selected exit still need representative deployment-position
calibration, actual deployment-CPU measurements, exact clock charging,
integrated depth/NPS, mate/only-move safety, and an isolated paired-game SPRT.
No Elo or tactical-safety claim is made here.
