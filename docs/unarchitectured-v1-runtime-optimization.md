# Unarchitectured v1 Chessformer runtime optimization

## Scope

This work optimizes the canonical Unarchitectured v1 compact student. It does
not resume Hydra/Apex predecessor development and does not wire the model into
search.

Baseline source: validated `AegisV4Chessformer` Rust full forward from main.
Package SHA-256:

```text
5fd9fc3fbf47bd2620c2e832e24c98525b59feeea791abf1c7ae32b9d311b16d
```

## Measured result

Sandbox host: two visible logical CPUs, Intel Xeon 2.60GHz, Rust 1.97.1.

Round one reduced the naive 8/256 path from 208.61ms to 14.92ms in the original
sandbox run. Round two uses an alternating, same-process benchmark so the
retained-int8 and dequantized backends see the same host conditions:

| Threads | Dequantized f32 matrices | Retained-int8 matrices | Backend speedup |
|---:|---:|---:|---:|
| 1 | 21.487748 ms | 15.454061 ms | 1.3904x |
| 2 | 16.007506 ms | 13.007211 ms | 1.2307x |

Latest standalone exit-ladder measurements with two inference threads:

| Exit | Latency |
|---|---:|
| Full 8/256 | 12.775529 ms |
| Middle 4/192 | 5.213683 ms |
| Shallow 2/128 | 2.428812 ms |

The prompt's earlier 89ms baseline and the reviewer's 75ms → 14.55ms round-one
measurement were produced on different hardware. This report does not infer
cross-host latency from sandbox ratios.

## Optimizations retained

- stable runtime AVX2/FMA dispatch with scalar fallback;
- four-token matrix microkernel that loads a weight row once for four token
  dot products;
- token/output cache blocking;
- scoped two-way parallel QKV and FFN-up projections;
- parallel attention-head groups;
- contiguous SIMD attention value accumulation;
- SIMD projection, history, policy, regret, value, RMSNorm, and adapter dots;
- SIMD GAB template accumulation;
- removal of a full 32,768-float geometric-bias copy per layer;
- retained symmetric int8 deployment matrices;
- per-token dynamic int16 activation quantization;
- AVX2 i16×i8 products with i32 accumulation and scalar fallback;
- two-output and three-output integer microkernels that reuse four activation
  rows across policy/block and QKV projections; and
- AVX2 activation max/quantization kernels.

`UNCHESSED_INFERENCE_THREADS` can override the default, which is the visible CPU
count capped at four. This is deliberately independent of search threads; the
model is not wired into search yet.

## Rejected experiments

### AVX-512 f32 kernels

On this Xeon VM, AVX-512 increased full latency from about 35ms to 39ms, likely
from frequency effects and wider horizontal reductions. Runtime dispatch stays
on AVX2/FMA.

### Eight-token dot microkernel

Eight simultaneous accumulators increased register pressure and regressed the
optimized full path from about 14.6ms to 17.9ms. The four-token microkernel was
retained.

## Correctness

The existing full-exit gates still pass unchanged:

- `start_position_matches_python_reference`;
- `midgame_position_matches_python_reference`;
- `position_to_input_matches_hand_built_start_position`;
- every checked output remains within `5e-3`; and
- both full-exit positions retain the same best move as Python.

`narrow_exits_match_python_reference` freezes independently generated Python
vectors for 2/128 and 4/192. Logits and best moves hold the normal `5e-3` gate;
pooled evidence/representation use the documented `2e-2` tolerance because the
64-term reduction order differs from PyTorch.

`integer_matrix_path_stays_close_to_dequantized_path` also compares every exit's
complete logits, regret mean/log-scale, evidence, and active representation.
Maximum observed deltas on the fixture are 0.000121 or less and the enforced
gate is `5e-4`. This is an inference-drift gate, not deployment calibration.

## Integration decision

No search integration was enabled. The earlier synchronous full-forward hint
lost 0-20 because inference consumed the move clock.

The next safe sequence is:

1. benchmark on the actual deployment CPU with controlled thread counts;
2. calibrate all exits and the integer backend over disjoint deployment
   positions, not only the frozen parity fixtures;
3. trial shallow inference only behind a clock-surplus or asynchronous option;
4. charge all synchronous preprocessing to the move deadline;
5. measure integrated depth/NPS and mate/only-move safety; and
6. run an isolated paired-game SPRT.

The 2.43ms two-thread shallow path meets the requested low-millisecond
standalone target on this host, but it is not a strength claim.

## Remaining performance work

- evaluate calibrated int8 activations and VNNI/dot-product kernels without
  relaxing the existing parity and end-to-end drift gates;
- prepack/transcode matrices for wider deployment microkernels;
- stop materializing duplicate f32 copies for matrices once all fallback and
  non-x86 deployment constraints are settled;
- use a persistent inference worker rather than scoped threads if integration
  demonstrates enough call volume;
- cache only exact full-position results keyed by state/model/persona; and
- evaluate asynchronous root hints that never block the move clock.

Per-node use remains unrealistic for this 4.22M-parameter transformer without a
much smaller distilled evaluator.
