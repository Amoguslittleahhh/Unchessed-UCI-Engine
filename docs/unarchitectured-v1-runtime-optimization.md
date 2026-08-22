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

| Runtime path | Latency | Speedup vs naive full |
|---|---:|---:|
| Naive scalar full 8/256 | 208.61 ms | 1.00x |
| Optimized full 8/256 | 14.92 ms | 13.98x |
| Optimized middle 4/192 | 6.37 ms | 32.77x |
| Optimized shallow 2/128 | 2.66 ms | 78.37x |

The prompt's earlier 89ms baseline was measured on different hardware. This
report does not infer cross-host latency from the sandbox ratio.

## Optimizations retained

- stable runtime AVX2/FMA dispatch with scalar fallback;
- four-token matrix microkernel that loads a weight row once for four token
  dot products;
- token/output cache blocking;
- scoped two-way parallel QKV and FFN-up projections;
- parallel attention-head groups;
- contiguous SIMD attention value accumulation;
- SIMD projection, history, policy, regret, value, RMSNorm, and adapter dots;
- SIMD GAB template accumulation; and
- removal of a full 32,768-float geometric-bias copy per layer.

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

An additional test executes 2/128, 4/192, and 8/256 and checks output shapes,
finite logits/regrets/evidence, and zeroed inactive representation channels.
This is not independent shallow/middle numerical parity.

`tools/reference_forward_aegis_v4.py --all-exits` now prints the vectors needed
to freeze those missing parity fixtures on a machine with PyTorch.

## Integration decision

No search integration was enabled. The earlier synchronous full-forward hint
lost 0-20 because inference consumed the move clock.

The next safe sequence is:

1. generate and freeze independent Python vectors for 2/128 and 4/192;
2. add Rust parity tests at the current tolerance;
3. benchmark on the actual deployment CPU;
4. trial shallow inference only behind a clock-surplus or asynchronous option;
5. charge all synchronous preprocessing to the move deadline;
6. measure integrated depth/NPS; and
7. run an isolated paired-game SPRT.

The 2.66ms shallow path meets the requested low-millisecond standalone target on
this host, but it is not a strength claim.

## Remaining performance work

- retain int8 weights and add integer dot products instead of load-time f32
  dequantization;
- prepack/transcode matrices for deployment microkernels;
- use a persistent inference worker rather than scoped threads if integration
  demonstrates enough call volume;
- cache only exact full-position results keyed by state/model/persona; and
- evaluate asynchronous root hints that never block the move clock.

Per-node use remains unrealistic for this 4.22M-parameter transformer without a
much smaller distilled evaluator.
