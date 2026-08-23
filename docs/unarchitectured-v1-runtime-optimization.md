# Unarchitectured v1 Chessformer runtime optimization

## Scope

This work optimizes the canonical Unarchitectured v1 compact student. It does
not resume Hydra/Apex predecessor development and does not wire the model into
search.

Baseline source: validated `UnarchitecturedV1Student` Rust full forward from main.
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

Round three compares the merged round-two `main` worktree with the new runtime
in alternating order on the same host. Each figure is the median of three
200-call release runs:

| Threads | Merged round 2 | Round 3 | Speedup |
|---:|---:|---:|---:|
| 1 | 15.831563 ms | 14.697528 ms | 1.0772x |
| 2 | 14.158920 ms | 13.163678 ms | 1.0756x |

A separate two-thread round-three exit-ladder run measured 13.102574ms at full
8/256, 5.613094ms at middle 4/192, and 2.570135ms at shallow 2/128. Shared-host
load makes absolute runs vary, which is why the alternating comparison—not a
cross-session minimum—is the speedup evidence.

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
  rows across policy/block and QKV projections;
- AVX2 activation max/quantization kernels;
- in-register AVX2 horizontal reductions instead of store-and-scalar folds;
- output-major blocking for ordinary integer linears, retaining each weight-row
  pair across all token blocks;
- one dispatch resolution per projection/attention operation instead of one
  `OnceLock` lookup per inner dot or AXPY;
- static per-layer tensor names instead of formatting roughly 80 names on every
  forward; and
- retained-int8 regret source/target projections under the existing complete-
  output drift gate.

`UNCHESSED_INFERENCE_THREADS` can override the default, which is the visible CPU
count capped at four. This is deliberately independent of search threads; the
model is not wired into search yet.

## Rejected experiments

### AVX-512 f32 and VNNI kernels

The earlier AVX-512 f32 experiment increased full latency from about 35ms to
39ms. Round three separately tested AVX-512 VNNI `vpdpwssd` on the retained-int8
backend; despite exact scalar agreement, it regressed this host from roughly
16.1/12.8ms to 23.4/19.1ms at one/two threads. Frequency throttling outweighed
the wider dot instruction, so runtime dispatch remains AVX2.

### Polynomial exponential

A range-reduced degree-six exponential passed all parity gates when substituted
for attention softmax and SiLU. It nevertheless regressed the full path to about
17.0ms single-thread and 14.3–14.6ms two-thread on this host. The standard
`f32::exp` path was restored; correctness passing is necessary but not enough to
keep an optimization.

### Output-major QKV/FFN-up blocking

Retaining QKV and FFN-up weight rows across every token block increased strided
output traffic and regressed the single-thread path to about 15.6–15.9ms. Only
ordinary linear projections retain the output-major ordering, where controlled
runs improved rather than regressed.

### Pairwise pooled-token reduction

A binary-tree 64-token pool produced effectively the same narrow-exit Python
differences as the sequential sum and added strided accesses. The reviewed
`2e-2` pooled-output tolerance was neither widened nor presented as fixed; the
sequential SIMD pool remains.

### Eight-token dot microkernel

Eight simultaneous accumulators increased register pressure and regressed the
optimized full path from about 14.6ms to 17.9ms. The four-token microkernel was
retained.

### Int8 activation prototype

A signed-int8 activation prototype used AVX2 `abs/sign/maddubs/madd` products
against the retained int8 weights. Per-token symmetric quantization exceeded the
existing Python gate (for example, the start-position first logit differed by
about `1.01e-2`, versus the required `5e-3`). Affine and small-group variants
also failed at least one frozen parity component. They were reverted rather than
loosening correctness. The retained backend therefore uses int16 activations;
its measured complete-output drift is below `1.21e-4`.

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
Round-three maximum observed deltas on the fixture are `9.4e-5` or less and the
enforced gate remains `5e-4`. This is an inference-drift gate, not deployment
calibration.

## Integration decision

Normal UCI play still has no neural integration. The earlier synchronous
full-forward hint lost 0-20 because inference consumed the move clock.

Round four added only a default-unreachable trial layer:

- a bounded nonblocking worker whose results require an exact position/persona/
  legal-action/exit key;
- a root-hint search entry point that keeps every legal move, uses policy only
  on the first pass, and charges synchronous preprocessing to the same deadline;
- an eight-position fixture-disjoint calibration smoke;
- three precharged `movetime 75` integrated depth/NPS trials; and
- adversarial mate, only-move, and stale-key tests.

The smoke corpus lacks training-membership provenance and uses depth-4 HCE
labels. The sandbox is not identified deployment hardware. One of eight best
moves differed in every time-limited trial. Therefore no UCI option was added,
`runtime_safety_suite` remains false, and no SPRT was attempted. See
`docs/unarchitectured-v1-integration-trial.md` for measurements and limitations.

The next safe sequence is now owner-dependent: supply the deployment CPU and a
representative provenance-disjoint teacher-labelled corpus, promote the harness
to a default-off UCI candidate, run the complete tactical/depth gate, and only
then run an isolated paired-game SPRT.

## Remaining performance work

- evaluate calibrated int8 activations and VNNI/dot-product kernels without
  relaxing the existing parity and end-to-end drift gates;
- prepack/transcode matrices for wider deployment microkernels;
- stop materializing duplicate f32 copies for matrices once all fallback and
  non-x86 deployment constraints are settled;
- evaluate whether the new persistent outer inference worker reduces real UCI
  latency once an owner-gated candidate exists (inner matrix scopes remain);
- extend exact-key caching only if real repeated-position hit rates justify it;
  and
- keep asynchronous root hints default-off until deployment calibration,
  tactical/depth gates, and paired-game SPRT all pass.

Per-node use remains unrealistic for this 4.22M-parameter transformer without a
much smaller distilled evaluator.
