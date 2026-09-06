# Unarchitectured Metal Chessformer runtime optimization

## Scope

This work optimizes the canonical Unarchitectured Metal compact student. It does
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

## Round 11 result (2026-08-28)

Task: make the forward pass fast enough that the root hint is actually
usable. This host: two visible logical CPUs, Intel Xeon 2.60GHz,
48 KiB L1d, **1.3 MiB L2 (shared)**, Rust 1.88.0.

### What changed (all in `unchessed-core/src/unarchitectured_metal_runtime.rs`)

1. **Inlined attention** (`attention_heads_inlined`, default; the
   indirect-kernel path remains behind `UNCHESSED_ATTN_INLINE=0`):
   the query row is loaded once per query; every (query, key) dot is
   four fused AVX2 chunk accumulations instead of an indirect kernel
   call with its own loop setup and tail check; each query's value
   combination keeps its chunk accumulators in registers across all 64
   keys instead of re-writing the output row on every key. Same
   operations, slightly different partial-sum order (parity gates
   hold: `start/midgame_position_matches_python_reference` and the
   frozen narrow-exit vectors all pass with it enabled).
2. **Vectorized exp for the attention softmax** (`exp8_softmax`,
   default; `UNCHESSED_SOFTMAX_POLY=0` restores `f32::exp`):
   range-reduced 2^n form, degree-six Taylor on [0, ln 2), max
   relative error ~5e-7; values below -50 clamp to exp(-50) (zero
   contribution to any normalized weight). Unit-tested against the
   scalar on 64 values in [-50, 0] to 1e-4
   (`exp8_softmax_matches_scalar`), and all three reference gates pass
   with it enabled.
3. **Per-forward scratch reuse** (`BlockScratch`): the eight layers
   reuse one set of buffers instead of allocating + zeroing ~0.9 MB of
   vecs per layer; every buffer is fully overwritten by its user.
4. **Per-head geometric-bias streaming**: each head's 64x64 bias is
   built into a 16 KiB scratch and discarded before the next head, so
   only one bias matrix is live at a time instead of the full 8-head
   256 KiB matrix (the parallel override still builds the full matrix).

### Measured (this host)

Same-process alternating A/B (median of four alternating pairs,
release build):

| Component | Baseline path | New path | Speedup |
|---|---:|---:|---:|
| Attention (indirect vs inlined) | 15.034 ms | 15.155 ms (run A) / 13.765 vs 14.052 (run B) | ~1.02x (run-to-run: 1.00-1.04x) |
| Softmax exp (scalar vs poly) | 14.988 ms | 14.693 ms | 1.025x |
| Buffers (fresh per-layer vs reused) | 15.227 ms | 14.947 ms | 1.02x |

Full-exit wall time, alternating cross-build A/B against the
pre-round commit (median of three interleaved 100-call runs each):
14.48 ms -> 14.41 ms. That is within this shared-host's run-to-run
noise, so the honest statement is: **the components are each
individually 1.02-1.04x faster, and the full-exit wall time did not
get measurably worse; it also did not get measurably better, on this
host.**

Why the wall gain is small here: the full-exit cost on this host is
not compute-bound. The isolated int16xi8 dot microkernel runs at
0.07 cycles/MAC (93% of the madd throughput ceiling); inside the
forward pass the QKV section runs at ~4 cycles/MAC. The 1.3 MiB
shared L2 cannot hold the qkv weights (196 KiB per layer), the
activation array, and q/k/v at once, so the weight streams thrash
L1/L2. That is a host property (the reviewer's Core Ultra 9 285H has
36 MiB L2), and the two structural attempts to attack it directly
(output-blocked QKV with an i32 tile, and a token-streaming
two-output kernel) both **regressed** this host in A/B (1.19x and
1.16x slower respectively) -- recorded under Rejected experiments.

### The usability lever: `UnarchitecturedHintExit`

The trial harness had hard-coded the **shallowest, worst-calibrated**
exit (2/128: top-1 0.185 on the 600-position corpus, see
`docs/unarchitectured-metal-calibration.md`). A new default-off-safe UCI
option now exposes the cost/quality frontier without changing default
behavior:

```
setoption name UnarchitecturedHintExit value 4/192   # 2/128 | 4/192 | 8/256
```

The default is `2/128` (current behavior, byte-for-byte the same
HintKey and submission path); unknown values are rejected with the
previous value kept. The exit is part of the `HintKey`, so cached
results from different exits can never be served to each other (new
test: `unarchitectured_hint_exit_option_selects_exit`). Measured on
this host: 2/128 = 2.04 ms, 4/192 = 5.14 ms, 8/256 = 13.6 ms.
4/192 costs 2.7x less than the full exit while calibrating better than
the 2/128 the harness used to force (top-1 0.195 vs 0.185); 8/256
remains the quality reference (0.255). Any future SPRT batch must
state which exit it tested -- the option makes that a configuration
choice instead of a code change.

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
  forward;
- retained-int8 regret source/target projections under the existing complete-
  output drift gate;
- inlined AVX2 attention (query row loaded once per query, fused chunk
  score dots, register-resident value accumulation across all 64 keys);
- vectorized range-reduced poly exp for the attention softmax (degree-six
  Taylor on [0, ln 2), 2^n bit-reconstruction, clamp at -50);
- per-forward scratch reuse across the eight elastic blocks (no per-layer
  allocation + zeroing); and
- per-head geometric-bias streaming (one 64x64 bias live at a time instead
  of the full 8-head matrix).

`UNCHESSED_INFERENCE_THREADS` can override the default, which is **1
(sequential)**, not the visible CPU count. Every parallel path here spawns
fresh OS threads per call rather than reusing a pool, and measured directly
on real hardware (Core Ultra 9 285H) that per-call spawn/join cost is a
monotonic net loss at every thread count tried — 1 thread measured 7.92ms,
rising to 16.42ms at 8 threads. The earlier `available_parallelism().min(4)`
default (11.01ms on that same host) was already worse than doing no internal
splitting; it has been corrected. The override remains available for a
different host, or once a persistent worker pool replaces the current
spawn-per-call design. This is deliberately independent of search threads;
the model is not wired into search yet.

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

### Output-blocked QKV with an i32 tile (round 11)

Processing 8 outputs at a time through a (3x8x64) i32 tile kept the
3x8 weight rows L1-resident across all token blocks and coalesced the
stores through the tile. It was **1.19x slower** than the unblocked form
on this host (same-process alternating A/B, real package weights): the
tile transpose and the extra i32 traffic cost more than the weight
re-reads they save when the L2 is only 1.3 MiB. The unblocked
token-major form was restored.

### Token-streaming two-output kernel (round 11)

A per-token kernel with the two weight rows resident in registers
(versus the four-token microkernel that re-hoists them per call)
regressed the FFN-up section 1.16x in the pilot before it was rolled
back. The four-token register layout the compiler already produces
(12 accumulators + 3 weight vectors = 15 zmm) is at the register limit;
going below four tokens per call buys no register headroom, and going
above it does not fit.

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

Normal UCI play still has no neural integration because the candidate option is
default-off. The earlier synchronous full-forward hint lost 0-20 because
inference consumed the move clock.

Round four added only a default-unreachable trial layer:

- a bounded nonblocking worker whose results require an exact position/persona/
  legal-action/exit key;
- a root-hint search entry point that keeps every legal move, uses policy only
  on the first pass, and charges synchronous preprocessing to the same deadline;
- an eight-position fixture-disjoint calibration smoke;
- three precharged `movetime 75` integrated depth/NPS trials; and
- adversarial mate, only-move, and stale-key tests.

The smoke corpus lacks training-membership provenance and uses depth-4 HCE
labels; round 6 superseded it with a 600-position over-the-board corpus labelled
by Stockfish 17.1 (see `docs/unarchitectured-metal-calibration.md`), which found
real but modest policy signal — full-exit top-1 0.255 against a 0.050 random
baseline and a 0.157 static-heuristic baseline. The sandbox is not identified
deployment hardware. One of eight best moves differed in every time-limited
trial. A real default-off UCI candidate now
constructs the exact-key worker and routes the main search through
`go_with_root_hints` only when explicitly enabled. It submits only on eligible
large-clock/fixed-budget `go` commands; short clocks neither submit nor wait.
All candidate preprocessing is charged to main and helper deadlines.

Round 7 closed the missing dependencies on real deployment hardware: the
deployment-CPU benchmark, the integrated depth/time calibration, and three real
SPRT batches. Those results are recorded in
`docs/unarchitectured-metal-integration-trial.md`; the short version is that no
configuration ever trended positive, the aggressive config measured -26.1 then
-15.1 Elo across 1,200 games, and the shipped conservative config measured
-5.8 Elo (statistically neutral) across 300.

`runtime_safety_suite` remains false, and round 8 closes the last *named* gap in
it rather than the flag itself. Every prior root-hint safety test fed a
hand-built adversarial ranking; none used the real checkpoint's own opinion.
`tools/find_unarchitectured_metal_hint_disagreements.py` now searches the exported
package for positions where its policy disagrees with the verifiably correct
move, and found five of six — including a forced back-rank mate the model ranks
**10th of 17**, and the Greek-gift sacrifice it ranks **18th of 38** while
preferring quiet castling. Two of those are now Rust safety tests driven by the
recorded logits, with a Python test asserting the transcribed numbers still
match the committed artifact.

The flag stays false because it is not a documentation checkbox: flipping it
implies the runtime is cleared for integration, and the SPRT evidence says the
opposite. See `docs/unarchitectured-metal-integration-trial.md` and
`docs/unarchitectured-metal-calibration.md`.

Round 11 added the `UnarchitecturedHintExit` option (default `2/128`,
i.e. behavior unchanged; `4/192` and `8/256` selectable, unknown values
rejected). The exit is part of the `HintKey`, so results computed for one
exit are never served for another. This exists because the trial harness
had hard-coded the shallowest exit, which also happens to be the
worst-calibrated one (top-1 0.185 vs 0.195 at 4/192 and 0.255 at 8/256 on
the 600-position corpus): any SPRT batch that enables the candidate must
now state -- and can actually vary -- which exit it tested. The option
changes no default behavior and does not enable the hint; the candidate
stays default-off and any real use still requires the SPRT gate.

## Remaining performance work

- ~~evaluate calibrated int8 activations~~ **done, and rejected.** Five
  calibration schemes (per-tensor static, per-channel, per-group, percentile
  clipping, and the previously-rejected per-token symmetric) were simulated
  against the real checkpoint: all fail the `5e-3` gate by 5-14x. A
  mixed-precision split passes on the two frozen fixtures at 44% of MACs in
  int8, but that assignment is overfit -- it exceeds the gate on 80 of 150
  unseen corpus positions. Forcing it to generalise shrinks int8 coverage to
  ~11% of MACs and *still* fails 2-3 of 150. Root cause is the missing weight
  clipping in training, so this is retrain-gated, not kernel-gated. See
  `docs/int8-activation-calibration-finding.md`;
- ~~evaluate VNNI/dot-product kernels for the existing int16 activation
  path~~ **evaluated (round 3), rejected**: `vpdpwssd` regressed the real
  deployment host from ~16.1/12.8ms to ~23.4/19.1ms at one/two threads
  (frequency throttling outweighed the wider dot); round 11 confirmed the
  AVX2 madd microkernel is already at 93% of its throughput ceiling in
  isolation, so there is no unthrottled VNNI headroom left to find;
- prepack/transcode matrices for wider deployment microkernels (moot for
  the int16 path while the round-3 VNNI rejection stands; the f32
  fallback and the scalar path still use the package layout directly);
- stop materializing duplicate f32 copies for matrices once all fallback and
  non-x86 deployment constraints are settled;
- measure whether the persistent outer inference worker reduces latency on the
  actual deployment UCI host (inner matrix scopes remain);
- extend exact-key caching only if real repeated-position hit rates justify it;
  and
- keep asynchronous root hints default-off until deployment calibration,
  tactical/depth gates, and paired-game SPRT all pass; and
- re-measure the round-11 components (inlined attention, poly softmax,
  scratch reuse, per-head bias) on the actual deployment host: this
  sandbox's 1.3 MiB shared L2 makes the full exit cache-bound in a way the
  285H's 36 MiB L2 is not, so the component A/Bs here are lower-bound
  evidence, and the full-exit wall gain may be larger (or smaller) there.

Per-node use remains unrealistic for this 4.22M-parameter transformer without a
much smaller distilled evaluator.
