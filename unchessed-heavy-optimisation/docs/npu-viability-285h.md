# Would the NPU help?

The Core Ultra 9 285H has an Intel AI Boost NPU (~13 TOPS INT8, ~36 platform
TOPS including CPU+iGPU). Assessment for this engine.

**Short answer: not for the NNUE (worse than the GPU case, by a lot). There is
exactly one place it genuinely fits — the Unarchitectured v1 transformer — but
that feature is currently blocked on *value*, not on cost, so the NPU does not
unblock anything today.**

## Case 1 — NNUE evaluation on the NPU: no

| | per eval |
|---|---:|
| CPU NNUE (int16, AVX2) | ~20 ns |
| NPU dispatch (optimistic estimate) | ~1,000,000 ns |
| NPU dispatch (**measured**, see below) | ~300,000-600,000 ns |
| | **~15,000-30,000x worse (measured), ~50,000x (original estimate)** |

This is worse than the GPU verdict (250x) for the same structural reason plus a
new one:

1. **Alpha-beta is sequential and cannot batch.** You cannot fill an NPU batch
   with positions you have not decided to search yet — deciding is what
   alpha-beta *does*.
2. **NPUs are throughput devices with high per-invocation overhead.** They are
   built for "run this whole model on this frame, 30 times a second", not "run a
   512-wide dot product, ten million times a second".
3. **NPUs want static shapes and a compile step.** Reported OpenVINO NPU model
   load/compile times run to *tens of seconds to minutes* (one report: 96 s vs
   4.7 s on CPU). Fine once per session; irrelevant to a per-node call.

A measured data point worth noting: independent benchmarks of *small* models on
Intel NPUs frequently find the **CPU is simply faster** (~7.5 vs ~6.8 words/s in
one Qwen-1.5B test), because the NPU's advantage is perf-per-watt, not latency.
For a 256-wide integer dot product the CPU wins outright — that is precisely
what AVX2/AVX-VNNI is for.

## Case 2 — the Unarchitectured v1 transformer: the one real fit

This is a genuinely different workload, and on paper it is *NPU-shaped*:

| Property | Value |
|---|---|
| Call frequency | **once per move**, not per node — dispatch overhead amortizes |
| Precision | **already int8-quantized** — the NPU's native format |
| Shapes | **static** (legal actions padded to 218) — what NPUs require |
| Structure | dense matmul + attention — the mapped operator set |
| Cost today | 13.1 ms on the 2-core sandbox; **12.67 ms measured** on this real 285H (default threading) — the earlier "~3-4 ms est. on 6 P-cores" guess was wrong |

**The "~3-4 ms on 6 P-cores" estimate did not survive contact with the real
chip.** Measured directly on this machine (`cargo test -p unchessed-core
--release benchmark_forward_pass/benchmark_exit_ladder -- --ignored
--nocapture`): full 8/256 exit is **12.67 ms at the default thread count**,
**14.94 ms at `UNCHESSED_INFERENCE_THREADS=6`**, and **18.36 ms at
`UNCHESSED_INFERENCE_THREADS=16`** — more inference threads made it *slower*,
not faster, on this real hardware (thread-spawn/join overhead per call
outweighing the parallelism gain at this problem size). The estimate assumed
threading would scale the sandbox's 2-core number down; on the real chip it
doesn't, and the earlier scaling assumption should be treated as wrong until
someone tunes `UNCHESSED_INFERENCE_THREADS` properly (a lower value than
either tried here, or restructuring to reuse threads across calls instead of
spawning per call, is the more likely fix than more cores).

And there are two real arguments for offloading it:

**Contention.** With `Threads = 16` every core is searching. Running a ~12.7 ms
transformer on a CPU thread steals a slice of search throughput for its
duration — a bigger slice than the earlier ~3-4 ms estimate implied, now that
the real number is measured. On the NPU it steals ~0.

**Thermals — a laptop-specific effect.** The 285H shares a 45 W TDP across 16
cores. The NPU delivers ~13 TOPS at ~2-5 W. Offloading frees thermal headroom so
the P-cores sustain higher clocks. On a desktop this would be a rounding error;
on a 45 W laptop it is real.

## Measured on real hardware

This machine is a real Core Ultra 9 285H with a working `Intel(R) AI Boost`
NPU (confirmed present via `Get-PnpDevice`, `Status: OK`) — the exact chip
every earlier round of this doc estimated from vendor/literature sources
because its own sandbox had no NPU. `tools/npu_dispatch_benchmark.py` installs
OpenVINO, enumerates real devices, and times synchronous `infer_request.infer`
calls on CPU vs NPU for two stand-in models (`--calls 200`, 20-call warmup,
after JIT/compile warmup so this is steady-state, not cold-start):

```text
available devices: CPU, GPU, NPU
CPU:  Intel(R) Core(TM) Ultra 9 285H
NPU:  Intel(R) AI Boost
```

| Model | Params (approx) | CPU median | NPU median | NPU/CPU |
|---|---:|---:|---:|---:|
| `tiny_256x1` (single linear layer, NNUE output-layer shape) | 256 | 0.020 ms | 0.395 ms | ~20x slower |
| `student_8x256` (8x 256x256 matmul+ReLU) | ~0.5M | 0.047 ms | 0.498 ms | ~11x slower |
| `student_64x256` (64x 256x256 matmul+ReLU, sized to roughly match Unarchitectured v1's ~4.2M runtime params) | ~4.2M | 0.388 ms | 0.580 ms | ~1.5x slower |

**This confirms Case 1's verdict with real data, and sharpens it: the real
measured NPU dispatch cost (~0.3-0.6ms) is *lower* than the ~1ms literature
estimate, but CPU still wins by 11-20x at NNUE/small-model scale** — the
literature estimate wasn't pessimistic enough about the estimate's absolute
size, but it was directionally right, and the real number changes nothing
about the ~50,000x-vs-~20,000x conclusion at NNUE scale (still catastrophic
either way).

**For Case 2, the gap narrows sharply as scale grows toward
Unarchitectured-v1-sized compute** (~20x slower at 256 params down to ~1.5x
slower at ~4.2M params) — consistent with "NPU has high fixed per-invocation
overhead, but its throughput advantage grows with workload size." This is
still not a CPU win *for* the NPU: even at the model's real parameter count,
CPU remained faster in this stand-in benchmark. But the trend is the right
shape for the doc's existing argument that Case 2 is "technically a good fit"
once the feature has value worth accelerating — it just isn't there yet.

**Caveat on the stand-in models, stated plainly:** `student_64x256` is 64
flat `matmul→ReLU` blocks, not Unarchitectured v1's real architecture
(attention, per-layer GAB bias, LoRA policy heads, multiple output heads).
Real compute is structured very differently and the real model has a real
memory-access pattern this doesn't reproduce, so this bounds the dispatch-vs-
compute tradeoff question rather than answering "would the real model be
faster on the NPU" directly. Exporting the actual PyTorch
`UnarchitecturedV1Student` to ONNX/OpenVINO IR and benchmarking that directly
would be the real answer, and remains undone — a natural next step if Case 2
ever clears its actual blocker (calibration/SPRT, not cost).

## Why this still doesn't justify the work

**The blocker on that feature was never inference cost.** Round 6 calibration
(600 provenance-disjoint OTB positions, Stockfish 17.1 teacher) measured:

| Ordering | top-1 |
|---|---:|
| Full 8/256 exit | **0.255** |
| Random legal move | 0.050 |
| Free MVV-LVA heuristic | 0.157 |

Real signal (p = 1.8e-9 vs the heuristic), but it is a **move-ordering prior**,
and p90 centipawn loss is 422. **Making a weak hint cheaper does not make it
strong.** The open questions are still calibration quality, integrated
depth/NPS, and an SPRT — none of which an NPU answers.

There is also a hard engineering cost: NPU access means **OpenVINO or
DirectML** — a large vendor-specific dependency, an ONNX/IR export path for
`aegis_v4_runtime.rs`, per-vendor drivers, and a silent-CPU-fallback failure
mode that is a well-documented footgun. For a dependency-free Rust engine that
is a significant architectural commitment for a default-off feature whose value
is unproven.

## Verdict

| Use | Verdict |
|---|---|
| NNUE eval per node | **No.** ~50,000x dispatch overhead; cannot batch. |
| Unarchitectured v1 per move | **Technically a good fit**, but the feature is blocked on value, not cost. Revisit only if it passes an SPRT on CPU first. |
| Training / offline labelling | Use the **iGPU or a real GPU**, not the NPU. |

**Recommended: ignore the NPU.** The correct sequencing is to prove the
transformer earns its place *on the CPU* — where it now measures 12.67 ms/move
on this chip (worse than the earlier ~3-4 ms guess, and worth its own tuning
pass on `UNCHESSED_INFERENCE_THREADS` before touching the NPU question at
all) — and only then treat the NPU as an optimization. Doing it in the other
order means paying a large dependency cost to accelerate something that may
not be worth running at all.

If the ordering hint ever *does* pass an SPRT, revisit this note: at that point
the contention and thermal arguments become real, and the model's int8/static
shape profile means the port would be relatively clean.

## Status flag

NPU dispatch is recorded as an **experimental, unimplemented** capability in
`config/unarchitectured_v1_runtime_capabilities.json`:

```json
"npu_dispatch": false,
"npu_dispatch_status": "experimental-unimplemented"
```

There is **no NPU code path anywhere in this repository** — inference is
CPU-only. The flag exists so the state is explicit rather than merely absent,
and it is *enforced*, not decorative: `tools/unarchitectured_v1_runtime_readiness.py`
treats it as an experimental capability that must stay false, and readiness
fails closed if it is ever flipped on without an implementation behind it —
even when every other required capability is true. Two regression tests in
`tools/test_unarchitectured_v1_runtime_readiness.py` lock that in.

If NPU work is ever started, it must:

1. keep the flag false until a real implementation and a paired-game SPRT exist;
2. remain default-off, like every other neural candidate in this tree; and
3. treat **silent CPU fallback as a failure**, not graceful degradation — a
   documented footgun with vendor NPU runtimes, and one that would otherwise
   make a "working" NPU path indistinguishable from a broken one.

## Caveats

- **The dispatch-overhead figure is now measured on real 285H NPU hardware**
  (see "Measured on real hardware" above) — this superseded the original
  ~1ms literature estimate, which turned out to be a reasonable order-of-
  magnitude guess (real: ~0.3-0.6ms) that did not change the Case 1 verdict.
- The Case 2 forward-pass cost is now directly measured on this real 285H
  (12.67 ms, see above) rather than estimated — the earlier ~3-4 ms guess
  was wrong by roughly 3-4x, and more inference threads made it worse, not
  better, on the real chip. The dispatch-latency numbers ("Measured on real
  hardware" above) still used stand-in flat-MLP models rather than the real
  architecture, so those specifically remain a bound on the question, not a
  direct answer for the real model's NPU cost.
- Platform TOPS figures (13 NPU / 36 total) are vendor numbers and are marketing
  peak, not achievable throughput on this workload.

## See also

- `docs/performance-ceiling-and-gpu-viability.md` — the GPU analysis this
  parallels, and the 2.67x Amdahl ceiling on evaluator work generally.
- `docs/unarchitectured-v1-calibration.md` — the top-1 0.255 measurement.
- `docs/tuning-core-ultra-9-285h-and-low-end.md` — AVX-VNNI, which is the
  *right* way to accelerate int8 work on this chip.
