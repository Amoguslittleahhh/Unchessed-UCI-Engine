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
| NPU dispatch (optimistic) | ~1,000,000 ns |
| | **~50,000x worse** |

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
| Cost today | 13.1 ms measured on the 2-core sandbox; **~3-4 ms est.** on 6 P-cores |

And there are two real arguments for offloading it:

**Contention.** With `Threads = 16` every core is searching. Running a ~3-4 ms
transformer on a CPU thread steals ~1/16 of search throughput for its duration.
On the NPU it steals ~0.

**Thermals — a laptop-specific effect.** The 285H shares a 45 W TDP across 16
cores. The NPU delivers ~13 TOPS at ~2-5 W. Offloading frees thermal headroom so
the P-cores sustain higher clocks. On a desktop this would be a rounding error;
on a 45 W laptop it is real.

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
transformer earns its place *on the CPU* — where it costs ~3-4 ms/move on your
chip, which is affordable at long time controls — and only then treat the NPU as
an optimization. Doing it in the other order means paying a large dependency
cost to accelerate something that may not be worth running at all.

If the ordering hint ever *does* pass an SPRT, revisit this note: at that point
the contention and thermal arguments become real, and the model's int8/static
shape profile means the port would be relatively clean.

## Caveats

- **No NPU is present in this sandbox**, so the ~1 ms dispatch figure is a
  literature-based estimate, not measured. It would have to be wrong by ~4
  orders of magnitude to change the Case 1 verdict.
- The ~3-4 ms 285H estimate scales a 2-core sandbox measurement by core count
  and IPC. It is an estimate, not a measurement — you have the hardware to
  check it directly once the round-1 work compiles.
- Platform TOPS figures (13 NPU / 36 total) are vendor numbers and are marketing
  peak, not achievable throughput on this workload.

## See also

- `docs/performance-ceiling-and-gpu-viability.md` — the GPU analysis this
  parallels, and the 2.67x Amdahl ceiling on evaluator work generally.
- `docs/unarchitectured-v1-calibration.md` — the top-1 0.255 measurement.
- `docs/tuning-core-ultra-9-285h-and-low-end.md` — AVX-VNNI, which is the
  *right* way to accelerate int8 work on this chip.
