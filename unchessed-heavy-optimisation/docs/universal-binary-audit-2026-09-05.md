# Universal-binary audit — 2026-09-05

## Decision

No new hand-written dispatch layer was added for search or transposition-table code. The evidence supports retaining the existing portable/x86-64-v3 build split until a profile identifies a specific non-NNUE function that benefits materially from a multiversioned implementation.

## Verified source facts

The SHA-256 of `unchessed-core/src/nnue.rs` is identical in the main checkout and `manus/research-facilities`: `604bb76e9b87414597666432257088a1aba566282f80c2a03b4e8f307d60c93f`. The file already performs cached runtime dispatch for AVX2+FMA through `OnceLock` and uses target-feature-gated kernels for NNUE accumulator updates and output dot products. This is already a single-binary universal-runtime design for the NNUE hot path.

A repository search found no BMI2, PEXT, or PDEP implementation in the checked-in core source. The move generator uses multiply-based magic bitboards. Therefore, the x86-64-v3 baseline is not currently unlocking a BMI2-specific move-generation kernel.

The remaining build difference is most plausibly diffuse LLVM code generation from `-C target-cpu=x86-64-v3`, not one obvious function that can be cheaply swapped behind a runtime branch. Rust stable does not provide a general, mature `target_clones` equivalent for transparently multiversioning the whole binary. Blindly adding dispatch would add branch and maintenance cost without evidence of a gain.

## Benchmark verification

The repository’s checked-in explicit-NNUE benchmark file is `benchmarks/results/portable-v3-20260905-034215.tsv`. Its rows record portable NPS above the v3 build for this stored run:

| Hash | Portable mean NPS | v3 mean NPS | v3 delta |
|---:|---:|---:|---:|
| 4 MiB | 1,008,514 | 986,055 | -2.23% |
| 8 MiB | 997,321 | 972,362 | -2.50% |
| 16 MiB | 1,006,226 | 973,763 | -3.23% |
| 32 MiB | 965,325 | 941,834 | -2.43% |
| 64 MiB | 942,466 | 911,536 | -3.28% |

This does not agree with a separate 2.44% v3 advantage reported in an earlier narrative. The stored TSV is the stronger local evidence because it contains per-position measurements, but its evaluator path points to `/home/ubuntu/unchessed-research/unchessed-heavy-optimisation/unchessed-nnue.bin`, which is not present in this checkout.

A fresh rerun with the available `artifacts/xt-nnue-v1-full.pt.best` file produced zero NPS rows because that file is a PyTorch checkpoint, not an `UNCHNNUE` runtime weights file. The engine explicitly reported `NNUE load failed: not an UNCHNNUE weights file` and used the hand-crafted evaluator. Those zero rows were rejected and are not treated as performance evidence.

## Actionable next step

The next valid experiment is to provide a canonical `UNCHNNUE` runtime file, rerun the portable-v3 script from this exact commit, and use Linux `perf record/report` or an equivalent profiler on the same binary and workload. Only if one non-NNUE function accounts for a repeatable portion of the gap should it receive a runtime-dispatched kernel. Candidate areas are the negamax loop and transposition-table probe/store path, but neither should be changed before profiling.

The attachment’s central conclusion is therefore accepted: NNUE already implements the universal-binary pattern correctly, BMI2 is not currently used, and the build split should not be replaced by speculative dispatch. The audit also corrects the benchmark interpretation: the available local TSV currently shows portable ahead of v3, while the previously stated v3 advantage remains unverified until the original evaluator asset and environment are reproduced.

## Implemented architecture

The branch now contains `unchessed-core/src/cpu.rs`, a centralized `OnceLock`-cached capability registry. It exposes `has_avx2()`, `has_avx2_fma()`, and `has_avx2_sse41()` predicates. NNUE and Aegis no longer perform feature detection independently; their existing target-feature kernels are selected through this shared interface, while scalar implementations remain the fallback. The refactor changes dispatch ownership, not numerical behavior or the weights ABI.

This is intentionally a **capability registry**, not a promise that every v3 instruction is available. Each kernel asks for exactly the features required by its `#[target_feature]` declaration. AVX2 integer kernels require AVX2; FMA kernels require AVX2+FMA; the softmax kernel requires AVX2+SSE4.1. The resulting binary remains portable, and unsupported hardware never enters an unsafe kernel.

The implementation is deliberately limited to existing verified kernels. No unprofiled search or transposition-table SIMD implementation was added.
