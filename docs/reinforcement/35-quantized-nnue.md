# 35 — Quantized NNUE inference

**Investigation ID:** `35-quantized-nnue`
**Tier:** 1 (research and verification only)
**Repository/branch:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/research-facilities`
**Scope:** Verify the current float32 and AVX2 NNUE paths, run a real existing NNUE evaluation where feasible, and assess int8/int16 quantization tradeoffs and integration/accuracy risks. **No quantization implementation or default change was made.**

## Executive answer

The repository’s NNUE runtime is **float32 throughout**: the feature-transformer weights, biases, accumulators, output weights, and output biases are `Vec<f32>`, and the SIMD kernels use AVX2 floating-point instructions (`_mm256_add_ps`, `_mm256_sub_ps`, and floating-point reductions). The AVX2 path is runtime-gated by `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`, cached in a `OnceLock`, with scalar fallback. This was verified by source inspection and by the host’s CPU flags.

A real shipped-network UCI run succeeded locally. With `unchessed-nnue.bin`, one search position, one thread, book disabled, and a 2-second move-time budget, the engine reached depth 11 and reported **657,832 nodes, 885,372 nps, 743 ms**, selecting `e5f6`. This is a **repository-on-this-host search throughput measurement**, not an isolated NNUE-forward benchmark and not evidence of a 2–4× quantization gain. The earlier depth-8 run also loaded the net and produced a valid NNUE score (`cp 359` at the first tested position), confirming the real evaluation path rather than merely parsing the file.

Quantization is technically plausible and is the production approach used by Stockfish-style NNUE runtimes, but the project should not assume a literature speedup transfers directly. The lowest-risk design is the conventional mixed scheme: **int16 accumulator for the feature transformer, conversion to uint8/int8 after clipped activation, and int8 linear weights with int32 accumulation and int32 biases**. It would require an export ABI, calibration/training constraints, integer SIMD kernels, parity fixtures, and strength testing. The principal risk is not just rounding: clipping/saturation, scale choices, bias rescaling, accumulator overflow, changed activation semantics, and silently loading a quantized file as the current float ABI can all alter evaluations materially. Recommendation: **pursue only as a separately gated Tier 2 experiment after adding a report-only benchmark/parity harness; do not promote or change defaults based on the published 2–4× expectation.**

## Verified repository evidence

| Question | Verified finding | Evidence | Status/limit |
|---|---|---|---|
| Is inference float32? | `Nnue` stores `ft_w`, `ft_b`, `out_w`, and `out_b` as `Vec<f32>`; `read_f32s` decodes four-byte IEEE float values. | `unchessed-core/src/nnue.rs:104–120, 151–161` | **Verified.** This describes the current runtime, not every training artifact. |
| Is the accumulator float32? | Evaluation state uses `[[f32; ACC]; 2]`; row updates add/subtract `f32` values. | `unchessed-core/src/eval.rs:18–34`; `nnue.rs:195–207` | **Verified.** |
| Is there an AVX2 path? | `add_row_avx2` and `sub_row_avx2` use `_mm256_loadu_ps`, `_mm256_add_ps`/`_mm256_sub_ps`, and `_mm256_storeu_ps`; reduction kernels are also AVX2 float code. | `nnue.rs:209–245, 276 onward` | **Verified.** |
| Is AVX2 guarded? | `have_avx2()` caches `avx2 && fma` detection in `OnceLock`; callers test it before entering unsafe kernels and otherwise use scalar code. | `nnue.rs:179–193, 247–274` | **Verified.** The source includes `SAFETY` comments and length-bounded loops. |
| Is quantized inference present? | No int8/int16 NNUE runtime path was found in `nnue.rs`; the current loader reads float arrays and the shipped file is accepted by that format. | Source inspection and successful load below | **Verified negative result** for the inspected runtime. |
| Does the real network evaluate? | `unchessed-nnue.bin` loaded successfully; UCI search reported NNUE loaded and produced scores/PV/nodes. | Command/output below | **Verified.** |

The network’s feature mapping is not being re-researched here. Existing code comments identify the hot spots as accumulator updates and output combination, and the repository already has SIMD-vs-scalar unit tests. The current question is representation and arithmetic, not a feature-layout rewrite.

## Real-world tests performed

### 1. Real NNUE UCI evaluation and search

Exact command:

```bash
cd /home/ubuntu/Unchessed-UCI-Engine
{ printf 'uci\nsetoption name Threads value 1\nsetoption name OwnBook value false\nsetoption name EvalFile value /home/ubuntu/Unchessed-UCI-Engine/unchessed-nnue.bin\nisready\nposition fen r1bq1rk1/ppp2ppp/2n1pn2/3pP3/3P4/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 8\ngo movetime 2000\n'; sleep 3; printf 'quit\n'; } | target/release/unchessed-adapter
```

Relevant output (the full UCI handshake was also observed):

```text
info string [Unchessed] NNUE loaded from '/home/ubuntu/Unchessed-UCI-Engine/unchessed-nnue.bin'
readyok
info depth 1 ... score cp 359 nodes 37 nps 37000 ...
info depth 8 ... score cp 347 nodes 94585 nps 975103 ... time 97 ...
info depth 9 ... score cp 323 nodes 223329 nps 911546 ... time 245 ...
info depth 10 ... score cp 344 nodes 383911 nps 905450 ... time 424 ...
info depth 11 ... score cp 320 nodes 657832 nps 885372 ... time 743 ...
bestmove e5f6
```

This is a real search using the existing net, with `Threads=1`, `OwnBook=false`, and a fixed FEN. It demonstrates end-to-end loading, incremental evaluation during search, and output. The reported nps is **search nps**: it includes move generation, pruning, transposition-table activity, and all other engine work. It must not be presented as an NNUE kernel benchmark.

A second exact command checked a different real position at shallow depth:

```bash
printf 'uci\nsetoption name EvalFile value /home/ubuntu/Unchessed-UCI-Engine/unchessed-nnue.bin\nisready\nposition fen r1bq1rk1/ppp2ppp/2n1pn2/3pP3/3P4/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 8\ngo depth 8\nquit\n' | target/release/unchessed-adapter
```

It reported `NNUE loaded`, then `depth 1 ... score cp 359 ...`, and selected `bestmove e5f6`. This confirms that the successful path is not just a loader check.

### 2. Existing NNUE tests and a negative toolchain result

Exact command:

```bash
cd /home/ubuntu/Unchessed-UCI-Engine
cargo test -p unchessed-core nnue -- --nocapture
```

Exact result:

```text
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`
```

The installed toolchain was `rustc 1.75.0` / `cargo 1.75.0`, and `rustup` was unavailable. I did **not** modify `Cargo.lock`, install a replacement toolchain, or regenerate dependencies merely to force this test. Source inspection found an existing test named `simd_kernels_match_scalar_reference` (`nnue.rs:792–860`) and v4 incremental/capture tests (`nnue.rs:1005–1041`), but this run could not execute them because of the pre-existing Cargo/toolchain blocker. This is reported as a negative result, not treated as a pass.

### 3. Host SIMD capability

Exact command:

```bash
lscpu | grep -E 'Model name|Flags' | head -2
```

The host reported an Intel Xeon @ 2.50 GHz with `avx2`, `fma`, and AVX-512 feature flags. Therefore the runtime guard’s positive branch is available on this host. This does not by itself prove the release binary selected AVX2 on every invocation, but it verifies the required CPU features and the source dispatch condition. No source changes were made to add a dispatch counter, so AVX2 branch selection is **strongly supported by code plus host features, but not directly instrumented in the binary run**.

## What quantization would mean here

The authoritative [Stockfish NNUE documentation][1] describes the conventional scheme. The feature transformer accumulates many sparse rows, so its accumulator needs more range than int8; Stockfish uses **int16** there and converts after clipped activation. Subsequent linear layers can consume **int8/uint8 activations**, use **int8 weights**, accumulate products in **int32**, and add **int32 biases** before the next clipped activation. This is a mixed-width pipeline, not “convert every float to int8 and hope.”

The same documentation emphasizes that quantization ranges constrain model parameters. In particular, int8 linear weights have a small permitted range after accounting for activation scaling; training therefore needs parameter-range control (for example, clamping in the optimizer) so the trained float model does not diverge from the deployable integer representation. The documented design also relies on clipped activations to make the conversion meaningful and predictable.

For this repository, a plausible future ABI would therefore need at least:

1. A versioned quantized header identifying layer dimensions, scales, zero points if any, signedness, and output scale.
2. Quantized feature-transformer rows and biases with a proven accumulator range. The current f32 incremental accumulator cannot simply be reinterpreted as i16 without a quantized update path.
3. Explicit activation conversion and clipping semantics. The current float code uses its own `SCReLU`/`CReLU` functions; integer clipping must reproduce the intended trained-domain operation, including rounding and saturation.
4. int8 dot-product kernels with int32 accumulation, plus scalar fallback and runtime dispatch for AVX2/AVX-VNNI/AVX-512 where applicable.
5. A separate loader or unambiguous format tag. A float net and a quantized net must never be accepted interchangeably.
6. Position-by-position parity fixtures over start position, captures, promotions, king-bucket changes, sparse/dense material, and extreme evaluations. A tolerance must be predeclared in centipawns and tested at the final engine output, not only at one layer.

## Speed, memory, and accuracy tradeoffs

| Choice | Likely benefit | Main risk/cost | Applicability to this repository |
|---|---|---|---|
| Current float32 + AVX2 | Simple ABI, straightforward numerical behavior, existing parity tests, no calibration step | Wider loads and floating arithmetic; larger weights/accumulators; potentially lower integer dot-product throughput | **Measured and working:** 885,372 search nps in the specified one-thread run. |
| int16 feature transformer + int8 later layers | Matches the established Stockfish-style NNUE design; smaller model and accumulator bandwidth; efficient integer SIMD | Scale selection, accumulator overflow, conversion/clipping errors, new serialization and kernels | Most credible first quantized design, but requires a separate experiment. |
| int8 feature-transformer accumulator | Smaller still and potentially high SIMD density | Unsafe range for sums of many feature rows; severe saturation/accuracy risk unless architecture/scales change | **Not recommended** as an initial path. |
| int16 weights in later layers | More range/precision and less weight quantization error | Twice the weight bandwidth versus int8; may give up much of the integer throughput advantage | A useful accuracy fallback, not the conventional fast target. |
| int8 weights + int32 accumulation | High SIMD density and safe product accumulation when ranges are bounded | Very limited weight range; training must account for it; output rescaling and bias handling are delicate | Candidate target only with quantization-aware export/training and parity gates. |

The project brief mentions a “2–4x inference speedup” associated with int8/int16 Stockfish practice. That figure is a **literature/project-context expectation**, not a measurement made in this repository. The only new local throughput number here is whole-search nps with the current float32 runtime. No quantized implementation exists in this branch, so a local speedup comparison is impossible without violating the “no implementation” scope. Hardware matters substantially: AVX2 versus AVX-VNNI/AVX-512, compiler flags, cache state, network dimensions, accumulator refresh frequency, and search mix can all change the result. Search nps cannot be converted into a kernel speedup without an isolated fixed-position evaluator benchmark.

Memory savings are more certain directionally: replacing four-byte float weights with one-byte weights can reduce weight bandwidth, while int16 accumulators are half the width of f32 accumulators. But the exact saving depends on whether all layers and biases are quantized, alignment/padding, duplicated perspectives, and whether a float copy remains resident. Lower memory traffic may help even where arithmetic throughput does not; it is not safe to claim a particular percentage before measuring.

Accuracy risks are also not captured by aggregate MAE alone. Quantization can produce small average errors but larger errors in rare material/king buckets or positions near search thresholds. Saturation can be asymmetric and can alter move ordering, pruning, aspiration-window behavior, and tactical stability. A one-centipawn average parity gate is not sufficient if tails or sign flips matter. Conversely, a small evaluation difference need not reduce Elo; only a matched game test can establish playing-strength impact.

## Recommendation and decision gate

**Recommendation: defer implementation, but keep quantized inference as a justified future performance experiment.** The current runtime is a verified float32/AVX2 implementation and already performs real NNUE search at approximately 0.89 million nodes/s in the specified local run. The conventional mixed int16/int8 design is technically well understood and likely the least risky route, but no repository-local speed or strength claim is available yet.

Before any Tier 2 coding, authorize only a small benchmark/parity plan: first fix the Cargo/toolchain blocker or provide a compatible isolated test environment; then add no-default-change instrumentation that reports isolated NNUE evaluations/s and full-search nps on fixed positions. A quantized prototype should be accepted for further testing only if it passes exact loader/shape checks, no accumulator overflow, bounded per-position and tail error against float32, incremental-versus-refresh parity, and a fixed-position speed comparison. Only after those gates should a short, same-hardware game match decide whether any speed gain survives search behavior and whether Elo is preserved.

Do **not** import the Stockfish speed figure as a local result, do **not** replace the shipped network format, and do **not** change defaults. The strongest verified conclusion today is narrower: **the repository has a real float32 AVX2 NNUE path, no quantized path, and a measurable baseline from which a future quantized experiment can be judged.**

## References

[1]: [Official Stockfish NNUE documentation — Quantization](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) (describes int16 feature-transformer accumulation, int8 linear weights/activations, int32 accumulation/biases, clipping, and training range constraints).
[2]: [Stockfish, “Introducing NNUE Evaluation”](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/) (official historical description of CPU-efficient NNUE and the need for empirical engine testing; its Elo results are Stockfish results, not Unchessed results).
[3]: [Stockfish feature-transformer source](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_feature_transformer.h) (production integer feature-transformer implementation and SIMD conversion/clipping details).

## Status

**Source and defaults unchanged. No quantization implementation, training run, cloud spend, Tier 2/Tier 3 work, commit, or push was performed.** Existing real-world testing was performed where feasible; the existing Rust NNUE test suite was blocked by Cargo.lock version/toolchain incompatibility and is recorded above.

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/35-quantized-nnue.md`
