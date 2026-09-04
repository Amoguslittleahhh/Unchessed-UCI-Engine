# 41 — AVX2 unsafe safety audit

## Executive conclusion

**Recommendation: no source change is warranted by this audit.** The AVX2 unsafe functions in `unchessed-core/src/nnue.rs` are private, compiled only on x86/x86-64 where applicable, and their production call sites are guarded by runtime feature detection. The primary dot-product dispatch checks both `avx2` and `fma`; the accumulator row dispatch checks `avx2` and `fma`; and the only AVX2+SSE4.1 softmax/attention code found in the repository is outside `eval.rs` and is independently guarded. `eval.rs` contains no AVX2 unsafe block, target-feature attribute, or runtime feature gate (the only `unsafe` match there is prose: “unsafe to sit on”).

The audit found no confirmed unsupported-instruction path. The main residual assumption is ordinary Rust unsafe-contract hygiene: the AVX2 kernels use raw-pointer arithmetic and unaligned SIMD loads, so their slice-length and aliasing preconditions are guaranteed by current Rust slice/reference construction and by loop bounds, but the repository does not document every kernel's preconditions with a dedicated `SAFETY` comment. That is a documentation/maintainability opportunity, not a demonstrated safety defect. No code was changed.

## Scope and method

I inspected every occurrence of `unsafe`, `avx2`, `target_feature`, and `is_x86_feature_detected` in `unchessed-core/src/nnue.rs` and `unchessed-core/src/eval.rs`, read the surrounding implementations and tests, checked the repository status, queried the local CPU and Rust compiler, attempted the real workspace build/test path, and ran a real UCI position smoke using the existing release executable. No Tier 2/3 work, training, match campaign, default change, commit, or push was started.

## Source audit

### Feature detection and dispatch

`nnue.rs:181–193` defines `have_avx2()`. On x86-64 it returns the conjunction of `std::is_x86_feature_detected!("avx2")` and `std::is_x86_feature_detected!("fma")`; on non-x86-64 it returns false. This is a conservative gate for the kernels annotated with `#[target_feature(enable = "avx2,fma")]`. The row kernels only require AVX2, but using the stronger common gate is safe (it selects the scalar path on AVX2-without-FMA hardware rather than executing an FMA kernel).

The production call sites are:

| Location | Unsafe function | Gate | Contract observations |
|---|---|---|---|
| `nnue.rs:247–260` | `add_row_avx2` | `have_avx2()` | Row is formed from `ft_w[idx * ACC..(idx+1)*ACC]`; kernel loops only over `min(acc.len(), row.len())`. |
| `nnue.rs:262–274` | `sub_row_avx2` | `have_avx2()` | Same bounds and feature gate. |
| `nnue.rs:341–354` | `screlu_dot_avx2` | `have_avx2()` | Dispatch comment explicitly says runtime AVX2+FMA guard. |
| `nnue.rs:356–369` | `crelu_dot_avx2` | `have_avx2()` | Same. |
| `nnue.rs:811–818` (test) | `add_row_avx2` | `have_avx2()` | Test gate mirrors production. |
| `nnue.rs:830–838` (test) | `sub_row_avx2` | `have_avx2()` | Test gate mirrors production. |

The `screlu_dot_avx2` and `crelu_dot_avx2` functions calculate `n = acc.len().min(w.len())`; vector loads occur only while `i + 8 <= n`, and scalar tail loads occur only while `i < n`. Thus each `a.add(i)` and `p.add(i)` is within its source slice. `add_row_avx2`/`sub_row_avx2` use the same pattern. `_mm256_loadu_ps` and `_mm256_storeu_ps` do not require 32-byte alignment. The functions receive shared slices or a mutable accumulator plus a separate shared row; Rust's references establish the required non-null, valid access and non-overlapping mutable/shared-reference contract at the call boundary.

### AVX2 target-feature functions

The complete x86-64 AVX2 kernel inventory is:

* `add_row_avx2` (`nnue.rs:210–225`), `#[target_feature(enable = "avx2")]`;
* `sub_row_avx2` (`nnue.rs:229–245`), `#[target_feature(enable = "avx2")]`;
* `hsum256` (`nnue.rs:276–287`), `#[target_feature(enable = "avx2")]`;
* `screlu_dot_avx2` (`nnue.rs:290–313`), `#[target_feature(enable = "avx2,fma")]`;
* `crelu_dot_avx2` (`nnue.rs:315–339`), `#[target_feature(enable = "avx2,fma")]`.

`hsum256` is private and called only from the two FMA-gated dot kernels. The AVX2 intrinsics used by the functions are therefore not reachable through a public API without passing through the guarded dispatch. The direct unsafe calls in the SIMD unit test are also protected by `have_avx2()` on x86-64; non-x86 builds compile the scalar test branches instead.

The source also contains other AVX2 target-feature code at `nnue.rs:1501` (`exp8_softmax`, requiring `avx2,sse4.1`) and `nnue.rs:1647` (`attention_heads_inlined`, requiring `avx2,fma`). Their dispatches are independently gated at `nnue.rs:1558–1564` by `avx2` plus `sse4.1`, and at `nnue.rs:1727–1731` by `avx2` plus `fma`, respectively. These are included here because the request was to audit every AVX2 unsafe block and feature gate in `nnue.rs`, even though they are not part of the small accumulator/dot section. The direct `exp8_softmax` unit-test call is within the module test and is not a production path; on the local x86-64 host its required features are present.

### `eval.rs`

No AVX2 unsafe block, `#[target_feature]`, or `is_x86_feature_detected!` occurrence exists in `unchessed-core/src/eval.rs`. The grep hit at line 149 is the comment phrase “unsafe to sit on,” not an unsafe operation. Therefore there is no eval.rs feature gate to validate and no AVX2 instruction path in that file.

## Real local evidence

### Host facts

Commands run:

```text
rustc --version --verbose
rustc 1.75.0 (82e1608df 2023-12-21)
host: x86_64-unknown-linux-gnu
LLVM version: 17.0.6

lscpu | grep -E 'Architecture|Model name|Flags|AVX|FMA'
Architecture: x86_64
Model name: Intel(R) Xeon(R) Processor @ 2.50GHz
Flags: ... fma ... sse4_1 ... avx ... avx2 ... avx512f ...

rustc /tmp/cpu_detect.rs -o /tmp/cpu_detect && /tmp/cpu_detect
avx2=true fma=true sse4.1=true avx=true
```

**Verified:** this execution environment reports x86-64 with AVX2, FMA, AVX, and SSE4.1 through both `/proc`-backed `lscpu` flags and Rust's runtime detection macro. **Not verified:** behavior on an AVX2-missing host; no such host or emulator was available, so the scalar fallback was source-audited but not executed here.

### Build and test attempts

The first real commands were:

```text
cargo test --workspace
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`

cargo build --release -p unchessed-core
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`
```

To distinguish the lockfile/toolchain blocker from source compilation, I temporarily moved the lockfile outside the repository (restored it with a shell trap; `git status` confirmed no lockfile change), then ran `cargo test --workspace` and `cargo build --release -p unchessed-adapter`. Both reached source compilation but failed on unrelated Rust-version compatibility errors:

```text
error[E0599]: no method named `is_multiple_of` found for type `usize` in the current scope
 --> unchessed-core/src/unarchitectured_v1.rs:118:24
error[E0599]: no method named `is_multiple_of` found for type `usize` in the current scope
 --> unchessed-core/src/aegis_v4_runtime.rs:3541:35
```

The installed compiler is Rust 1.75.0, while these methods require a newer compiler. Consequently, a fresh workspace compile and test pass **could not be verified**. I did not edit source or the lockfile to work around this blocker.

### Real UCI smoke

A pre-existing `target/release/unchessed-adapter` executable was available. I ran:

```text
printf '%s\n' 'uci' 'isready' 'position startpos' 'go depth 1' 'quit' | timeout 30s target/release/unchessed-adapter
```

The process returned a legal result without a panic:

```text
id name Unchessed Game Adapter 0.2.3
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
readyok
info string [Unchessed] book: English Four Knights (A29) [main] — opponent ~1500, playing the popular stuff
bestmove c2c4
```

This is a real engine/position smoke, but it **does not exercise NNUE AVX2 inference** because no NNUE file was found and the existing binary predates the blocked rebuild. It is therefore supporting process evidence, not an AVX2 pass. An attempted contrived king-only FEN exposed an unrelated existing `movegen.rs:367` index-out-of-bounds panic; it was not used as an NNUE safety result. The legal `startpos` smoke did not reproduce that panic.

The repository already contains a SIMD parity unit test (`nnue.rs:798` onward) that checks elementwise add/sub bit identity and bounded dot-product agreement, but the test could not run because of the unrelated compiler errors above.

## Safety assessment

| Property | Status | Evidence / caveat |
|---|---|---|
| AVX2 kernels are isolated behind target-feature attributes | **Verified** | Complete inventory above. |
| Production AVX2 dispatch has runtime feature detection | **Verified by source** | `have_avx2()` checks AVX2 and FMA; softmax and attention have their own conjunctions. |
| FMA kernels require FMA as well as AVX2 | **Verified** | Attribute and gate both specify/check `avx2,fma`. |
| Softmax requires SSE4.1 and gate checks it | **Verified** | `exp8_softmax` attribute and dispatcher agree. |
| Raw-pointer SIMD loads stay within slices | **Verified by loop proof/source inspection** | `min` length and `i + 8 <= n` vector condition; scalar tail condition. |
| SIMD loads require alignment | **Verified** | `_loadu`/`_storeu` intrinsics are unaligned variants. |
| Unsupported-host behavior was executed | **Not verified** | No non-AVX2 host/emulation available. Scalar fallback is present by source. |
| Fresh workspace compile/tests pass | **Negative result** | Rust 1.75 cannot parse lockfile v4; lockfile-isolated build then fails on `is_multiple_of` in unrelated files. |
| NNUE AVX2 evaluation smoke passed | **Not verified** | Existing binary used hand-crafted eval because no NNUE file was found; rebuild was blocked. |
| Confirmed safety defect | **None found** | No code change recommended. |

## Assumptions and residual risks

The conclusion assumes normal Rust compiler/code-generation semantics for `#[target_feature]` and that no build configuration globally enables AVX2/FMA for code outside these functions. A release build with `-C target-cpu=native` could make unrelated generated code host-specific, but that would be a deployment/build-policy issue rather than an unsafe dispatch defect in these functions. The audit also assumes valid chess positions contain the expected accumulator and weight dimensions; the kernels defensively use the shorter slice length, while the row slice construction itself will panic rather than create an invalid slice if an NNUE index is malformed.

For maintainability, future edits should retain or expand the existing `// SAFETY:` comments to state both halves of the contract: (1) runtime feature support matches every instruction required by the target-feature annotation, and (2) loop bounds make each raw-pointer access valid. This is recommended documentation hardening only; adding comments is not required to declare the current implementation unsafe.

## References

1. [Rust RFC 2045: `target_feature` / runtime feature detection](https://rust-lang.github.io/rfcs/2045-target-feature.html). It explains that `#[target_feature]` permits code generation assuming the feature and that safe wrappers require runtime detection before calling such functions.
2. [Rust standard library `is_x86_feature_detected!` documentation](https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html). It documents supported feature names including `avx2`, `fma`, and `sse4.1`, and states that runtime detection currently relies mostly on CPUID.
3. [Rust `std::arch` x86/x86-64 intrinsics documentation](https://doc.rust-lang.org/stable/core/arch/x86_64/index.html). This is the standard intrinsic API surface used by the kernels.

## Final disposition

**Verified source result:** all production AVX2 target-feature calls in `nnue.rs` have compatible runtime gates, and all inspected SIMD loops have defensible bounds. **Negative results:** `eval.rs` has no AVX2 path; fresh tests/builds are blocked by the installed Rust 1.75 toolchain and unrelated `is_multiple_of` errors; a real UCI smoke did not reach NNUE because the available binary had no NNUE file. **Recommendation:** preserve the implementation and defaults, do not change code for this item, and rerun the existing workspace tests plus an NNUE-loaded start-position smoke under a Rust toolchain compatible with the lockfile before treating the audit as empirically complete.
