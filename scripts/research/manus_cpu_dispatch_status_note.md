# Stockfish 19's "universal binaries" idea, checked against the actual code

Stockfish 19 released 2026-09-05 with universal binaries (one download,
runtime CPU-feature detection instead of separate AVX2/AVX-512
downloads) among other changes. Worth checking whether the same idea
applies to `unchessed-heavy-optimisation`'s portable-vs-x86-64-v3 build
split before treating it as a real gap.

## The NNUE hot path already does this

`unchessed-core/src/nnue.rs` -- byte-identical between main and
`unchessed-heavy-optimisation`, confirmed earlier this investigation --
already implements exactly the universal-binary pattern for its actual
hot functions:

```rust
fn have_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    })
}
```

`add_row`/`sub_row` (accumulator update) and `screlu_dot`/`crelu_dot`
(output layer) each check `have_avx2()` once (cached via `OnceLock`)
and dispatch to a `#[target_feature]`-gated AVX2 kernel or a plain
scalar fallback. This is correct, safe, and already ships in one
binary that runs everywhere and gets the SIMD speedup automatically
when the CPU supports it -- no separate build required for this part.

## BMI2 isn't actually used anywhere

`x86-64-v3` also implies BMI2. Checked `movegen.rs` for `pext`/`pdep`
usage: there is none. The move generator uses plain multiply-based
magic bitboards, not hardware `pext`. So the BMI2 requirement in the
v3 build isn't unlocking any specific hot function either -- there's no
BMI2-specific kernel to multiversion the way the NNUE ones were.

## What the portable/v3 split is actually buying, then

Given the two known SIMD-sensitive kernels (NNUE accumulator + output
dot products) already runtime-dispatch regardless of build flags, and
there's no BMI2-gated code anywhere, the remaining NPS difference
between the portable and v3 builds (2.44% measured in your sandbox VM,
26% measured on this reviewer's WSL box -- a real, hardware-dependent
gap already on record) is diffuse general LLVM codegen improvement
across the rest of the binary under `-C target-cpu=x86-64-v3`, not
something concentrated in one swappable function.

Rust doesn't have a mature stable equivalent to GCC/Clang's
whole-function multiversioning (`target_clones`) to capture that kind
of diffuse gain generically in one binary the way the NNUE kernels do
it manually. So the two-build split isn't a gap that more hand-written
dispatch code can close cheaply -- it's a reasonable way to get that
remaining benefit given the current tooling.

**If a genuine single "universal binary" is still worth pursuing**, the
real next step is profiling (not blind dispatch-writing) to find
*which* non-NNUE code actually accounts for the measured gap -- likely
candidates are `search.rs`'s main negamax loop or `tt.rs`'s probe/store
path, both hot enough to plausibly benefit from `-C target-cpu`'s wider
default register/instruction assumptions. Once a specific function is
identified as the source of the gap, it can get the same
`have_avx2()`-style runtime-dispatch treatment nnue.rs already uses --
but that's a profile-first, not a design-first, exercise.
