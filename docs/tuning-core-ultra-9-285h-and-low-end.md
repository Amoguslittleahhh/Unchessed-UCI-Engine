# Tuning for a Core Ultra 9 285H — and for low-end machines

Two audiences, one document, because the answers interact: what to optimize for
an Arrow Lake-H laptop chip, and what to optimize so the engine stays good on
weak hardware.

## The headline: a one-line change was worth more than every kernel so far

**The engine defaulted to `Threads = 1`.** On a Core Ultra 9 285H — 16 cores —
that used roughly **6% of the CPU**. No amount of SIMD tuning recovers that.

Fixed in this commit: `Threads` now defaults to `available_parallelism()`,
capped at 32.

Illustrative Lazy SMP scaling (0.65 efficiency, *estimated not measured*):

| Threads | ~node throughput |
|---:|---:|
| 1 (old default) | 1.0x |
| 4 | ~3.0x |
| 16 (your chip) | ~10.8x |

Even discounting heavily for Lazy SMP's imperfect scaling, this is far larger
than the ~1.8x the entire evaluator roadmap can deliver. **If you change one
thing, change this** — and it costs nothing in quality, because Lazy SMP helpers
only warm a shared transposition table; the main thread still produces the
result.

Caveat: most GUIs set `Threads` explicitly, which overrides the default. This
fixes the out-of-the-box case (direct UCI use, scripts, and GUIs that respect
the advertised default).

## Your CPU vs the machine I've been measuring on

This matters, because my earlier measurements were on a CPU **unlike yours**:

| | sandbox Xeon | **Core Ultra 9 285H** |
|---|---|---|
| cores / threads | 2 (1c, SMT) | **16 / 16** |
| topology | uniform | hybrid: 6 P (Lion Cove) + 8 E + 2 LP-E (Skymont/Crestmont) |
| SMT | yes | **no** |
| AVX-512 | **yes** | **no** (fused off on Arrow Lake) |
| AVX2 / FMA | yes | yes |
| AVX-VNNI (256-bit) | — | **yes** |
| L3 | 54 MB | **24 MB** |

Four consequences:

### 1. Ignore AVX-512 entirely — and the round-1 code is already correct

My previous note measured AVX-512 as *slower* than AVX2 on the Xeon
(downclocking). On your chip the question is moot: **Arrow Lake has no
AVX-512 at all.** The round-1 kernels chose AVX2 with runtime detection, which
turns out to be exactly right for your hardware and degrades gracefully
everywhere else. No change needed.

### 2. AVX-VNNI is the interesting instruction for you

Arrow Lake ships 256-bit AVX-VNNI (`VPDPBUSD` / `VPDPWSSD`). That is a
single-instruction int8/int16 dot-product-accumulate — the exact primitive a
quantized NNUE output layer wants — and crucially it comes **without** the
AVX-512 frequency penalty that made VNNI a loss on the Xeon (documented in
`docs/unarchitectured-v1-runtime-optimization.md`).

This strengthens the int16 recommendation from
`docs/performance-ceiling-and-gpu-viability.md`: on your CPU, quantization has
first-class hardware support. It remains a **retraining** project (QAT, format
v4, SPRT) — not a runtime patch.

### 3. 24 MB L3 makes the int16 net matter *more* on your machine

The current f32 feature transformer is **23.1 MB** against a **24 MB** L3. It
essentially fills the cache by itself, leaving nothing for the transposition
table. The int16 version is **11.5 MB** and leaves real room.

On the 54 MB Xeon this was a nice-to-have. On your chip it is close to a
correctness-of-design issue for cache behavior.

### 4. Hybrid cores: don't over-engineer it

6 P + 8 E + 2 low-power E is awkward in theory (helpers land on slow cores).
In practice I'd deliberately **not** try to detect and pin P-cores:

- Lazy SMP helpers are TT-warmers, not latency-critical. A slow E-core still
  contributes useful entries.
- The OS scheduler already prefers P-cores for early-spawned threads.
- Portable userspace P/E detection is unreliable and would be a large amount of
  fragile platform code for a speculative gain.

Use all 16 and let the scheduler work. If you ever want to experiment,
`Threads 6` (P-cores only) vs `Threads 16` is a cheap SPRT to settle it
empirically rather than by argument.

## Low-end machines

The good news: **the same changes help.** The bad news: one common piece of
advice is actively harmful.

### Don't crank Hash on a weak machine

Measured TT probe cost vs table size (random access, this host):

| Hash | ns/probe |
|---:|---:|
| 1 MB | 2.5 |
| 4 MB | 2.5 |
| 16 MB | 2.5 |
| 64 MB | 7.2 |
| 256 MB | **14.9** |

A table that overflows cache costs **~6x per probe**. On a low-end box with a
small L3, setting `Hash 2048` makes the engine *slower per node*, not stronger.
The 128 MB default is sensible; low-end users should consider **32–64 MB**, not
more.

This is a documentation/guidance fix, not a code change — the right value
depends on the machine, and the option already exists.

### What actually helps low-end hardware

In priority order:

1. **Threads auto-detect** (this commit) — a 4-core budget laptop goes from 1 to
   4 threads. Same relative win as your 285H, from a lower base.
2. **int16 quantization** — halves memory traffic and the accumulator working
   set. Weak CPUs are *more* memory-bound, so they benefit *more* than a
   high-end chip does.
3. **Dirty-plane `apply_diff`** (~77 ns/move, identical arithmetic) — pure
   win, no retraining, helps every machine.
4. **Scalar fallback correctness** — the round-1 kernels keep a scalar path, so
   pre-AVX2 hardware still runs. Worth keeping and not regressing.

### What does *not* help low-end

- AVX-512 paths (absent on most consumer hardware, including yours).
- Anything GPU. Already covered in
  `docs/performance-ceiling-and-gpu-viability.md`: alpha-beta is sequential and
  cannot batch, and a ~5 µs round trip against a ~20 ns eval is 250x the wrong
  direction. Integrated GPUs are worse, not better, here.
- Bigger nets. The 4.2M Unarchitectured v1 student measured top-1 0.255 and
  costs more than it returns; a low-end machine is the last place to pay that.

## Recommended order for *your* setup

1. **`Threads` auto-detect** — done here. Biggest single win, zero quality cost.
2. **Compile and test the round-1 SIMD work.** Still never compiled — no Rust
   toolchain is reachable from this sandbox. Everything below is speculative
   until this is validated on real hardware.
3. **Dirty-plane `apply_diff`** — safe refactor, ~77 ns/move.
4. **One retrain: int16 (QAT) + 8 piece-count output buckets + format v4**, then
   SPRT. On Arrow Lake this lands on AVX-VNNI hardware and fits L3.
5. Then, and only then, the search-side items (SEE hoisting, ProbCut filter,
   `gives_check` threading, TT prefetch) — after the eval work they become the
   dominant term.

## Verification status

- The `Threads` change is a small, reviewable diff with a regression test
  asserting the default tracks `available_parallelism()`, stays within the cap,
  is never 1 on a multi-core host, and matches the value advertised over UCI.
- **Not compiled.** No Rust toolchain is reachable here (crates.io, rustup and
  the distro archives are all blocked). Balance-checked only.
- CPU specifications were taken from vendor/reviewer sources, not assumed from
  memory: 16 cores (6 P + 8 E + 2 LP-E), no SMT, no AVX-512, 24 MB L3.
- Thread-scaling Elo figures are **illustrative estimates**, not measured, and
  are labelled as such above.
