# Performance work — implementation round 1

Implements the safe items from `docs/performance-survey-2026-08-24.md`.

## Verification status — read this first

**No Rust toolchain is reachable from this sandbox.** crates.io, rustup, the
Debian archives, and every mirror tried are blocked, so `cargo build` and
`cargo test` **were not run on this change**. That is a real gap, and it is the
single most important caveat here.

What was done instead:

- **Brace/paren/bracket balance** verified on every touched file with a checker
  that strips comments, string literals, raw strings, and char literals first
  (so it is not fooled by braces inside text).
- **Algorithmic equivalence proved by porting the exact new and old logic to C
  and differential-testing them** (see numbers below). This validates the
  *algorithms*, not the Rust.
- **Call sites audited** for signature changes (`is_repetition`, `move_score`,
  `SearchParams`) across all four crates.

The changes still need `cargo test --workspace --release` before they can be
trusted, and the one tree-changing item is default-off. Treat the timings as
opportunity sizing, not as engine measurements.

## What changed

### 1. NNUE SIMD (`nnue.rs`) — the big one

The NNUE had no SIMD at all, despite being the evaluator that runs in every
real game, while `unarchitectured_metal_runtime.rs` (unwired, default-off) had five rounds of
AVX2 tuning. Added AVX2+FMA kernels with runtime dispatch and the existing
scalar code retained as fallback and as the behavioral reference:

- `add_row` / `sub_row` — accumulator updates, run on every make;
- `screlu_dot` / `crelu_dot` — the output layer, run at nearly every node.

Dispatch resolves once through a `OnceLock<bool>`, matching the pattern already
used in `unarchitectured_metal_runtime.rs`. Non-x86 keeps the scalar path.

Differential test (2000 random vectors, C port of both implementations):

| Kernel | Max deviation from scalar |
|---|---:|
| `add_row` / `sub_row` | **0.0 (bit-exact)** |
| `screlu_dot` / `crelu_dot` | 7.6e-6 |

The dot products reduce, so they differ only in float summation order — the
same class of noise the existing 1cp-tolerance eval tests already document. The
elementwise kernels are bit-exact, as they must be.

### 2. Halfmove-bounded repetition detection (`search.rs`)

`is_repetition` scanned the entire path — which is seeded with the whole game
history — at every node. At move 40 that is ~80 comparisons per node that
**cannot possibly match**: the halfmove clock resets on every capture and pawn
push, and those moves are irreversible, so nothing before the last one can recur.

The scan is now bounded by `pos.halfmove`. This is a strict narrowing, not a
heuristic — every skipped entry was provably non-matching, so the search tree is
unchanged.

Differential test, 200,000 randomised trials with a deliberately small hash
alphabet to force collisions: **0 semantic mismatches** — it never invents a
repetition and never misses one inside the window.

Deliberately *not* done: stepping the scan by 2 (only same-side-to-move
positions can repeat). It benchmarked no better than the halfmove bound alone at
realistic clock values, and `make_null` perturbs path parity, so the extra
reasoning risk bought nothing.

### 3. Futility pruning no longer pays for a discarded accumulator update

The move loop ran `update_state` (a full NNUE accumulator update plus a 2KB
`EvalState` write) *before* the futility test, so every futility-pruned quiet
paid for work that was then thrown away. The update now happens after the
pruning block.

Audited: nothing between the old and new position reads `eval_states[ply + 1]`,
and the futility condition depends only on `static_eval`, `depth`, `alpha`,
`legal_count` and move flags. Pure reordering — same pruning decisions, same tree.

### 4. Hoisted SEE pin scans (`see.rs`)

`see()` computed `pinned_blockers()` for **both colors** on every call, i.e.
once per capture, even though the result depends only on the position. Added
`Pins` / `see_with_pins`, plus `LazyPins` so a node that never scores a capture
does not pay for the scan at all.

`see()` keeps its original signature and behavior, so existing SEE tests are
untouched. A new test asserts `see`, `see_with_pins`, and `LazyPins` agree on
every legal move across six positions including the pinned case.

### 5. TT prefetch (`tt.rs`)

Added `TT::prefetch`, issued right after the child position is made. The table
is megabytes wide so `probe` is a near-certain cache miss; prefetching there
overlaps the latency with the legality check and accumulator update that follow.
A prefetch is a pure hint with no architectural effect. Tested for inertness.

### 6. ProbCut SEE filter — implemented but **default-off**

ProbCut ran reduced-depth verification searches on captures its own SEE score
had already flagged as material-losing. The filter reuses that already-computed
value (`break`, not `continue`, since the list is sorted best-first).

**This one changes the search tree**, so per this project's discipline it is
behind `SearchParams::probcut_see_filter`, default `false`, exposed as the UCI
option `ProbcutSeeFilter` so a baseline-vs-candidate SPRT can be run from a
single binary. Nothing changes until that gate passes.

## Measured opportunity

Modelling one node as *one output-layer evaluation + one accumulator update +
one repetition scan* (path 100, halfmove 8), C port, `gcc -O2`, 2-vCPU Xeon:

| | ns/node |
|---|---:|
| Before | 692.8 |
| After | 106.6 |
| | **6.50x** |

This is a model of the changed operations in isolation, **not** an engine NPS
measurement. Real gain will be lower — nodes also do movegen, make/unmake, and
TT work that this change does not touch. It sizes the prize; it does not claim it.

## What still needs to happen

1. `cargo build --workspace --release` and `cargo test --workspace --release` on
   a machine with a toolchain. **Nothing here has been compiled.**
2. A real NPS/depth benchmark before/after on identical positions.
3. SPRT for `ProbcutSeeFilter` before it is turned on.
4. Items deliberately left alone: the `MoveList`/`scores` zero-fill
   (item 4 of the survey — needs either `MaybeUninit` or a `Searcher`-owned
   scratch buffer refactor, more invasive than the rest), threading
   `gives_check` into the child to skip a redundant `attacked()` call, and a
   qsearch TT probe.
