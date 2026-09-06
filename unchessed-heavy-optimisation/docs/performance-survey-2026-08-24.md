# Codebase performance survey — 2026-08-24

Read-only survey of `main` and the research branch
(`arena/01a02efe-unchessed-uci-engine`), looking for bugs and speed
improvements that do **not** cost depth or move quality.

## Scope and honesty notes

- **`main` and the research branch contain byte-identical Rust.** The only
  difference is this round's Python calibration tooling, docs, and artifacts
  (`git diff main HEAD --stat` = 14 files, all Python/docs/artifacts). So every
  finding below applies equally to both branches.
- **No Rust toolchain is reachable from this sandbox.** crates.io, rustup, and
  the Debian/mirror archives are all blocked, so *nothing here was compiled or
  benchmarked in-tree*. These are static-analysis findings.
- The timings quoted are **standalone C microbenchmarks** of the same
  operation shape, compiled with `gcc -O2` on this 2-vCPU Xeon. They size the
  opportunity; they are **not** measurements of this engine and will not
  transfer 1:1. Each item still needs a real `cargo` benchmark and, where it
  can change move selection, an SPRT.
- Findings are ordered by expected payoff. Items 1–4 are pure speed with
  **identical node semantics** (same tree, same moves) — those are the safe
  ones. Items 5–8 can change the search tree and therefore need SPRT.

---

## 1. NNUE evaluation is entirely scalar — no SIMD (biggest win)

`unchessed-core/src/nnue.rs`

`combine()` (line ~374) runs a 512-iteration scalar loop with an `screlu` call
per element, and it is invoked from `eval_with_state` at essentially every
search node. `add_row`/`sub_row` (lines 138/146) are likewise scalar loops over
256 floats, run on every accumulator update.

This is the single largest miss in the codebase. `unarchitectured_metal_runtime.rs` got five
rounds of AVX2/FMA work and is now heavily vectorized — but the NNUE, which is
the evaluator that *actually runs in every real game*, never received any of it.

Microbenchmark of the `combine` shape (ACC=256, SCReLU, 2 perspectives):

| Implementation | Time | Speedup |
|---|---:|---:|
| Scalar (current shape) | 649.9 ns | 1.00x |
| AVX2 + FMA | 66.1 ns | **9.84x** |

Both produced identical output (`-0.3226`).

**Why it's safe:** clamp/square/multiply-accumulate vectorizes exactly. The only
difference is float summation order, which is why the change must reuse the
existing `nnue.rs` tolerance tests (there is already a 1cp-tolerance test that
documents f32 summation-order noise). Runtime `is_x86_feature_detected!`
dispatch with the current scalar code as fallback keeps non-x86 correct — the
same pattern `unarchitectured_metal_runtime.rs` already uses, so there is a working in-repo
model to copy.

**This is the highest-value item and it does not change the search tree at all.**

## 2. Repetition detection rescans the entire game history at every node

`unchessed-core/src/search.rs:280`

```rust
fn is_repetition(&self, hash: u64) -> bool {
    self.path.iter().rev().any(|&h| h == hash)
}
```

`path` is seeded with the **whole game history** (`path: history.to_vec()`,
line 856) and grows with search depth. At move 40 of a game that is ~80 entries
before the search even starts, and every negamax node walks all of them.

Two independent problems:

**a) It ignores the halfmove clock.** A repetition is impossible across any
irreversible move (capture or pawn push). The scan can stop after
`pos.halfmove` entries. In the common case (`halfmove` small, history long)
that turns an ~100-element scan into a ~8-element one.

**b) It checks every ply instead of every other ply.** A position can only
repeat with the same side to move, so the scan should step by 2.

| Scan (path=100, halfmove=8) | Time |
|---|---:|
| Current: full reverse scan | 56.6 ns/node |
| Capped at halfmove, stride 2 | 4.2 ns/node |
| | **13.4x** |

**Why it's safe:** this is strictly *more* correct, not a heuristic. Positions
beyond the halfmove clock cannot legally be repetitions, and same-side-to-move
stepping is what the rule actually says. Node counts and move choices are
unchanged; only wasted comparisons disappear.

Worth also noting: `path` being a `Vec<u64>` means a heap indirection on a very
hot path. A fixed array plus length would remove that too.

## 3. Futility pruning runs *after* the expensive accumulator update

`unchessed-core/src/search.rs:627-660`

The move loop currently does, in this order:

```rust
let next = pos.make(m);
if !king_safe_after(&next, us) { continue; }
legal_count += 1;
let child_state = self.eval.update_state(...);   // full NNUE accumulator update
self.eval_states[ply + 1] = child_state;         // 2KB copy
let gives_check = in_check(&next);
...
if /* futility conditions */ { ...; continue; }   // <-- discards all of the above
```

Every futility-pruned move pays for a complete NNUE accumulator update and a
2KB `EvalState` copy, then throws the result away. Futility pruning fires on a
large fraction of late quiet moves at shallow depth, so this is a lot of wasted
work in exactly the highest-node-count part of the tree.

The futility test depends only on `static_eval`, `depth`, `alpha`, `is_cap`,
`m.is_promo()`, `legal_count`, and `gives_check`. Of those only `gives_check`
needs `next` — and it needs `in_check(&next)`, not the accumulator. So the
update can simply be moved below the futility block.

**Why it's safe:** pure reordering. Identical pruning decisions, identical tree.

Related: `EvalState` is 2KB (`[[f32; 256]; 2]`) and `update_state` returns it
**by value**, so it is copied into `eval_states[ply + 1]`. Writing in place
would avoid a 2KB memcpy per node.

## 4. `MoveList::new()` zero-fills 512 bytes per node

`unchessed-core/src/movegen.rs:480`

```rust
pub fn new() -> MoveList { MoveList { moves: [Move::NONE; 256], len: 0 } }
```

The array is fully overwritten by `generate()` up to `len` and never read
beyond it, so the zero-fill is dead work. The same applies to the
`let mut scores = [0i32; 256];` arrays in `negamax`, `qsearch`, and the ProbCut
block (1KB each), which are likewise only read below `len`.

| Operation | Time |
|---|---:|
| Zero-init 256-entry move array | 15.5 ns/node |
| Length-only reset | 2.2 ns/node |
| `scores[256]` zero-fill | 14.0 ns/node |

Roughly 25-30 ns/node combined, on top of items 2 and 3. `qsearch` allocates
both a `MoveList` and a `scores` array and is the most-visited node type.

**Why it's safe:** no semantic change, provided the arrays are only ever read
below `len` (they are). `MaybeUninit` is the fully general fix, but simply
reusing per-ply scratch buffers owned by `Searcher` avoids `unsafe` entirely and
also improves cache locality.

## 5. SEE is recomputed from scratch for every capture, including two pin scans

`unchessed-core/src/see.rs:46`

`move_score` calls `see()` for every capture and promotion. Each `see()` call
runs `pinned_blockers()` **twice** (once per color, lines 72-73) — each of which
does two slider-attack lookups from the king plus a loop over snipers — before
the exchange loop even begins.

Two observations:

- **The pin scan is position-wide, not move-specific.** Both results depend only
  on `pos`, so for a node scoring 8 captures they are computed 8 times
  identically. Hoisting them to the caller and passing them in would cut that to
  once per node.
- **Most captures never need full SEE.** The standard approach is a cheap
  MVV-LVA/threshold pre-filter, falling back to full SEE only for the ambiguous
  cases. Stockfish's `see_ge` is a threshold test rather than an exact value for
  exactly this reason.

**Caution:** hoisting the pin computation is semantically neutral, but replacing
exact SEE with a threshold test **changes move ordering** and therefore the
tree. The hoist is safe; the pre-filter needs an SPRT.

## 6. ProbCut searches losing captures

`unchessed-core/src/search.rs:548-590`

The ProbCut loop generates captures and orders them by `move_score`, but never
filters on the SEE sign it just computed. `move_score` returns
`-1_000_000 + sc` for material-losing captures, so those values are present and
simply unused — the loop happily runs a reduced-depth search on a capture that
SEE already says loses material.

A capture that loses material is very unlikely to beat `beta + 200`, so these
searches are near-pure waste. Skipping entries whose score is below
`-1_000_000` (the same test `qsearch` already applies at line 388) would be a
one-line change reusing an already-computed value.

**Caution:** this removes searches, so it can change when ProbCut fires →
SPRT required. Low risk, but it is a tree change, not a free win.

## 7. Redundant `attacked()` computation per move

`unchessed-core/src/search.rs:628,635` and `movegen.rs:389-397`

Per move the engine calls `king_safe_after(&next, us)` (an `attacked()` call on
the mover's king) and then `in_check(&next)` (an `attacked()` call on the
opponent's king). Then the child node immediately recomputes `in_check(pos)` at
line 469 — the *same* query that the parent already answered as `gives_check`.

Threading the known `gives_check` into the recursive call (as an argument, the
way `is_pv`/`allow_null` already are) removes one full `attacked()` call per
visited node.

**Why it's safe:** it passes down a value that is already computed and provably
equal — `in_check(&next)` at the parent is by definition the child's
`in_check(pos)`. No behavioral change.

## 8. No transposition-table prefetch

`unchessed-core/src/tt.rs`

There is no `prefetch` anywhere in the codebase. The TT is sized in megabytes,
so `probe()` is a near-guaranteed cache miss, and it happens right at the top of
each node after the make.

Issuing `_mm_prefetch` on the child's TT slot immediately after `pos.make(m)`
(the hash is available at that moment) overlaps the memory latency with the
`king_safe_after`/`update_state` work that follows.

**Why it's safe:** a prefetch is a pure hint with no architectural effect.

---

## Correctness observations (not performance)

These are not bugs today, but they are fragile:

- **`qsearch` has no TT probe or store.** It is the most-visited node type and
  currently re-derives everything. Adding a TT probe there is a common
  optimization, but it interacts with the mate-score encoding, so it needs care
  and an SPRT rather than a quick patch.
- **`is_repetition` returns true on the *first* repetition.** That is the usual
  search-tree convention (treat any repetition in the tree as a draw) and is
  intentional, but note it differs from the actual threefold rule; combined with
  the halfmove fix in item 2, the semantics stay the same.
- **`scores` arrays are `[i32; 256]` while `MoveList` holds 256 moves.** These
  are consistent, but nothing statically prevents `generate()` from exceeding
  256 in an extreme position. The theoretical maximum for legal chess is 218
  (which `unarchitectured_metal_runtime.rs` uses as `MAX_ACTIONS`), so 256 is safe — worth a
  `debug_assert!` to keep it that way.

## Suggested order of work

1. **NNUE SIMD** (item 1) — by far the largest win, no tree change.
2. **Repetition scan** (item 2) — large, trivial, strictly more correct.
3. **Futility/accumulator reordering + in-place `EvalState`** (item 3) — pure
   reordering.
4. **Scratch buffers for `MoveList`/`scores`** (item 4) — no semantic change.
5. **Hoist SEE pin scans** (item 5, first half) — no semantic change.
6. Then, each behind its own SPRT: ProbCut SEE filter (6), `gives_check`
   threading (7, though this one is provably neutral), TT prefetch (8).

Items 1-5 are all "same tree, same moves, less work per node", which is exactly
the "faster without sacrificing depth or quality" target — they should raise NPS
and therefore depth at a fixed time control, rather than trading anything away.
