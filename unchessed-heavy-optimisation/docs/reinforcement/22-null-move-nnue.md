# 22 — NNUE accumulator correctness under null-move pruning and Lazy SMP

**Investigation ID:** `22-null-move-nnue`
**Tier:** 1 (cheap correctness research)
**Repository/branch:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/research-facilities`
**Status:** Source audit and parity-test review complete; no implementation change made.

## Question

A null move changes only the side to move (and clears en-passant state). In a mover-perspective NNUE, the accumulator belonging to the new side to move must become the first output half, while the other perspective becomes the second. The investigation asked whether this swap is correct in the presence of incremental accumulators, null-move pruning, and the shared infrastructure used by Lazy SMP threads.

## Executive finding

**No confirmed bug was found.** The implementation stores two accumulators by absolute color, not by “side-to-move” slot. A null move therefore correctly carries the accumulator state unchanged and relies on the child position’s flipped `side` to select the halves in the opposite order. This is equivalent to swapping the two input halves without mutating the stored state.

The search state is also thread-local: every `go_with_root_hints` call constructs its own `Searcher` and its own `eval_states` vector. Lazy SMP shares the immutable evaluator and the lock-free transposition table, but not accumulator arrays. The evaluator is required to be `Send + Sync`, and the NNUE implementation has no mutable accumulator cache or global mutable inference state. The shared TT uses atomics and validates key/data coherence on probe; a raced replacement can reduce cache quality but does not alias NNUE state.

Recommendation: **defer/drop as a bug fix; preserve the current design.** Add a focused null-move accumulator regression test in a future correctness-only maintenance pass, but do not alter runtime behavior or defaults on the basis of this review. No SPRT is applicable because no candidate behavior was changed.

## Repository evidence

| Claim | Evidence | Interpretation |
|---|---|---|
| State is indexed by absolute perspective | `unchessed-core/src/eval.rs:12–19` documents “one accumulator per absolute perspective (White, Black), indexed by `Color::idx()`”; `NnueEvalState` is `acc[[f32; ACC]; 2]`. | The state itself is not ordered as STM/NSTM, so a null move need not physically swap arrays. |
| Fresh state fills both absolute-color halves | `unchessed-core/src/nnue.rs:642–648` computes `accumulate(pos, White)` into index White and `accumulate(pos, Black)` into index Black. | Both perspectives are available before and after a null move. |
| Evaluation selects STM/NSTM at the call site | `unchessed-core/src/nnue.rs:684–691` sets `acc_stm = state.nnue.acc[pos.side.idx()]` and `acc_nstm = state.nnue.acc[pos.side.flip().idx()]`, then calls `combine`. | Flipping `pos.side` automatically reverses the halves presented to the output head. |
| Output-head ordering is STM first | `unchessed-core/src/nnue.rs:595–613` combines `acc_stm` with the first output slice and `acc_nstm` with the second (and documents this file format at lines 30–33). | The runtime matches the mover-perspective contract. |
| Null move changes no piece features | `unchessed-core/src/board.rs:402–412` copies the position, removes EP hash/state, increments halfmove, flips `side`, and toggles the side hash; no bitboard or mailbox piece changes occur. | The feature-derived accumulators remain valid; EP is not an NNUE piece feature in this implementation. |
| Null branch preserves state intentionally | `unchessed-core/src/search.rs:559–590`, especially lines 570–577, explicitly says accumulators are unaffected and assigns `eval_states[ply + 1] = eval_states[ply]` before recursing on `make_null()`. | This is the correct operation for absolute-color accumulator storage. |
| Ordinary moves update both absolute perspectives | `unchessed-core/src/nnue.rs:651–680` copies the two arrays, then applies the before/after bitboard diff separately for White and Black; an own-king move refreshes that perspective. | A later real move after a null move starts from the correct state and updates both sides normally. |
| Search state is per searcher/thread | `unchessed-core/src/search.rs:988–1008` constructs a fresh `Searcher` and `eval_states: vec![eval.initial_state(pos); MAX_PLY + 1]` for every call. | Helpers do not share accumulator stacks or ply slots. |
| Lazy SMP shares only intended infrastructure | `unchessed-core/src/uci.rs:1656–1728` passes shared `eval`, `TT`, limits, history, and stop flag to scoped helper threads; each helper calls the search entry point, which creates its own state vector. | No cross-thread accumulator mutation is visible. |
| Evaluator sharing is constrained | `unchessed-core/src/eval.rs:40` requires `Eval: Send + Sync`; the NNUE object contains weight vectors and configuration, with no per-search mutable state. | Concurrent calls are structurally safe for the evaluator. |
| TT race is separate from NNUE state | `unchessed-core/src/tt.rs` uses atomic fields, probes by checking `xor_key ^ data == hash`, and documents that a torn read is treated as a miss. | Shared TT can cause misses/suboptimal replacement, not accumulator corruption or a wrong STM swap. |

## Side-to-move parity reasoning

Let `A_W` and `A_B` be the two entries in `EvalState`. At a position `P` with White to move, `eval_with_state(P)` passes `(A_W, A_B)` to `combine`. `make_null(P)` produces `P'` with the identical piece placement and Black to move; the null branch copies the same state, so `eval_with_state(P')` passes `(A_B, A_W)`. That is exactly the required STM/NSTM reversal.

The inverse case is identical: if Black is to move, the order is `(A_B, A_W)`, and after null it is `(A_W, A_B)`. No accumulator contents need to be exchanged because the arrays are explicitly absolute-color indexed. This also avoids a common error mode in implementations that store `[STM, NSTM]` in the state and then copy it across a null move without swapping.

The same conclusion applies to v1 and v3 incremental networks. `update_state` changes only rows corresponding to bitboard differences and rebuilds a perspective when that perspective’s king moves. Since `make_null` produces no bitboard difference, bypassing `update_state` is semantically equivalent to an update with an empty feature diff, while `eval_with_state` still observes the changed side.

## Tests and parity evidence

The repository contains strong ordinary-move accumulator parity tests in `unchessed-core/src/nnue.rs:1236–1359`:

* `incremental_accumulators_match_full_refresh_for_special_moves` covers quiet moves, captures, en passant, both castlings, promotions, and king movement for v1 and v3.
* `incremental_accumulators_match_full_refresh_over_move_tree` carries state over a 20-ply opening sequence containing captures and castling and compares every incremental state with a full refresh.

Those tests establish the absolute-perspective update machinery, but **there is no dedicated null-move state/evaluation parity test in the inspected test block**. The ideal cheap regression would construct a deterministic dummy v1/v3 net, evaluate a position with a fresh state, make a null move, copy the state as the search does, and assert that `eval_with_state(null_position, copied_state)` equals `eval(null_position)` (within the existing integer/float tolerance). It should also assert that the copied arrays are byte-for-byte unchanged and exercise both initial colors. This is a test recommendation only; no source implementation was changed.

The requested Cargo tests could not be executed in this sandbox. The checked-in `Cargo.lock` is version 4, while the available toolchain is `rustc 1.75.0`/`cargo 1.75.0`; Cargo terminated before compilation with `lock file version 4 requires -Znext-lockfile-bump`. No lockfile edit or toolchain workaround was applied. The report therefore distinguishes source proof and existing checked-in tests from newly executed tests: **new test execution: not run due to toolchain/lockfile incompatibility; source audit: completed.**

The three master-brief `unarchitectured_metal_runtime` parity gates are unrelated to this NNUE/search path and were not modified. Existing documentation records that those gates must remain passing for any changes to that runtime; no such change was made here.

## Authoritative external evidence

The official [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) describes two separate accumulators, one per perspective, and explicitly gives the mover-perspective combination rule: place the accumulator for the side to move first and the other second. Its pseudocode uses `pos.accumulator[stm]` followed by `pos.accumulator[!stm]`, which supports the parity reasoning above. The same page explains that both perspectives are maintained separately and that king moves require refreshes.

Stockfish’s current [NNUE accumulator header](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_accumulator.h) likewise defines accumulation indexed by `COLOR_NB` and describes per-thread accumulator caches. That is relevant as an authoritative design comparison for the shared-infrastructure risk: caches are associated with a thread-local accumulator facility, not a single mutable global search accumulator. Unchessed’s simpler implementation is even more isolated because its `Searcher` owns the entire ply-indexed state vector.

These sources establish the general NNUE design claim, not a claim that Unchessed is source-identical to Stockfish. The repository-specific conclusion comes from the Rust control flow and data layout cited above.

## Verified versus assumed

| Category | Status |
|---|---|
| Absolute-color accumulator indexing | **Verified by source**. |
| STM/NSTM selection after side flip | **Verified by source** in `eval_with_state`. |
| Null move leaves feature rows unchanged | **Verified by `Position::make_null` source**. |
| Null branch copies state unchanged | **Verified by source**. |
| Per-thread state isolation in Lazy SMP | **Verified by source**. |
| Shared TT cannot corrupt accumulator arrays | **Verified structurally**: no shared accumulator reference exists; TT races are handled as atomic key/data misses. Formal concurrent stress execution was not run. |
| Dedicated null-move parity test | **Absent in inspected repository tests**. |
| Fresh Cargo test result for this investigation | **Not run**, blocked before compilation by Cargo 1.75 versus lockfile v4. |
| Strength impact or Elo | **Not measured and not applicable**; no implementation candidate was produced. |

## Recommendation and next gate

**Drop the bug-fix hypothesis and defer implementation.** The current representation correctly handles null moves by retaining absolute-color accumulators and selecting them according to the child position’s side to move. Lazy SMP does not introduce a shared accumulator race: state is allocated inside each search invocation, while only the immutable evaluator, stop flag, and atomic TT are shared.

A future maintenance change may add the focused null-move parity test described above once a compatible Rust toolchain is available. Such a test would improve regression coverage, not justify a runtime change. If future refactoring changes the state layout from absolute-color to STM/NSTM indexing, the null branch must explicitly swap or rebind the halves, and the parity test should become a mandatory gate. No default flip, search tuning, retraining, game match, or SPRT is warranted by this negative finding.

## References

[1]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html "Official Stockfish NNUE documentation"
[2]: https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_accumulator.h "Stockfish NNUE accumulator header"
[3]: ../reinforcement/00-synthesis.md "Existing reinforcement synthesis and gates"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/22-null-move-nnue.md`
