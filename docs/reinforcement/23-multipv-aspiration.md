# 23 — MultiPV aspiration-window interaction

**Investigation ID:** `multipv-aspiration`
**Tier:** 1 (cheap research/design)
**Status:** Research and design only. No implementation, default change, benchmark campaign, match, or SPRT was performed.

## Executive conclusion

The repository does **not** appear to have an immediately demonstrated correctness bug in the MultiPV/aspiration combination: each requested line is eventually re-searched when its result falls outside its window, and the completed result is recorded only after the line has an in-window result. However, the implementation has a real **time-budget and intermediate-output starvation risk**. A narrow window is applied independently to every PV slot, including secondary slots, and a fail-low secondary search can trigger repeated whole-root re-searches before the next line is produced. Under a stop or hard time limit, later lines may never be completed at the current depth. That is a resource-allocation weakness, not proof that a secondary move is permanently omitted from a completed, uninterrupted search.

**Recommendation: defer any search change.** First add cheap instrumentation/tests that distinguish exact completed iterations from aborted partial iterations and quantify aspiration re-searches by PV index. No safe code change is justified from source inspection alone. In particular, do not simply disable aspiration globally for MultiPV or widen every secondary window: either can impose a large node/time cost and requires fixed-position safety evidence followed by the project’s real paired-game gate if it changes game-search behavior.

## What the literature says

Aspiration windows center an alpha-beta search around a score from the preceding iterative-deepening iteration. The narrower window can increase cutoffs, but a true score outside the window requires a re-search; standard descriptions recommend widening the bound that failed, often progressively or exponentially [1]. The PVS/aspiration interaction is specifically known to require care at the root: a root fail-low can be handled by finishing the move list and restarting, or by immediately resetting the window; the tradeoff is efficiency versus learning the corrected score sooner [2]. These sources establish the mechanism and the cost of a miss, but do **not** establish that this engine’s particular MultiPV loop is unsound.

MultiPV means reporting the N best distinct root moves rather than only one PV. The UCI/Stockfish documentation describes `MultiPV` as outputting the N best lines and recommends leaving it at one for best performance [3]. Stockfish’s current implementation is useful comparative evidence: it performs a separate root search for each PV slot, preserves already searched PV lines when sorting, and applies aspiration per slot; its comments also include explicit handling for an aborted later PV so an incomplete line cannot corrupt a completed mate/TB result [4]. That is evidence that interruption and ordering are practical edge cases, not evidence that Unchessed should copy Stockfish’s policy.

## Repository inspection

The checkout is on `manus/research-facilities`. The relevant implementation is `unchessed-core/src/search.rs`:

* `SearchParams::default()` sets `aspiration_delta = 25` centipawns and `aspiration_min_depth = 4` (lines 22–78 in the inspected tree). The UCI parser clamps these options to 5–200 and 1–12 respectively (`unchessed-core/src/uci.rs`, lines 712–721).
* `go_with_root_hints` clamps requested MultiPV to the number of legal root moves and initializes one `RootMove` per legal move (`search.rs`, approximately lines 948–1000). Root hints can affect only the first-pass ordering; later passes sort by completed search scores.
* At each iterative-deepening depth, the loop runs `for pv_idx in 0..multipv`. It initializes an aspiration window around `roots[pv_idx].score`, not around a shared score or the score of the preceding selected line (`search.rs`, approximately lines 1028–1043).
* Each aspiration attempt scans all root moves except those in `chosen`. The first searched candidate gets a full PV search. Later candidates first receive a null-window search and are re-searched full-window if they beat the current alpha. A candidate that fails low is not given a new PV/authoritative score in that attempt; its existing score is reduced only for ordering (`search.rs`, approximately lines 1047–1085).
* If the best result fails low or high, the loop expands only the failing side (`window_lo` downward or `window_hi` upward), multiplying the widening amount by four. It repeats the complete unchosen-root scan until the result lies inside the window (`search.rs`, approximately lines 1087–1108).
* Once a line has an in-window result, its move is appended to `chosen` and an `InfoEvent` is emitted with `multipv = pv_idx + 1`. The completed iteration snapshot contains only the selected `chosen` moves (`search.rs`, approximately lines 1110–1130).
* The UCI layer may search more lines than it displays when adaptive persona mode is active: `multipv_search = multipv_shown.max(5)` while output still uses `multipv_shown` (`unchessed-core/src/uci.rs`, lines 1618–1623 and 1703–1731). Helper Lazy-SMP threads run single-PV searches; only the main thread’s full MultiPV result is used, while helpers warm the shared TT (`uci.rs`, lines 1651–1701). This makes timing/TT effects worth measuring, but does not itself create duplicate output lines.
* An existing unit test, `multipv_returns_distinct_moves`, searches start position to depth 6 with four PVs and checks four distinct moves and non-increasing scores (`search.rs`, approximately lines 2090–2117). It is a useful basic invariant but does not exercise narrow aspiration, fail-low/high re-searches, time aborts, or iteration completeness.

## Can secondary PV lines be starved?

### In a completed, uninterrupted iteration

The loop does not permanently reserve a candidate for a secondary line merely because its first aspiration attempt fails low. On every re-search it scans all unchosen roots again. The current line’s `best_idx` is selected from the strongest result found under the current window; if the window is too high, the lower bound is progressively reduced. Assuming the search terminates normally and the window eventually covers the true value, all unchosen candidates remain eligible and a distinct move can be selected for each slot.

There are nevertheless two correctness-sensitive details to test rather than assume. First, the first candidate is accepted unconditionally (`searched == 1`) even when it fails below alpha; this is necessary to have a provisional best result, but means the fail-low decision depends on later widening rather than a conventional exact root score. Second, failed-low candidates retain a modified/stale ordering score rather than a new exact score. That is safe only because the outer loop re-searches the complete unchosen set and the final snapshot is taken from selected lines; it should be regression-tested against a full-window oracle.

### Under a stop, node limit, or hard time limit

Starvation is plausible and observable. A secondary PV’s window may be centered on its previous slot score, which can be far above or below the current true score after search instability. A fail-low then causes a full scan and another search. With several PV slots, these costs are serial. If `s.abort` occurs inside a scan, the code breaks the aspiration loop and then breaks the entire deepening loop. It returns `completed`, which is the prior fully recorded iteration, rather than claiming the partially searched current iteration is complete. Thus later lines can be absent at the deepest attempted depth, and the current depth can produce fewer than the requested number of lines if there was no prior complete MultiPV iteration.

This behavior is conservative: it avoids publishing a partially completed iteration as if it were exact. It still means a GUI or caller can see only PV1 (or fewer lines than requested) while PV2/PV3 are spending their allocation on aspiration recovery. That is the practical meaning of “secondary PV starvation” in this implementation. It is not yet evidence of a wrong final PV ranking when the search is allowed to finish.

### Why a single shared aspiration window is not an obvious fix

The k-th PV is a constrained ranking problem: already selected moves are excluded, and the score scale for the remaining candidates can differ from the previous iteration. Sharing PV1’s window with every secondary line would likely cause more fail-lows for weaker lines, not less. Conversely, full-width secondary searches would reduce aspiration misses but can multiply node cost. A safe policy must be based on measured re-search frequency and completed-depth behavior, not intuition.

## Cheap correctness tests

The following tests are recommended before considering any code change. They are fixed-position and CPU-cheap; they do not replace a playing-strength test.

| Test | Procedure | Required invariant |
|---|---|---|
| Distinct/ranked baseline | Existing start-position depth-6, MultiPV 4 test | Exactly four legal distinct root moves; returned scores are non-increasing. |
| Aspiration-off oracle | Run the same deterministic positions with `aspiration_min_depth` above the requested depth (or an equivalent full-window test configuration), then compare with default aspiration at the same depth and clean TT. | For a completed search, same set of root moves, legal PVs, and scores within declared search nondeterminism bounds; exact mate ordering must match. |
| Narrow-window stress | Set `aspiration_delta=5`, `aspiration_min_depth=1`; run a suite containing start position, tactical positions, quiet positions, and positions whose scores change sharply between depths. | Every completed iteration emits exactly N distinct lines, lines are legal, and scores are non-increasing. Record aspiration attempts/re-searches by `pv_idx`. |
| Wide-window control | Set `aspiration_delta=200` and compare with narrow stress. | Wide control should not reveal a completed-search line omitted by the narrow configuration. Any difference is a stop-and-investigate signal, not an automatic reason to change defaults. |
| Forced fail-low/unstable score | Use a test evaluator or deterministic fixture that makes the previous score intentionally distant from the next-depth score. If adding a fixture is too invasive, use a small synthetic root-search harness around the existing loop. | The loop widens the lower side, eventually returns an in-window result, and does not select a duplicate or retain an unsearched candidate. |
| Node/time abort matrix | Run MultiPV 2/4/8 with small deterministic node limits and fixed `movetime`; collect emitted lines and final returned lines. | No illegal or duplicate moves; incomplete current iterations are not reported as complete; previous complete iteration remains internally consistent. Document that fewer than N lines under an abort is allowed or decide that the API must guarantee N lines. |
| Permutation/order check | Present the same legal root set with different initial root ordering/root hints, then compare after enough depth and no abort. | Completed results should converge to the same ranked set; policy/root ordering may affect cost but must not change the completed oracle result. |
| Lazy-SMP comparison | Run one thread and multiple threads with the same position and limits, separately from the correctness oracle. | Treat score/PV differences as a concurrency/search-budget diagnostic; do not require byte-identical results from a shared-TT parallel search. Check especially that helper TT warming does not make later PVs disappear earlier. |

Useful instrumentation fields are `depth`, `pv_idx`, attempt count, initial/final alpha and beta, fail-low/high count, root candidates searched, nodes consumed per attempt, and whether the iteration was committed or aborted. Instrumentation should be opt-in and should not alter default search behavior.

## What was actually run

The repository and existing reports `docs/reinforcement/00` through `12` were inspected, along with the current `search.rs` and `uci.rs` implementation. The focused command below was attempted:

```text
cargo test -p unchessed-core --release search::tests::multipv_returns_distinct_moves -- --nocapture
```

It did **not** compile. The installed Cargo is 1.75.0 and rejected the checkout’s lockfile before compilation:

```text
error: failed to parse lock file
Caused by: lock file version 4 requires `-Znext-lockfile-bump`
```

Therefore this report claims no test pass, no node count, no benchmark, no fixed-position comparison, and no runtime observation. The existing MultiPV test is verified by source inspection only. No implementation or lockfile modification was made.

## Decision and safe-change boundary

**Decision: defer implementation; pursue the diagnostic tests/instrumentation only if the toolchain blocker is resolved.** Source inspection supports a narrow conclusion: the uninterrupted loop appears to recover aspiration failures, while serial recovery can starve secondary lines under finite budgets. It does not support changing aspiration parameters or algorithmic policy.

A future Tier 2 candidate could be considered only after the tests above identify a reproducible failure or material allocation problem. The least risky design direction would be an explicitly conservative handling of incomplete MultiPV iterations—preserve the last complete iteration and expose clear status/telemetry—rather than silently publishing provisional secondary scores. Any change that alters the number or order of nodes searched in normal game play remains subject to the master brief’s paired-game SPRT gate. Defaults must remain unchanged until that evidence exists.

## References

[1]: https://www.chessprogramming.org/Aspiration_Windows "Chessprogramming Wiki — Aspiration Windows"
[2]: https://www.chessprogramming.org/PVS_and_Aspiration "Chessprogramming Wiki — PVS and Aspiration"
[3]: https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html "Stockfish Wiki — UCI Protocol and Commands (MultiPV)"
[4]: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp "Official Stockfish — current search.cpp MultiPV/aspiration implementation"
[5]: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/blob/manus/research-facilities/unchessed-core/src/search.rs "Unchessed-UCI-Engine — search implementation"
[6]: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/blob/manus/research-facilities/unchessed-core/src/uci.rs "Unchessed-UCI-Engine — UCI and Lazy-SMP integration"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/23-multipv-aspiration.md`

**No implementation was made.**
