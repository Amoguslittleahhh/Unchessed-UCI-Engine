# 24 — UCI `searchmoves` and `go mate N`

**Investigation ID:** `tier1-uci-searchmoves-mate`
**Tier:** 1 — protocol completeness / infrastructure
**Repository / branch:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/research-facilities`
**Status:** Research and design only. No Rust implementation, UCI behavior change, benchmark, match, SPRT, or default change was performed.

## Executive conclusion

The repository does not currently implement either requested UCI feature. `parse_go` in `unchessed-core/src/uci.rs` recognizes clocks, depth, nodes, movetime, and `infinite`, but silently ignores `searchmoves` and `mate`. The search API always constructs its root list from all legal moves. The UCI worker can also return an opening-book move before main search. Thus a GUI cannot reliably restrict analysis to specified candidates, and it cannot request a mate-targeted search with the protocol's intended termination semantics.

**Recommendation: pursue a narrowly scoped Tier 2 implementation as protocol-correctness and analysis usability work, not as a strength feature.** `searchmoves` is the lower-risk half: it should filter legal root moves before book selection, adaptive selection, root hints, Lazy SMP, and MultiPV, while leaving the underlying alpha-beta search unchanged. `go mate N` needs an explicit contract because UCI defines the request as “search for a mate in x moves,” while this engine's search depth is in plies and its mate scores are internal values. It should stop when a completed iteration proves a mate within the requested bound, while continuing when no qualifying mate is proven, subject to `stop`, clocks, movetime, nodes, and depth.

This is **Tier 2-worthy for implementation and deterministic protocol tests**, but not Tier 3-worthy by itself. A default-preserving protocol fix needs no playing-strength SPRT. Any change to ordinary search, a new tree-changing mate heuristic, a default change, or an Elo claim remains subject to the master brief's real paired-game gate.

## Protocol semantics

The canonical UCI description defines `searchmoves <move1> ... <movei>` as a `go` subcommand that restricts the search to those moves. Its example is `position startpos` followed by `go infinite searchmoves e2e4 d2d4`; only those two initial moves should be searched [1]. This is a root restriction, not a move-ordering hint: moves outside the list must not be searched or selected. The implementation must intersect requested UCI moves with legal moves in the current position; malformed, illegal, duplicate, or absent moves must never become executable roots.

The same specification defines `mate <x>` as “search for a mate in x moves” [1]. The specification separately defines `info score mate <y>` as mate in `y` moves, with negative values when the engine is being mated [1]. Therefore requested distance in full moves must remain distinct from internal ply depth and score formatting. Stockfish's current source is a useful interoperability cross-check: its parser stores `searchmoves` as a final move list and parses `mate` as a distinct limit [2]; its search loop stops a mate request only when a root mate score is within the requested distance [3]. These references confirm semantics, not a requirement to copy Stockfish internals.

## Repository inspection

`parse_go` (approximately `uci.rs:958–986`) creates `Limits` and handles `depth`, `movetime`, clocks, increments, `movestogo`, `nodes`, and `infinite`; unknown tokens fall through to `_ => {}`. There is no `searchmoves` field, no `mate` field, and no diagnostic for an ignored request. The command thread joins the previous worker before a new `go`, so a parsed request can be made an immutable `GoJob` snapshot without a mutable-options race.

`run_go` first obtains all legal moves, then performs adaptive observation and possible book selection before main search. Main and Lazy SMP helper searches call `search::go` or `go_with_root_hints`. A filter only inside the alpha-beta root loop would therefore be insufficient: book and adapter paths could still return an excluded move. The restriction must be applied at the worker boundary and passed consistently to every root-search invocation, or those paths must explicitly test membership.

`search.rs` defines `Limits` without either field. `go_with_root_hints` calls `legal(pos)`, builds a root entry for every legal move, and clamps MultiPV against that full list. Existing mate machinery is useful: `MATE`, `MATE_IN_MAX`, `is_mate_score`, `mate_in`, mate-aware aspiration bypass, and TT mate-distance normalization (`to_tt`/`from_tt`). This supports an implementation, but does not provide a mate-limit stop condition. Existing search tests include mate fixtures, but no UCI transcript tests for these commands.

## Proposed implementation scope (design only)

### Immutable parsing and validation

Extend the request/limits representation with `mate: Option<u32>` and a root restriction, preferably as raw `searchmove` tokens until the current position is available. Use a boundary such as `parse_go(line) -> GoRequest`, followed by `validate_searchmoves(position, tokens)`. Parse non-negative mate values safely; reject or explicitly define zero (rejecting `mate 0` is the safest initial contract). Missing or malformed numeric arguments must not panic or consume the next keyword.

Treat `searchmoves` as the remainder of the `go` line, matching Stockfish's parser [2]. Convert tokens through the existing `parse_uci_move`, retain only legal moves, deduplicate, and preserve requested order for deterministic diagnostics. A list with no legal intersection must **not** fall back to unrestricted search or a book move; return a documented no-move response, preferably `bestmove 0000`, or emit a clear protocol diagnostic.

### Root restriction

Add a root-filtered search entry point, retaining the current API as an unrestricted wrapper. Build roots from legal moves intersected with the allowed set. Clamp MultiPV against the filtered count. Pass the same set to main search, every Lazy SMP helper, root hints, adapter selection, and book handling. An allowed book move may be returned; an excluded book move may not. Hints must be generated or filtered over the constrained list.

Without a restriction, fresh-TT fixed-depth searches must retain the old score, PV, and node count. With a restriction, changed score/node count is expected because the root problem is different. Do not alter child legality, evaluation, pruning, TT semantics, or ordinary move ordering.

### Mate-targeted search

Do not translate `go mate N` into `depth 2N`: depth is a ply ceiling, while mate is a proof/termination target. After each completed iterative-deepening iteration, inspect the best completed requested-root line. If it has a positive mate score and `mate_in(score) <= N`, the target is satisfied and search may stop. Negative mate output should be handled only under an explicitly documented policy; the safest first version interprets the command literally as seeking a mating line for the side to move.

The stop condition must be based on a proven mate score, not a high centipawn score or guessed depth. Existing TT distance normalization and mate formatting must be reused and tested at odd/even distances. When combined with depth, nodes, movetime, clocks, or `stop`, the earliest valid condition wins. A no-qualifying-mate request with no finite bound may continue indefinitely, so negative tests must supply a finite limit or send `stop`. The final output must use the last completed trusted iteration, not a partially searched line.

Do not add a new tactical search, change mate constants, or change default search behavior in this item. Existing mate-aware aspiration behavior should remain intact.

## Test matrix

This matrix is a design gate for Tier 2 implementation; it was not run because this task forbids implementation.

| Area | Test | Expected result |
|---|---|---|
| Existing parser | `go depth 5`, nodes, clocks, movetime, `infinite` | Existing behavior and limits remain unchanged. |
| Basic restriction | Start position, `go depth 4 searchmoves e2e4 d2d4` | `bestmove` and every PV root are one of the two moves. |
| Exclusion | Exclude known unrestricted best move | Excluded move is never returned, booked, hinted, or shown as root PV. |
| Single candidate | One legal move with depth/nodes/movetime | Exactly that move is returned; search remains legal. |
| Invalid list | Illegal, malformed, wrong-position and duplicate tokens | Invalid entries never broaden search; duplicates do not inflate roots. |
| Empty intersection | No legal requested move | No unrestricted/book fallback; documented `0000`/error response. |
| Book/adapter | Exclude book move; adaptive engine enabled | Main constrained search or an allowed book move only; adapter cannot bypass set. |
| MultiPV/SMP/hints | Restricted set smaller than MultiPV; Threads > 1; hint on | All roots in all paths are identically filtered; MultiPV clamps correctly. |
| Default equivalence | Unrestricted HCE and NNUE fixed-depth, fresh TT | Same score, PV, and node count as pre-feature behavior. |
| Mate parsing | `mate 1`, `mate 2`, malformed, missing, zero, very large | Safe parsing; invalid input fails closed; no depth alias. |
| Forced mates | Existing mate-in-one and mate-track fixtures | Completed line reports `score mate 1` or lower positive distance and stops on proof. |
| Distance conversion | Mate in 1/2/3, odd and even proof plies | UCI reports moves, not plies; target compares converted mate distance. |
| No mate | Quiet position with finite depth/nodes/time plus mate | Finite boundary wins; legal move returned without false mate claim. |
| Explicit stop | Long/no-mate `go mate N`, then `stop` | Exactly one legal `bestmove`; worker joins cleanly. |
| Combined limits | Mate with depth, nodes, clocks, movetime, and `infinite` | Earliest valid stop wins; exact node limits and analysis semantics remain. |
| TT safety | Warm/fresh TT and different ply contexts | Mate distance is normalized; TT cannot falsely satisfy target. |
| Regression | Focused mate/repetition/search tests and workspace suite | Existing tests remain green; no evaluator/parity/default path changes. |

A transcript harness should also assert one `bestmove` per `go`, legal UCI formatting, no excluded move in any `info ... pv`, and clean `stop` handling. Use fresh TTs for deterministic comparisons. Distinguish “mate proven” from “legal move returned at an external finite boundary.”

## Tier decision and rollout gates

**Tier 2-worthy: pursue implementation, separately from strength work.** The scope is bounded, `searchmoves` semantics are clear, and the engine already has a root-search seam plus mate score infrastructure. Implement one isolated change with focused parser, root, book/adapter/SMP, and mate tests, then run the workspace suite on a current compatible toolchain. Do not bundle with MoveOverhead, AnalyseMode, pruning, tuning, or evaluator changes.

Acceptance requires proving that no path escapes the allowed root set and that mate termination uses completed proof distance in moves. A default-preserving protocol fix needs no SPRT. Any ordinary-search behavior change, changed default, new tree-changing heuristic, or strength claim requires the master brief's real paired-game evidence. No cloud compute or training asset is needed.

## Verification ledger

| Item | Status |
|---|---|
| Master brief | Read `/home/ubuntu/upload/pasted_content_6.txt`, including item 16 and Tier 2/Tier 3 rules. |
| Existing docs | Read reinforcement context 00–12 and inspected neighboring protocol reports 25–29. |
| Source | Inspected `unchessed-core/src/uci.rs` parser, `GoJob`, worker lifecycle, book/adaptive paths, hints, SMP, and `unchessed-core/src/search.rs` limits, root construction, mate conversion, and TT normalization. |
| External research | Read canonical UCI protocol text [1], official Stockfish parser [2], Stockfish search stopping logic [3], and Chessprogramming UCI background [4]. |
| Implementation | **Not performed**, as required. |
| Tests | **Not run as passing**. Cargo 1.75.0 cannot parse this checkout's lockfile version 4 (`requires -Znext-lockfile-bump`). |
| Strength evidence | None: no benchmark, game, SPRT, Elo claim, default change, or Tier 3 work. |

## References

[1] [Description of the Universal Chess Interface](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf), command definitions for `go searchmoves`, `go mate`, and `info score mate`.

[2] [Stockfish `src/uci.cpp`](https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/uci.cpp), current official parser for `searchmoves` and `mate` limits.

[3] [Stockfish `src/search.cpp`](https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/search.cpp), current official mate-limit stopping condition.

[4] [Chessprogramming Wiki — UCI](https://www.chessprogramming.org/UCI), protocol background.

[5] Repository source: [`unchessed-core/src/uci.rs`](../../unchessed-core/src/uci.rs) and [`unchessed-core/src/search.rs`](../../unchessed-core/src/search.rs).

[6] Repository context: [`00-synthesis.md`](00-synthesis.md), [`01-search.md`](01-search.md), [`11-tier1-synthesis.md`](11-tier1-synthesis.md), [`12-tier2-calibration.md`](12-tier2-calibration.md), [`25-uci-move-overhead.md`](25-uci-move-overhead.md), and [`26-uci-analyse-mode.md`](26-uci-analyse-mode.md).

**Final decision:** **Pursue as a bounded Tier 2 protocol-correctness implementation; do not implement in this Tier 1 investigation and do not promote it as a chess-strength change.**

**Report status:** complete; design-only, no implementation, no passing-test claim, no SPRT, and no default change.
