# 18 — Razoring

**Scope:** Tier 1 research only. This report evaluates razoring as a possible search reinforcement, distinguishes it from the incumbent futility and reverse-futility rules, and defines a conservative diagnostic. **No source implementation, default change, tuning, game match, or SPRT was performed.**

## Executive conclusion

**Recommendation: defer implementation; pursue only a default-preserving counterfactual diagnostic if the owner wants to collect evidence.** Razoring is a forward selective-search technique: at a likely fail-low non-PV node, it searches a reduced horizon—often quiescence search (qsearch)—instead of the full requested depth. It is therefore not equivalent to the repository's existing per-move futility pruning, reverse futility pruning (RFP), late-move reduction, or null-move pruning. It can save substantial work near the horizon, but it has the same fundamental danger as other forward pruning: a quiet move, check, promotion, defensive resource, or tactical transformation omitted by the reduced search can change the true node score.

The current engine already applies several overlapping low-depth heuristics. Razoring should not be added opportunistically or combined with those rules in a first patch. The safest research path is an **observe-only probe** that runs qsearch (or a shallow reduced search) under candidate conditions, records what the hypothetical razor would have returned, and then executes the incumbent search unchanged. Only a repeatable positive net-node signal with no tactical or bound-integrity failures would justify a separate default-off A/B candidate and, eventually, a real paired-game SPRT.

## What razoring is—and is not

The [Chessprogramming Wiki's Razoring reference][1] describes classical razoring as forward pruning when a static or reduced search indicates that a node cannot improve alpha. Its characteristic distinction from ordinary futility is that it generally **reduces the search depth rather than simply skipping one move**. Amir Ban's concise definition is that the subtree is searched “to a reduced depth, typically one less than normal depth,” which retains much of the saving at lower risk than deleting the subtree outright.[1]

There are several historical meanings:

* **Pre-frontier razoring:** at approximately depth two, statically examine child moves in order and stop investigating once sorted moves no longer appear able to raise alpha.
* **Deep or limited razoring:** at approximately depths three or four, use a margin to reduce the node one or more plies, often building on futility assumptions. Ernst Heinz's primary publication page explicitly presents limited razoring at pre-pre-frontier nodes and discusses integrating it with extended futility pruning.[2]
* **Qsearch razoring:** in a non-PV null-window node whose static evaluation is far below the bound, call qsearch and return its fail-low result instead of entering the full-depth move loop. The modern Stockfish source uses this form with a depth-dependent quadratic margin, while guarding mate-related values.[3]

Razoring is consequently a **node-level reduced-search decision**. The current engine's plain futility rule is instead a **move-level skip**: it rejects individual late quiet moves when `static_eval + futility_margin * depth <= alpha`, while other moves at the same node remain eligible for normal search. RFP is also node-level, but in the opposite score direction: when `static_eval - rfp_margin * depth >= beta`, it returns the static evaluation at shallow depth. Razoring would target the fail-low/alpha side; RFP targets likely fail-high/beta nodes.

The distinction matters operationally. Futility's current fail-soft floor preserves a conservative lower bound for the best value seen before skipped moves. A razor that returns qsearch's result is asserting that the reduced search is a sufficient upper-bound-like fail-low witness for the full node. That assertion is heuristic, not alpha-beta soundness. The [general pruning reference][4] classifies such branch removal as forward pruning and notes that it always carries risk of overlooking a result-changing continuation; reductions that still search a branch are generally less risky than outright pruning, but are not risk-free.

## Repository audit

The audit covered the master brief and reinforcement documents **00–12**, plus the later search-focused reports **14–17** and **22**. The standing project rules are consistent: preserve shipped defaults; separate diagnostics from strength claims; record what was actually run; and require a real paired-game SPRT before promoting any game-reachable pruning change.

The relevant implementation is `unchessed-core/src/search.rs` on the assigned `manus/research-facilities` branch. The following behavior is verified by source inspection:

| Area | Current behavior | Razoring implication |
|---|---|---|
| Search entry | `negamax` computes `static_eval` after draw/depth/TT handling; if depth is nonpositive it enters `qsearch` (`search.rs`, approximately lines 493–520). | A razor must be placed after legality/check/TT decisions are understood, and must not confuse an existing depth-zero qsearch transition with a new forward-pruning return. |
| RFP | Non-PV, not in check, depth ≤ 6, non-mate beta, and `static_eval - rfp_margin * depth >= beta` returns `static_eval`; default `rfp_margin = 90` (`approximately lines 23–78, 549–556`). | RFP already removes likely fail-high nodes. Razoring should use the complementary alpha-side condition and must not duplicate or reorder RFP without an isolated experiment. |
| Null move | Non-PV, not in check, depth ≥ 3, `static_eval >= beta`, non-mate beta, and non-pawn material; reduced null search can return beta (`approximately lines 557 onward`). | A razor should not run in the same node class as a null attempt without an explicit precedence rule. Both are forward selective heuristics and their combined miss risk is not additive in a predictable way. |
| Plain futility | Non-PV, not in check, no checking/capturing/promotion move, depth ≤ `futility_max_depth` (default 8), legal count > 1, and `static_eval + futility_margin * depth <= alpha` skips an individual quiet move (`approximately lines 700–750`). Default margin is 150. | This is the closest incumbent. A node-level razor could cause all quiet and tactical-in-qsearch omissions to disappear from the full loop, so the diagnostic must measure overlap and incremental savings rather than raw hypothetical hits. |
| Fail-soft floor | Before a futility `continue`, the code raises `best` to `static_eval + futility_margin * depth`; comments identify this as protection against an unnecessarily pessimistic fail-soft bound. | A razor must define what score and TT bound it returns. Returning a qsearch fail-low score without a floor can make fail-soft scores too pessimistic; inventing a floor can make the recorded value inconsistent with the reduced search. Both must be measured. |
| Qsearch | Stand-pat evaluates first, captures are SEE/delta filtered, checks are searched fully when in check, and mate is returned when no legal evasion exists (`approximately lines 326–445`). | Qsearch is tactical but not a full proof: quiet moves are absent except check evasions. A razor can miss a quiet zwischenzug, quiet defense, promotion setup, or non-capture mate threat. |
| TT and PV | TT exact/lower/upper cutoffs precede static pruning; PV nodes are excluded from RFP, null move, and futility. | Candidate eligibility must exclude PV nodes, root, in-check nodes, mate-range bounds, and incomplete/aborted searches. A diagnostic must not write hypothetical values into TT. |

The implementation uses fail-soft scores and aspiration windows at the root. Therefore a pruning rule that returns an artificially low score can cause more aspiration fail-lows, alter later root ordering, and change PV selection even when a single node's bound appears directionally plausible. This is a central reason not to infer safety from node counts alone.

## Interactions and failure modes

### Plain futility and RFP

Razoring is directionally adjacent to both existing rules but not redundant. Futility skips selected quiet children after move ordering. RFP exits before the move loop when the static score is comfortably above beta. A razor would exit before the move loop when static score is comfortably below alpha, usually after a qsearch confirmation. At shallow depth, the same node may satisfy a razor's alpha margin while many children would also satisfy the current futility condition. The incremental opportunity is therefore likely smaller than the nominal number of low-eval nodes.

A first candidate should **not** replace futility or RFP, and should not stack an unmeasured razor on top of them. The observe-only probe should classify each hypothetical opportunity as: already effectively covered by futility; not covered; qsearch agrees with fail-low; qsearch reaches/crosses alpha; or qsearch produces a tactical/mate value. This gives an overlap estimate before any tree change.

### Null move and ProbCut/multi-cut

Null move tests a reduced search after passing the beta-side static condition; ProbCut and multi-cut also use reduced searches to infer a bound. The current repository's reports 17 and 22 emphasize that each such heuristic needs isolated, default-off evidence. Razoring should not be treated as a free complement simply because it operates on the other side of the window. A position can be both strategically unstable and statically far below alpha: zugzwang, fortress, king attack, and sacrificial positions are exactly where a static low score is least reliable. A combined rule may save nodes while compounding tactical misses.

### Qsearch limitations

The qsearch implementation is intentionally selective: it resolves stand-pat and tactical captures, applies SEE and delta pruning, and searches evasions in check. That makes it a sensible reduced confirmation, but not a substitute for a quiet full-width search. In particular, qsearch can fail low before seeing a quiet move that creates a threat, blocks a line, unpins a piece, improves king safety, or avoids a forced tactical sequence. The diagnostic must preserve the full incumbent search and compare the hypothetical qsearch return with the completed full result.

### Fail-soft, TT, and aspiration risks

A fail-soft search can return a value beyond alpha/beta; that value is useful information but only under the semantics of the search that produced it. A qsearch result below alpha is not automatically the full node's exact score or a safe upper bound. Writing it as an upper-bound TT entry risks poisoning later probes if the full search would have found a quiet alpha-raising move. Returning it without a TT write avoids poisoning but can still distort the caller's aspiration/PVS behavior. Returning alpha instead hides the magnitude of the miss and can alter mate-distance handling.

The diagnostic must therefore record, without changing TT state: static evaluation, alpha/beta, depth, qsearch score, completed incumbent score, whether the incumbent raised alpha or cut beta, and whether the candidate classification would have been fail-low. Mate-range scores and aborts must be excluded from candidate conclusions. Any mismatch involving a mate score, legal evasion, or an incumbent alpha raise is a reject signal.

## Conservative diagnostic design (no implementation in this task)

If approved as a follow-up, add only a **default-off diagnostic/counterfactual probe**, not live pruning. At eligible nodes—non-root, non-PV, not in check, no mate-range alpha/beta, depth in a narrowly selected shallow band such as 1–3, and no incomplete search—the probe would:

1. Capture the incumbent node state and evaluate the current static score.
2. Apply an explicit candidate margin family, initially a small fixed set rather than tuning. For example, test a conservative linear margin and one quadratic-depth margin in centipawns; do not import Stockfish constants as defaults.
3. If `static_eval <= alpha - margin`, run qsearch with the same window the candidate would use, but **do not return its result, skip moves, write TT, or alter alpha/beta**.
4. Continue the normal incumbent negamax. Record the completed full score, bound type, PV/mate status, and node count.
5. Classify the hypothetical candidate as safe agreement, missed alpha raise, tactical/mate disagreement, or indeterminate due to abort/TT/terminal conditions.

The diagnostic should include a strict counterfactual mode and, only after that evidence is clean, a separate default-off A/B tree mode. The first mode measures opportunity and error without changing the tree. The later A/B mode should be a one-feature toggle with otherwise identical parameters, no combination with new multi-cut/ProbCut changes, and fresh TT per run.

### Required telemetry

Each eligible probe should record at least: FEN or stable position hash; side to move; depth; alpha/beta; static evaluation; candidate margin and form; qsearch score and node cost; incumbent score and bound; whether incumbent raised alpha or caused a beta cutoff; PV and mate flags; TT hit metadata; evaluator identity/hash; parameter snapshot; engine commit; and abort status. Aggregate results should report opportunity rate, qsearch cost distribution, overlap with futility/RFP, hypothetical agreement rate, missed-alpha rate, tactical/mate mismatch count, and estimated net nodes after probe cost.

### Fixed-position safety corpus

Use exact FENs and repeatable depth searches covering: forced mates and checks; checking evasions; promotions; sacrificial attacks; quiet zwischenzugs; pinned pieces and discovered attacks; zugzwang and sparse endgames; fortresses; repetition and 50-move boundaries; positions where the best move is a quiet move late in ordering; and positions just above/below the margin. Run fresh-TT repeats and test both sides to move. A candidate is rejected immediately for any legal-evasion loss, mate-distance regression, tactical miss, unexplained score/PV divergence on a forced fixture, or any fail-soft/TT-bound inconsistency.

The diagnostic should also test boundary values: depth below/at/above the candidate band, alpha near mate thresholds, margin zero and large, qsearch fail-high, and aborted searches. Probe counters must not be interpreted as a strength result. A positive signal means only that a controlled candidate experiment may be warranted.

## Verification performed

| Check | Status | Meaning |
|---|---|---|
| Master brief and reinforcement docs 00–12 reviewed | **Completed** | Applied the project’s preservation, provenance, and SPRT gates. |
| Search-focused docs 14–17 and 22 reviewed | **Completed** | Incorporated existing cautions on check extensions, qsearch/SEE pruning, IIR, multi-cut, and null-move/NNUE interactions. |
| `unchessed-core/src/search.rs` source audit | **Completed** | Verified current RFP, null move, futility, qsearch, fail-soft floor, TT, and PV exclusions described above. |
| External literature/web research | **Completed** | Consulted the Chessprogramming Wiki razoring and pruning references, Heinz’s primary limited-razoring/extended-futility publication pages, and current official Stockfish `search.cpp`. |
| Diagnostic implementation | **Not performed** | Explicitly out of scope; no source or default was changed. |
| Fixed-position sweep | **Not performed** | No diagnostic code exists in this task. |
| Rust tests/bench | **Not performed** | This is a research report, not an implementation claim. |
| Cutechess match or SPRT | **Not performed** | No strength or Elo claim is made. |

## Decision gate

Keep the shipped search unchanged. Do not add razoring to the default search, expose a behavior-changing option, or tune a margin from historical Stockfish values. If the counterfactual diagnostic is later approved, require repeatable measurements across fresh-TT runs, a positive net-node effect after qsearch probe cost, and zero tactical/legal/mate safety failures. If opportunity is mostly already covered by futility, or probe cost is positive, **drop razoring for this engine**. If it survives, graduate only the isolated candidate to a separate default-off A/B screen; any default change remains Tier 2/3 work requiring the project’s real paired-game SPRT and explicit approval.

## References

[1]: https://chessprogramming.org/Razoring "Chessprogramming Wiki — Razoring: definitions, historical implementations, and Stockfish example"
[2]: http://people.csail.mit.edu/heinz/dt/node28.html "Ernst A. Heinz — Limited Razoring at Pre-Pre-Frontier Nodes"
[3]: https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/search.cpp "Official Stockfish — current search.cpp, Step 8 Razoring and surrounding pruning"
[4]: https://chessprogramming.org/Pruning "Chessprogramming Wiki — Pruning: forward-pruning risk and bound distinctions"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/18-razoring.md`
**No implementation was made.**
