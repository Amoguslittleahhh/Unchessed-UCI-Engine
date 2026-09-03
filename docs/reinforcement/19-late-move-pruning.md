# 19 — Late-move and move-count pruning

**Investigation ID:** `tier1-late-move-pruning`
**Tier:** 1 (research/design only)
**Status:** Source audit and literature review complete; no implementation, tuning, fixed-position sweep, match, or SPRT was performed.

## Executive conclusion

**Recommendation: defer implementation, and treat the item as low priority unless telemetry shows a substantial untouched node opportunity.** The engine already has the three mechanisms that occupy most of the same shallow-search territory: reverse futility pruning (RFP), per-move quiet futility pruning, and late-move reductions (LMR). A conventional late-move-pruning (LMP) rule would add a fourth, more aggressive forward-pruning gate whose decision depends heavily on move ordering. It may save nodes, but it can silently remove the only quiet move that refutes a threat. There is no evidence in this repository that the likely incremental gain has been measured, and no implementation is justified by this research alone.

If this item is revisited, it should be an **isolated, default-off diagnostic** first. The candidate must be evaluated against a control with identical ordering, TT, evaluation, and all existing pruning. It must pass the shallow tactical-safety suites below, show positive net node savings after move-generation/ordering overhead, and then clear the project’s required paired-game cutechess SPRT before any default change. This report does not authorize Tier 2/3 work.

## What the technique is—and is not

Late-move reductions and move-count pruning are related but not interchangeable:

| Mechanism | Action | Typical evidence required | Main failure mode |
|---|---|---|---|
| **LMR** | Search a late move at reduced depth, then re-search at full depth if the reduced result raises alpha. | Reduced search must preserve discovery through the re-search trigger. | A good move appears bad at reduced depth and is not re-searched because of another condition. |
| **LMP / move-count pruning** | Skip eligible late moves entirely once a move-count threshold is reached. No child search and no re-search occur. | Ordering quality and safety exclusions must make omitted moves very unlikely to be best or necessary. | A quiet tactical defense or zwischenzug is never searched: tactical blindness and changed root choice/score. |
| **Plain futility pruning** | Skip an individual quiet move when static evaluation plus a depth-scaled margin cannot reach alpha. | Margin calibration and tactical exclusions. | Static evaluation misses a non-material positional or tactical swing. |
| **RFP** | Return a whole-node score when static evaluation is sufficiently above beta. | Conservative beta/mate/check boundaries. | A node-wide cutoff suppresses a surprising refutation. |

The official Stockfish terminology page defines LMP as pruning quiet moves after the first few moves supplied by move ordering, while it defines LMR as trying late moves at lower depth to prove them below alpha [1]. The Chessprogramming Wiki describes LMR as reduced-depth search with possible full-depth re-search [2]. Its futility-pruning page identifies move-count-based pruning/LMP as a later combination of extended futility, history leaf pruning, and LMR ideas [3]. Thus “late” describes the move’s **ordinal position after ordering**; LMP is not merely a stronger LMR reduction.

The distinction matters operationally. LMR leaves a chance for a late move to fail high and trigger recovery. LMP removes that chance. In alpha-beta terms this is forward pruning: unlike a bound proven by already-searching a refutation, an omitted branch is discarded on a heuristic assumption. The Chessprogramming pruning overview explicitly characterizes forward pruning as risky because it can overlook lines that influence the root score [4].

## Repository audit

The master brief was read first. It confirms that this is a Tier 1 research item, that RFP, LMR, futility, ProbCut, null-move pruning, SEE capture ordering, and the project’s no-default-flip/real-SPRT rule already exist. Reinforcement reports `00`–`12` were reviewed for the evidence and promotion-gate conventions; the search-focused reports through `18` were also inspected where they bear on overlap and safety.

The relevant implementation is `unchessed-core/src/search.rs`:

* `SearchParams` defaults are `rfp_margin = 90`, `lmr_min_depth = 3`, `lmr_min_movenum = 3`, `lmr_big_movenum = 12`, `futility_margin = 150`, and `futility_max_depth = 8` (`search.rs:22–80`). These are tunable UCI parameters, but changing any pruning behavior remains a tree-changing experiment, not a harmless calibration.
* RFP is a whole-node non-PV cutoff at non-check depths up to 6 when `static_eval - rfp_margin * depth >= beta` and the score is not near mate (`search.rs:548–556`).
* Plain futility is a per-move non-PV gate. It applies only to quiet, non-check, non-promotion moves, after legality and `legal_count > 1`, at depths up to `futility_max_depth`; it skips when `static_eval + futility_margin * depth <= alpha` (`search.rs:719–750`). The implementation deliberately raises the fail-soft floor before continuing.
* LMR applies to quiet, non-promotion, non-check moves only when the current depth is at least 3 and more than the configured minimum number of legal moves has been searched. It reduces by one or more plies based on move ordinal, with extra reduction at large move counts and non-PV nodes, and re-searches at full depth if the reduced search exceeds alpha (`search.rs:765–790`).
* Legality is checked before `legal_count` is incremented, and checking moves, captures, promotions, and in-check nodes are excluded from the current quiet futility/LMR gates. This is a useful safety baseline, but it does not prove that all quiet tactical moves are recognized as checks or captures before a future LMP test.
* Move ordering uses the TT move, killers/history, and SEE-backed capture scoring; consequently, an LMP threshold would be coupled directly to ordering quality. A move late because of a history error is indistinguishable from a genuinely irrelevant move to a count-only rule.

This audit found **no existing move-count skip of the main-search move list**. The current `legal_count` is used for futility and LMR conditions, not to stop considering later quiet moves. Qsearch has its own capture/SEE and delta-pruning behavior, which is outside this item and must remain a separately controlled factor.

## What a candidate would look like

A conventional candidate would be limited to non-root, non-PV, non-check nodes at shallow depth, with non-pawn material, and would skip only quiet moves after a threshold such as a depth- and improving-dependent function. Current Stockfish source illustrates this shape: it increments `moveCount`, then at shallow-search Step 15 calls a quiet-move skip after a threshold of the form `(3 + depth * depth) / (2 - improving)`, while retaining separate LMR and futility logic [5]. Those constants are **not** portable recommendations. Stockfish has different evaluation, history/continuation-history infrastructure, search maturity, and extensive testing; importing its formula would be unjustified.

A safer experimental design would add a diagnostic predicate, not a default behavior:

1. Count only **legal** moves after the same existing ordering and exclusions.
2. Apply only at shallow non-PV nodes, initially no deeper than the depths where plain futility already operates.
3. Exclude nodes in check, near-mate bounds, PV/root nodes, positions with no non-pawn material, and all captures, promotions, checking moves, killer/TT moves, and moves with strong history. Consider excluding passed-pawn pushes, pawn moves, recaptures, and evasive moves entirely in the first screen.
4. Keep LMR and plain futility unchanged in the control and candidate so attribution is possible.
5. Preserve legal-move accounting and a fail-soft result when all eligible quiets are skipped; never manufacture a stalemate or mate result from “no searched move.”
6. Record per-node depth, move count, improving flag, candidate move class, static score/window, and whether the skipped set contained the eventual control PV move. Counters are essential: without hit rate and overlap data, node totals cannot show whether LMP is additive or merely duplicating futility.

The first experiment should compare a small family of conservative thresholds, not tune a broad parameter surface. A threshold that fires mostly on moves already rejected by futility has no additive value; a threshold that fires on quiet moves surviving futility is precisely where tactical risk is highest.

## Fit with RFP, futility, and LMR

The expected incremental opportunity is narrow. RFP already handles nodes whose static score is comfortably above beta. Plain futility already removes quiet moves that cannot reach alpha under its margin. LMR already makes late quiet moves cheap and provides a full-depth recovery when a reduced search is promising. LMP can only be additive in the intersection where a move is (a) not covered by a whole-node RFP cutoff, (b) not statically futile, and (c) late enough that even a reduced search is considered unnecessary.

That intersection is also the least safe one: the move survived the static evaluation margin, yet its ordinal position says it is unimportant. In a mature engine, move ordering may make this useful; in this engine, the absence of continuation history/countermove infrastructure (the subject of the separate report `20-countermove-continuation.md`) makes a count threshold less informed. A poor ordering signal can push a tactical defense behind the threshold. Conversely, if ordering is already excellent, LMR may have captured most of the inexpensive savings, leaving LMP with little node reduction after its own checks.

Interactions requiring isolation include:

* **RFP:** never use LMP to compensate for a changed RFP margin. RFP returns a node score; LMP skips siblings. Their errors propagate differently.
* **Plain futility:** measure overlap explicitly. Applying LMP after futility is logically additive only for surviving quiets, but the surviving set is exactly the safety-sensitive tail.
* **LMR:** do not interpret “reduced to a very small depth” as equivalent to “pruned.” LMR’s re-search condition is a correctness buffer that LMP removes.
* **ProbCut and null move:** both are already forward-pruning mechanisms. A combined candidate can look fast while accumulating correlated tactical blindness. Keep them unchanged and report node savings conditional on whether a node was already cut by either mechanism.
* **TT and Lazy SMP:** a changed tree changes TT contents and thread interaction. Fixed-position comparisons need fresh TT and deterministic single-thread runs first; a Lazy SMP result is not a clean attribution test.

## Tactical-blindness risk model

The central risk is not ordinary positional inaccuracy; it is a **quiet move with a delayed tactical purpose**. Examples include moving a pinned defender, uncovering a discovered attack, creating a mating threat, interposing against a ray, sacrificing a piece to force a queen win, or playing a zwischenzug that is not itself a capture/check. A move may also be quiet by move type while being the only legal response to an imminent mate. Static evaluation and SEE do not reliably classify these cases because the tactical payoff can occur several plies later and may be non-material at the move’s destination.

Risk rises with:

* narrow null windows and non-PV nodes, where a false fail-low can be stored as a bound and influence later searches;
* shallow depth, where the delayed payoff is beyond the horizon;
* weak or stale history/order data, especially after a position transition unlike the training/search history;
* positions with exposed kings, forcing threats, overloaded defenders, pins, x-rays, and promotion races;
* multi-cut-like assumptions that several early moves are representative of all later moves;
* MultiPV or root-adjacent analysis, where a skipped alternative can change ranking even if the first move remains stable.

Risk is lower—but not zero—for quiet moves at well-ordered, non-tactical middlegame nodes with ample material and a stable static evaluation. That is a hypothesis to test, not a safety guarantee. The forward-pruning literature supports this caution: Hoki and Muramatsu’s study analyzes futility, null-move, and LMR as forward-selective techniques and reports strong branching reduction in shogi, but shogi’s branching factor and tactical geometry differ from chess; its result cannot establish a chess Elo benefit for this engine [6].

## Shallow safety suites

Before any strength test, a candidate must be compared with an unmodified control on fixed positions using fresh TT, deterministic single-thread search, and multiple depths/windows. The test should assert more than “the best move is unchanged.” A proposed minimum suite is:

| Suite | Required positions | Failure signal |
|---|---|---|
| **Checks and evasions** | In-check positions with quiet blocks, king moves, captures, and only-one-evasion mates; quiet moves that create or prevent check. | An evasion is skipped, mate distance changes, or a checking threat disappears. |
| **Quiet tactical defenses** | Pins, x-rays, overloaded defenders, discovered attacks, interpositions, clearance moves, and moves that stop a mate one or two plies later. | Control finds a forced tactical defense absent from the candidate PV/score. |
| **Quiet sacrifices and zwischenzugs** | Non-capture sacrifices, deflections, interference, zwischenzug-before-recapture, and quiet moves winning material after a delayed response. | Candidate misses the control’s material/tactical swing. |
| **Promotions and races** | Promotion threats where the best move is a quiet pawn push or king move; underpromotion continuations and race positions. | Candidate changes win/draw/loss or misses promotion timing. |
| **Pins and legality** | Pinned pieces, discovered checks, en-passant x-rays, king-adjacent attacks, and positions with many pseudo-legal but few legal moves. | Illegal-move handling, mate/stalemate status, or legal move count differs. |
| **Endgames** | Sparse material, fortress/zugzwang, triangulation, opposition, pawn breakthroughs, and insufficient-material transitions where supported. | A quiet waiting move or only drawing move is pruned. |
| **Boundary matrix** | Depths at and around threshold; improving/non-improving; PV/non-PV; alpha just reachable/unreachable; mate-near scores; move counts 1, threshold−1, threshold, threshold+1. | Off-by-one activation or unsafe activation in excluded bounds. |

For every position, compare root score, WDL sign if available, PV first move, PV legality, mate score and distance, and the set of legal root moves. At interior nodes, instrument whether the control PV move was in the candidate’s skipped set. Include tactical test suites from standard engine test collections if licensing and reproducibility permit, but do not treat a passing hand-picked suite as proof of safety.

A node-count study should then use a representative quiet/middlegame corpus plus the adversarial suite. Report median and tail node change, wall time, LMP hit rate, overlap with futility, and the count of score/PV divergences. Measure the cost of move classification and counters. A candidate that saves nodes but increases wall time is not a performance win.

## Verification performed

| Check | Result | Meaning |
|---|---|---|
| Master brief | **Read** | Applied Tier 1 scope, existing-feature inventory, and no-default/real-SPRT rules. |
| Reinforcement docs `00`–`12` | **Reviewed** | Followed established evidence, safety-suite, and promotion conventions. |
| Search source audit | **Completed** | Verified current RFP, plain futility, LMR, SEE ordering, legality checks, and defaults in `unchessed-core/src/search.rs`. |
| External research | **Completed** | Read official Stockfish terminology, current Stockfish `search.cpp`, Chessprogramming pruning/futility/LMR pages, and the Hoki–Muramatsu forward-pruning study record. |
| Implementation | **Not performed** | No source, default, UCI option, or telemetry was changed. |
| Fixed-position safety suite | **Not run** | No diagnostic toggle or harness was added. |
| Rust tests | **Blocked before compilation** | Installed Cargo reports that `Cargo.lock` version 4 requires `-Znext-lockfile-bump`; therefore no passing-test claim is made here. |
| Cutechess match/SPRT | **Not run** | No Elo, strength, or default-change claim is made. |

## Decision and next gate

Keep the current search unchanged. Do not copy Stockfish thresholds, expose an on-by-default LMP option, or infer safety from node reductions. The default disposition is **defer** because LMP is plausibly redundant with existing futility plus LMR and has a more direct tactical-blindness failure mode. It becomes worth a bounded experiment only if instrumentation demonstrates a meaningful population of late quiet moves that survive futility, are not already made cheap by LMR, and account for enough nodes to justify the risk.

If that condition is met, the next step is a default-off, one-mechanism diagnostic followed by the safety suites and fresh-TT node study specified above. A surviving candidate still requires an isolated paired-game cutechess SPRT under the repository’s standing rules. Any regression in forced mate, legal evasion, promotion race, or quiet tactical-defense coverage is a stop signal even if nodes fall. No Tier 2/3 work or compute spend follows automatically from this report.

## References

[1]: https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html "Official Stockfish Docs — Terminology"

[2]: https://chessprogramming.org/Late_Move_Reductions "Chessprogramming Wiki — Late Move Reductions"

[3]: https://chessprogramming.org/Futility_Pruning "Chessprogramming Wiki — Futility Pruning"

[4]: https://chessprogramming.org/Pruning "Chessprogramming Wiki — Pruning"

[5]: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp "Official Stockfish — current search.cpp"

[6]: https://www.sciencedirect.com/science/article/pii/S1875952011000450 "K. Hoki and M. Muramatsu, Efficiency of three forward-pruning techniques in shogi (2012)"

**No implementation was made.**

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/19-late-move-pruning.md`
