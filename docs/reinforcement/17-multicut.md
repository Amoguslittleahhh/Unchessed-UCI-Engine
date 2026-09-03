# 17 — Multi-cut pruning

**Investigation ID:** `tier1-multicut`
**Scope:** Tier 1 research/design only. Assess whether multi-cut pruning is a sensible addition to Unchessed given its existing null-move pruning (NMP), ProbCut, LMR, futility pruning, and reverse futility pruning. No implementation, default change, match, or SPRT was performed.

## Executive disposition

**Recommendation: defer; do not implement as a second speculative pruning rule at present.** Multi-cut pruning is a real and historically useful forward-pruning technique, but in this engine its decision signal is substantially covered by existing mechanisms. The remaining possible niche—using several reduced fail-high child searches at an expected cut node to distinguish a genuinely broad tactical advantage from a single speculative refutation—would require extra searches at exactly the nodes where NMP and ProbCut already spend selectivity budget. Without telemetry, a clean incumbent/candidate toggle, and a real paired-game SPRT, adding it would be an unpriced tree change rather than a safe reinforcement.

A narrowly scoped future diagnostic is worthwhile: count *would-be* multi-cut opportunities and simulate only the reduced probes in an opt-in/offline mode, while never changing the returned score or ordinary tree. That can estimate opportunity rate and cost before any code-changing experiment. It is not evidence of Elo and is not implementation authorization.

## What multi-cut means

The original Björnsson–Marsland method invests additional work at an expected cut node. It searches the first `M` ordered moves at a reduced depth `d - 1 - R` with a null window, and prunes the whole node when at least `C` of those moves fail high (`C < M`). A common illustrative setting is `M=6`, `C=3`; the point is not that any fixed constants transfer to this engine, but that **multiple independent-looking refutations** make it less likely that the node is singular and that a currently unsearched move is the relevant alternative.

This is speculative forward pruning: it can return the hard beta bound before all legal moves are searched. It differs from ordinary alpha-beta, which is exact when its searched bounds and move ordering assumptions are respected. The Chessprogramming Wiki describes the classic form as searching a reduced set of moves at cut nodes and returning beta after multiple fail-highs, and notes later variants at expected all nodes and uses alongside restricted singular extensions [1]. The primary paper reports a correlation between the number of promising alternatives at cut nodes and a new principal variation, then introduces a forward-pruning method based on that information [2]. Those are literature claims, not measurements in Unchessed.

## Repository inspection and fit

The master brief confirms that the following are already present: reverse futility pruning, NMP, LMR, ProbCut with an optional SEE filter, plain futility pruning, and broad SEE-based capture ordering. The source inspection confirms the details below.

| Existing mechanism | Current source behavior | Overlap with multi-cut |
|---|---|---|
| Reverse futility pruning | At non-PV, non-check nodes with depth at most 6, non-mate beta, and `static_eval - rfp_margin * depth >= beta`, returns the static evaluation immediately (`search.rs`, around lines 549–557). | Whole-node static-eval optimism already cuts many easy fail-high nodes without child probes. |
| NMP | At eligible non-PV, non-check nodes with depth at least 3, `static_eval >= beta`, non-mate beta, and non-pawn material, searches a null move at reduced depth and returns beta on fail-high (`search.rs`, around lines 559–591). | Both are expected-cut-node forward pruning. NMP is cheaper and tests the null-move observation; multi-cut spends several child searches where NMP may already cut. |
| ProbCut | At non-PV, non-check nodes from depth 5, searches ordered captures/promotions at `depth - probcut_reduction` against `beta + probcut_margin`; one fail-high returns beta (`search.rs`, around lines 593–676). Default values are margin 200, reduction 4, minimum depth 5; `ProbcutSeeFilter` is false by default. | ProbCut is a one-or-more tactical reduced-search cutoff. Multi-cut generalizes the evidence from one reduced fail-high to a count of fail-high alternatives, but would add quiet moves and/or additional probes and could duplicate tactical capture work. |
| Futility pruning | At non-PV non-check nodes, skips individual late quiet moves at shallow depth when the static evaluation plus a depth-scaled margin cannot reach alpha (`search.rs`, around lines 721–750). | Both exploit the low probability of useful alternatives, but on opposite bounds: futility is fail-low/alpha-side and per-move; multi-cut is fail-high/beta-side and node-wide. |
| LMR | For sufficiently deep, late, quiet, non-check, non-PV-eligible moves, searches reduced depth and re-searches at full depth only if the reduced search raises alpha (`search.rs`, around lines 769–792). | Multi-cut would add another reduced search layer before the normal move loop. LMR already reduces the cost of late alternatives after move ordering; a multi-cut probe can make the same ordering assumptions expensive rather than cheaper. |
| Move ordering / TT | TT move, history, killers, and SEE-based capture ordering feed both the ProbCut and main move loops. | Multi-cut’s safety depends critically on which first `M` moves are selected; ordering quality determines both savings and tactical blindness. |

There are no check, singular, or recapture extensions in the current search according to the inspected source and prior report; a checking move does receive the existing one-ply extension and in-check nodes receive a depth floor. That makes multi-cut especially risky around checks and forced evasions: a probe policy must not bypass legal checks, mate-range scores, or the existing extension semantics.

The existing code also shares a transposition table across Lazy SMP helper threads. Any future diagnostic must therefore distinguish local probe counters from TT effects and use fresh, fixed-size tables in deterministic single-thread runs first. A multi-cut implementation that writes shallow bounds into the shared TT could alter later searches and make an apparent node win a cache-policy artifact.

## Redundancy and possible niche

The strongest redundancy is with NMP and ProbCut, not LMR. NMP answers: “does the side to move still fail high after effectively passing?” ProbCut answers: “does a tactically promising reduced child already exceed a margin above beta?” Multi-cut answers: “do several reduced child alternatives independently fail high, so this cut node is not singular?” The signals are not identical, but they are all speculative fail-high evidence at non-PV nodes.

The possible additive niche is a node where static evaluation is below beta (so NMP and RFP do not trigger), no single capture reaches the ProbCut threshold, but multiple well-ordered legal alternatives each produce a reduced fail-high at the normal beta bound. In theory this can prune a broad subtree. In practice, the cost is front-loaded: the node must search up to `M` reduced children before saving the full search, and the first fail-high is not enough. At shallow or low-branching nodes the overhead dominates; at tactical nodes, checks, captures, promotions, and TT cutoffs make the interaction with ProbCut and existing move ordering difficult to attribute. A broad multi-cut test over all legal moves would also duplicate work and increase tactical-risk surface.

Multi-cut should not be described as a safer version of ProbCut. It is still speculative forward pruning, and its “multiple witnesses” are correlated because child searches share the same position, evaluator, move-ordering heuristics, and TT. Counting two fail-highs is not statistically independent proof. The original literature supports the concept and reports chess experiments, but does not establish that classic constants or a modern NNUE alpha-beta engine benefit after NMP/ProbCut/LMR are already active.

## Safe fixed-position diagnostic (design only)

The cheapest useful test is an **offline, opt-in telemetry/probe diagnostic**, not a live tree change. It should be implemented only after an owner approves a Tier 2 plan; this report did not implement it. The diagnostic should run with `Threads=1`, a fresh TT per position, fixed hash size, fixed evaluator, fixed binary/commit, and deterministic fixed-depth searches. Run both HCE and the shipped NNUE because search calibration against HCE cannot be treated as NNUE evidence.

Use 20–50 positions, stratified across: start position and an opening tabiya; forcing tactical positions with captures and checks; quiet middlegames with many legal moves; hanging-material positions; mate-in/forced-evasion positions; sparse endgames and zugzwang/null-sensitive positions; and positions designed to put moves just across LMR/futility boundaries. Include exact FENs and side to move. For each FEN and depth, record best move, score, mate distance, PV, completed iteration, node count, abort status, TT size, evaluator identity/hash, parameter snapshot, and engine commit.

The diagnostic needs two conceptually separate modes:

1. **Counterfactual opportunity mode:** at eligible non-PV, non-check nodes, identify the first `M` ordered legal moves and count how many would fail high in a reduced null-window search at candidate `(M,C,R)` values. Do not return beta and do not skip the normal search. Record probe nodes, qualifying nodes, fail-high count, and whether the incumbent later produced a real beta cutoff.
2. **A/B tree mode (only if a clean toggle is added):** compare incumbent and multi-cut candidate with identical parameters except the feature. Candidate must return beta only after the chosen count `C`; no PV-node, in-check, mate-range, root, or legal-evasion shortcut may use it. This mode is a reject-only safety screen, not a strength result.

The principal measurements are opportunity rate, average and tail probe cost, saved full-search nodes, net nodes, incumbent/candidate score and PV agreement, and tactical misses. A promising signal would require repeatability across fresh-TT runs and a positive net-node effect after probe cost, not merely a high count of hypothetical cutoffs. Flag unexplained node growth above the project’s provisional 25% category threshold for review; that threshold is a diagnostic guard, not a promotion criterion. Any score/PV/mate regression on a forced tactical or legal-evasion fixture is an immediate reject.

The safety suite must include null-sensitive positions and repetition/50-move cases because null moves already alter the path and halfmove state. It must include positions where the best move is a quiet alternative outside the first `M`, because that is the central failure mode of ordering-dependent multi-cut. It must test `M` larger than legal move count, `C=1`, `C=M`, depth below the reduction threshold, and mate-range beta bounds as fail-closed parameter boundaries. Candidate parameters must be explicit and default-off; no imported Stockfish or historical multi-cut constants are valid without calibration to this engine’s evaluator and search.

## Verification performed

| Check | Result | Interpretation |
|---|---|---|
| Source inspection of NMP, ProbCut, RFP, futility, LMR, move ordering, and TT | **Completed** | The overlap analysis above is repository-grounded. |
| Review of master brief and reinforcement docs 00–12 | **Completed** | Applied the standing rules: defaults do not move; fixed tests are not Elo evidence; new pruning needs a real SPRT. |
| Web research | **Completed** | Consulted the Chessprogramming Wiki overview and Björnsson–Marsland’s Springer chapter/abstract; links are listed below. |
| Multi-cut implementation | **Not run / not made** | This is design-only Tier 1 work. |
| Fixed-position multi-cut sweep | **Not run** | No candidate toggle or telemetry exists, and no code change was authorized. |
| Focused Rust tests | **Blocked before compilation** | Installed Cargo 1.75 rejects this checkout’s lockfile version 4 (`-Znext-lockfile-bump`). This is an environment limitation, not a passing test result. |
| Cutechess match or SPRT | **Not run** | The required real paired-game gate and configured cutechess/book environment are unavailable; no Elo or strength claim is made. |

## Decision gate and next step

Keep the shipped search unchanged. Do not add a live multi-cut rule, expose a behavior-changing default, or combine it with ProbCut in one experiment. If this topic is reopened, first implement only default-off counters/counterfactual probes behind a reviewed diagnostic interface, run the fixed-position screen, and publish the full parameter/telemetry data. A candidate that survives that screen still needs an isolated paired-game experiment against the incumbent, followed by the project’s real cutechess SPRT before any default change. If opportunity rate is low or net probe cost is positive, drop the item. If it is high and net-saving with no fixed-position tactical regression, graduate it to Tier 2/3 as a separate candidate; do not infer that it is additive with ProbCut until a combined arm is tested independently.

## References

[1]: https://chessprogramming.org/Multi-Cut "Chessprogramming Wiki — Multi-Cut (definition, pseudocode, modern usage, and bibliography)"
[2]: https://doi.org/10.1007/3-540-48957-6_2 "Yngvi Björnsson and Tony Marsland, Multi-cut Pruning in Alpha-Beta Search, Computers and Games / LNCS 1558, pp. 15–24 (1999; CG 1998 paper)"
[3]: https://chessprogramming.org/Null_Move_Pruning "Chessprogramming Wiki — Null Move Pruning"

**No implementation was made.**
