# 16 — Internal iterative reduction for transposition-table-miss nodes

**Investigation ID:** `tier1-iir`
**Tier:** 1 (research/design only)
**Status:** Complete review; **no implementation, default change, benchmark campaign, or SPRT was performed**.

## Executive decision

**Recommendation: defer implementation, but retain IIR as a narrowly scoped Tier 2 candidate.** The idea is technically compatible with this engine, and the repository has a plausible reason to consider it: recursive alpha-beta search already depends heavily on move ordering, while a transposition-table miss removes the strongest move-ordering hint. However, the same engine already has SEE-based capture ordering, killers, history, checking-move priority, LMR, reverse futility pruning, null-move pruning, ProbCut, aspiration windows, and a one-ply check extension. A depth reduction at a TT-miss node would therefore be an additional selective mechanism layered on top of substantial existing selectivity, not a missing prerequisite.

The conservative conclusion is not that IIR cannot help. It is that the expected benefit is an insurance policy against expensive poorly ordered nodes, while the failure mode is a systematic loss of depth exactly where the TT is least informative. The first useful work is telemetry and a fixed-position A/B diagnostic with an opt-in toggle, not a production default. Any behavior-changing candidate remains subject to the project’s fixed-position safety gates and real paired-game cutechess SPRT; no simulation or node-count result can authorize a default flip.

## What IIR is, and what it is not

Internal iterative deepening (IID) performs a reduced search at a node with no usable best move and uses the resulting move to improve ordering at the intended depth. Internal iterative reduction (IIR), in the modern usage reviewed here, does not perform that preliminary search. It simply searches the entire node at a reduced depth, on the heuristic assumption that a node without a hash move is less important or less likely to justify the full nominal depth [1]. Thus IIR exchanges some nominal horizon for lower cost; it is not a cheaper way to discover a move ordering key.

The Chessprogramming Wiki records IIR as an idea introduced by Ed Schröder in Rebel in 2020 and describes its use as a replacement for, or complement to, IID. It also records the historical evolution from all-node use to PV-only use in Stockfish and then to expected cut-node use in Stockfish and Ethereal [1]. The related IID reference describes the underlying motivation: when neither the previous iterative-deepening PV nor the TT supplies a best move, a reduced search can find a plausible first move; typical reductions include one or two plies or division-based reductions, with node-type and minimum-depth conditions [2]. These sources support the mechanism and its intended trade-off, but do **not** establish a universally profitable reduction amount for this engine.

A TT miss is also not proof that a position has never been searched. It can mean that the entry was evicted, that a concurrent Lazy SMP write raced with the probe, that the entry was not stored, or simply that the current TT size and workload did not retain it. Conversely, a TT hit may contain no usable cutoff even though it supplies a move. The candidate must therefore distinguish at least `probe miss`, `hit with move`, `hit without move`, and `hit with an entry too shallow for cutoff`; treating all of these as equivalent would confound TT capacity and replacement effects with IIR.

## Repository inspection and current search shape

The reviewed checkout is branch `manus/research-facilities`. The principal implementation is `unchessed-core/src/search.rs`; the TT is `unchessed-core/src/tt.rs`.

| Area | Verified current behavior | Relevance to IIR |
|---|---|---|
| Recursive entry | `negamax` counts nodes, checks limits, handles draw/repetition, floors depth while in check, and enters qsearch at `depth <= 0` (`search.rs:505–529`). | An IIR decrement must not turn a legal in-check node into an invalid zero/negative-depth path. |
| TT probe | `search.rs:532–545` probes once, obtains an optional move, and permits exact/lower/upper cutoffs only for non-PV entries at sufficient depth. | The natural signal is `tt.probe(...) == None`, but “usable hash move” is a stricter and more meaningful signal than a raw hit. |
| TT representation | `tt.rs:145–156` validates the XOR key/data pair and returns `None` on miss or a torn concurrent read; entries contain move, score, depth, and bound. | Concurrent misses are possible by design. IIR must remain safe under Lazy SMP and must not infer semantic importance from a race-prone miss. |
| Reverse futility | Non-PV, non-check nodes at depth up to 6 can return the static evaluation when the RFP margin reaches beta (`search.rs:549–557`). | IIR should be applied after cheap early exits, otherwise it spends effort changing nodes that RFP already removes. |
| Null move | Eligible non-PV nodes at depth ≥3 with adequate static eval search a reduced null-move line and may cut off (`search.rs:559–591`). | IIR layered before NMP could alter eligibility and tactical behavior; layered after NMP avoids unnecessary reductions but leaves fewer nodes to measure. |
| ProbCut | Eligible non-PV nodes search captures/promotions at reduced depth and cut on a raised beta (`search.rs:593–704`). | IIR should not reduce the ProbCut verification search; it should be considered only for the later full move loop after ProbCut fails to cut. |
| Move ordering | All legal moves are scored, then selection-sorted; `move_score` uses the TT move plus capture/promotion SEE and quiet heuristics (`search.rs:~700` and the master brief’s verified summary). Killers and history are updated on quiet cutoffs. | IIR’s potential gain is specifically at nodes where these non-TT signals choose a bad first move or where the node is unusually expensive. |
| LMR | After the first legal move, quiet moves past configurable depth/move-count thresholds are reduced; a reduced search is re-searched at full `nd` on fail-high (`search.rs:765–792`). | IIR is node-level reduction and LMR is move-level reduction. Combining them can compound reductions and make a tactical miss hard to recover. |
| Extensions | Checking moves receive `ext = 1`; the child depth is `depth - 1 + ext` (`search.rs:~700–710`). In-check nodes have a minimum depth of 1 before qsearch (`search.rs:525–529`). | IIR must not erase the check extension or bypass the in-check floor. A checking move and a node currently in check are high-risk exclusions. |
| TT storage | The completed node stores its best move, score, depth, and bound (`search.rs` later store path; `tt.rs:store`). | A reduced IIR result must not be represented as if it were a full-depth result. The existing store depth must describe the actual searched depth, or the candidate risks unsound cutoffs. |

The root already uses iterative deepening and aspiration windows (`search.rs:~1033–1108`). Consequently, many ordinary child positions can inherit a move from an earlier completed iteration when the TT retains it. IIR is most likely to matter in newly reached branches, after eviction, after a TT clear, under low hash sizes, or in parallel searches where one thread arrives before another has stored a result. Those are exactly the conditions in which a raw miss is common but its relationship to tactical importance is least certain.

## Fit with current ordering and pruning

### Why a benefit is plausible

A TT move is generally the strongest single ordering hint because it is a move that previously produced a meaningful result at the same position. On a miss, the engine still has useful ordering: captures and promotions are SEE-scored, quiet moves use killers and history, and checking moves are recognized for extension. That means the likely IIR benefit is smaller than it would be in a minimally ordered alpha-beta engine. The remaining opportunity is the tail of high-branching nodes where the first quiet candidate is not the eventual best move and LMR has not yet provided enough protection.

IIR can also reduce pathological time variance. The IID literature explicitly frames internal reduced searching as an insurance mechanism for isolated nodes that become unexpectedly expensive, rather than as a guaranteed average node-count win [2]. This is relevant to a time-limited UCI engine: avoiding a few pathological subtrees may improve completed depth or time predictability even if average fixed-depth nodes are neutral. That hypothesis must be measured rather than assumed.

### Why the interaction risk is material

The engine’s current selectivity is already broad. RFP and NMP may remove or shorten non-PV nodes before the move loop; ProbCut searches tactical captures at reduced depth; plain futility can skip late quiet moves; LMR reduces late quiet moves and re-searches only fail-high candidates. An IIR reduction at the parent changes the depth passed into all of those mechanisms. It can lower the RFP/ProbCut/LMR thresholds, shorten the search horizon, and alter the move-history updates that later nodes use. Therefore an apparently small one-ply reduction can have a nonlinear tree effect.

The key concern is that TT-miss nodes are not necessarily unimportant. A miss may be a fresh tactical position, a position created by a forcing move, or a position whose old entry was evicted precisely because the table is under pressure. In a PV node, reducing depth can drop the principal variation or change aspiration-window outcomes. In an expected cut node, the reduction may be more defensible because only one refutation is needed, but a missed refutation can cause a large subtree expansion or a tactical error. The literature’s historical progression toward node-type restrictions is evidence that eligibility matters, not evidence that one policy transfers unchanged to this codebase [1].

The candidate must also define what “TT miss” means. Applying IIR whenever `probe` returns `None` is simple but includes torn-read misses and positions with no stored entry because the preceding search aborted. Applying it only when the position is non-PV, not in check, sufficiently deep, has no TT move, and is not immediately handled by RFP/NMP/ProbCut is safer. The latter policy gives up some possible speed but avoids using IIR as a blanket replacement for the engine’s existing tactical safeguards.

## Conservative diagnostic design (design-only; not run)

The cheapest useful next step is an **instrumented, default-preserving diagnostic**, not a strength test. It should add counters and an opt-in candidate policy while leaving the default path bit-for-bit unchanged when disabled. The diagnostic should record, per search and by node category:

* total interior nodes and TT probes;
* raw TT misses, hits with a non-`NONE` move, hits without a move, and hits that cannot cutoff due to depth/bound;
* candidate-eligible nodes after excluding PV, in-check, shallow, forced-one-legal-move, RFP/NMP/ProbCut exits, and mate-score windows;
* candidate IIR reductions by nominal depth and node type;
* nodes, completed iteration depth, best move, score, PV, abort status, and hashfull under baseline and candidate modes;
* re-search/fail-high rates after LMR, aspiration-window fail-low/fail-high rates, and tactical fixture outcomes.

A candidate diagnostic policy should be deliberately narrow:

1. Start only at a conservative nominal depth threshold, for example `depth >= 6` (the exact threshold is a hypothesis, not a repository-validated constant).
2. Require a genuine absence of a usable TT move, not merely absence of a cutoff.
3. Exclude PV nodes, nodes in check, checking-move contexts, mate-window scores, forced-move nodes, and qsearch.
4. Run after TT cutoff checks and after RFP/NMP/ProbCut opportunities, immediately before the normal move-loop depth is consumed.
5. Reduce by at most one ply initially, clamp so the child still reaches a legal positive search depth, and preserve the existing check extension. Do not stack a second independent IIR decrement on top of null move or ProbCut.
6. Store the resulting TT entry with the actual reduced searched depth; never advertise the nominal depth for a reduced result.
7. Keep the option default-off and make the telemetry path available with a fixed seed, fixed hash size, and reproducible node limit.

The first fixture set should be small and diagnostic rather than statistically ambitious: start-position and representative quiet middlegames, tactical positions with checks/captures, positions that force a TT miss after a clean TT, low-hash and larger-hash runs, and a small mate/evasion set. Compare clean-TT and warmed-TT behavior, because a policy intended for misses can appear positive or negative solely through TT occupancy. The report should preserve exact FEN, engine commit, evaluator identity, parameter snapshot, TT size, node limit, and mode in machine-readable output.

Safety gates should be:

| Gate | Required result | Stop condition |
|---|---|---|
| Legality/mate | Every returned move legal; checked positions retain legal evasions; mate sign and distance remain correct. | Any regression stops the candidate. |
| Determinism | Repeated fixed-depth/node-limited runs reproduce baseline results within the engine’s deterministic contract. | Unexplained result/PV/node instability stops the candidate. |
| Horizon behavior | No candidate search reaches an invalid negative depth; in-check floor and check extension remain effective. | Any assertion, panic, or lost extension stops the candidate. |
| Cost | Report node change, wall time, completed depth, and abort rate by fixture/category. A provisional review flag is >25% node growth in any category, consistent with the repository’s extension-report convention. | Broad unexplained growth or time loss stops promotion. |
| Strength | Only a real paired-game SPRT can decide game-facing value. | No default flip from fixed positions or telemetry alone. |

A useful diagnostic comparison is not just “baseline versus IIR.” It is a matrix of **baseline, telemetry-only, and opt-in IIR**, crossed with clean/warm TT and at least two hash sizes. This separates the policy’s effect from TT pressure. If misses are rare at the project’s normal `Hash` setting, that is itself a reason to defer the feature; if they are common but candidate-eligible nodes are rare, the expected return is likewise small.

## Risks and mitigations

**Tactical horizon loss.** A fresh position can be important despite lacking a TT move. Exclude checks, in-check nodes, shallow nodes, and mate windows; begin with one ply and run a mate/tactics suite.

**Compounded selectivity.** IIR reduces the parent before LMR, futility, and other child decisions act. Keep the policy out of already reduced/pruned verification paths, record the effective depth, and inspect cases where a reduced node also triggers LMR or a cutoff heuristic.

**Unsound TT metadata.** A reduced result stored at nominal depth could cause an invalid future cutoff. Store actual depth and test warmed-TT repetitions against clean-TT results.

**TT race misclassification.** `tt.probe` can return `None` for a miss or torn concurrent read. Do not treat a miss as a semantic confidence score; compare single-thread and Lazy SMP telemetry, and retain the default-off policy until both are safe.

**PV and aspiration disruption.** PV nodes and aspiration re-searches are particularly sensitive to a lost best line. Exclude PV nodes initially and measure aspiration failures separately.

**False performance attribution.** A lower node count can be a weaker search. Require score/PV/mate safety, completed-depth and time-to-depth measurements, and ultimately SPRT.

**Low practical coverage.** Good iterative deepening plus a shared TT may mean most relevant nodes have a move or useful history. Measure eligible coverage before spending implementation or match budget.

## Verified, assumed, and not run

### Verified from the checkout

The branch is `manus/research-facilities`. The recursive search probes the TT before static-evaluation and pruning; non-PV TT entries can cut off at sufficient depth. The engine has RFP, NMP, ProbCut, plain futility, SEE-based capture ordering, killers/history, LMR, aspiration windows, and a one-ply checking-move extension. The TT uses a lock-free two-atomic-slot representation, depth-preferred replacement, and a miss result that also covers invalid/torn observations. The master brief explicitly identifies IIR for TT-miss nodes as Tier 1 item 8 and requires design-only work to state assumptions and preserve defaults.

A baseline test command was attempted:

```text
cargo test -p unchessed-core --release
```

It was **blocked before compilation** because the installed Cargo rejected the repository lockfile: `lock file version 4 requires -Znext-lockfile-bump`. Therefore no test pass, node benchmark, fixed-position comparison, or parity result is claimed in this investigation.

### Assumed or design-only

The likely benefit from missing TT moves, the proposed depth threshold, one-ply amount, node-type exclusions, and the provisional 25% review flag are hypotheses/design constraints, not measured results. No implementation, telemetry, fixed-position A/B run, SPRT, or default change was performed. The literature supports the mechanism and historical use of node-type restrictions, but does not predict this engine’s Elo or establish a portable parameter.

## Recommendation and next action

**Defer production IIR.** Do not add a default reduction to `negamax` in this Tier 1 review. If the owner later promotes the idea to Tier 2, implement only an opt-in, one-ply, sufficiently-deep, non-PV diagnostic with explicit TT-miss/usable-move counters and actual-depth TT storage. Run the safety matrix first. Proceed to a paired-game SPRT only if the candidate is legal, deterministic, tactically safe, and shows a credible time/depth benefit without broad node growth. Keep the incumbent default unchanged unless that SPRT is positive and independently reviewable.

## References

[1]: https://chessprogramming.org/Internal_Iterative_Reductions "Chessprogramming Wiki — Internal Iterative Reductions"
[2]: https://chessprogramming.org/Internal_Iterative_Deepening "Chessprogramming Wiki — Internal Iterative Deepening"
[3]: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp "Stockfish search implementation (current source; TT-hit state and depth/selectivity handling)"
[4]: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/blob/manus/research-facilities/unchessed-core/src/search.rs "Unchessed-UCI-Engine search implementation"
[5]: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/blob/manus/research-facilities/unchessed-core/src/tt.rs "Unchessed-UCI-Engine transposition table implementation"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/16-iir.md`
**No implementation was made.**

## Evidence note

The external literature review used the Chessprogramming Wiki pages [1] and [2] and the current Stockfish `search.cpp` source [3]. The local code and report inventory were inspected directly in the named checkout; citations to local line ranges refer to the current working tree at report time.
