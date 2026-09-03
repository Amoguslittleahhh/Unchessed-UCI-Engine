# Countermove and continuation history

**Tier 1 investigation — design and research only**
**Status:** No implementation, benchmark, fixed-position sweep, match, or SPRT was run.
**Repository:** `Unchessed-UCI-Engine`, branch `manus/research-facilities`, audited at commit `667e9068c26b8201e81435d4a00c2d4a5ec9fd13`.

## Executive conclusion

Countermove and continuation history are a plausible, standard, and relatively low-risk extension of the engine's existing killers/history move ordering. They do not alter the evaluation, legal move set, or alpha-beta bounds when used only as ordering signals. The current implementation has a deliberately small state footprint—two ply-indexed killer moves and a side/from/to quiet history table—so it has no response-conditioned memory. A **one-ply countermove table** is the natural first candidate: reward a quiet move that produces a beta cutoff as the response to the immediately preceding quiet move, and give that exact move a modest ordering bonus in the corresponding child position. A **two-ply continuation-history table** is a reasonable follow-up, but it adds substantially more memory and update/initialization complexity and should not be bundled with the first experiment.

The recommendation is **pursue a telemetry-only diagnostic, then conditionally pursue an isolated default-off A/B candidate; do not change a shipped default yet**. The diagnostic must measure whether response-conditioned information is both frequent and useful after the existing TT move, SEE ordering, killers, and global history have had their chance. Promotion requires zero correctness regressions, positive net search-cost evidence after table overhead, and eventually a fresh paired-game SPRT. Fixed positions and telemetry can justify the next experiment, not a strength claim or default promotion.

## What was researched

The master brief identifies this item as a natural extension of killers/history: order moves by how well they worked as a response to the opponent's last move(s), rather than only by general success. The reporting and promotion constraints in the brief were applied: preserve defaults, distinguish design from executed work, and treat all game-facing changes as real-SPRT-gated.

I reviewed the master brief and reinforcement reports `00` through `12`, including the search baseline and Tier 1 synthesis, and the adjacent search investigations `14`–`18` plus the NNUE/null-move report `22`. Those reports consistently require an incumbent-preserving diagnostic, exact provenance, fixed-position safety checks, and no inference of Elo from node counts. The local source audit covered `unchessed-core/src/search.rs` and `tt.rs`.

External references were read rather than relying on snippets. The Chessprogramming Wiki describes countermove heuristic as introduced by Jos Uiterwijk in 1992; it stores a likely response indexed by the prior move, commonly as `[from][to]` or memory-friendlier `[piece][to]`, and updates it on a quiet beta cutoff [1]. Its history-heuristic article describes **countermove history** as a history table conditioned on a previous move and **continuation history** as the general n-ply form: one-ply and two-ply cases correspond to countermove and follow-up history, while stronger engines may retain more plies [2]. Current official Stockfish source provides an authoritative contemporary implementation reference: its `PieceToHistory` entries are used inside nested continuation-history tables, and its search updates continuation histories alongside quiet histories rather than treating them as a replacement for the ordinary history table [3][4]. These sources establish the mechanism and terminology, not transferable constants or an expected Elo gain.

## Current engine baseline

The relevant search state is in `unchessed-core/src/search.rs`:

| Component | Current behavior | Consequence for this item |
|---|---|---|
| TT move | `move_score` returns `1 << 22` for the TT move | Any response heuristic must remain below the TT priority. |
| Captures/promotions | SEE-positive moves receive `1_000_000 + SEE`; SEE-negative moves receive a very low score | The first candidate should concern quiet moves only; do not reorder tactical moves. |
| Killers | Two moves per ply receive `800_000` and `790_000` | A countermove bonus must not silently override a valid killer without evidence. |
| Global quiet history | `history[side][from][to]`, an `i32` table initialized to zero; `move_score` returns it for other quiets | Countermove/continuation history should be an additive, bounded component, not a replacement. |
| Updates | A quiet beta-cutoff move is installed in the two killer slots and gets `depth * depth` global-history credit; values above `1 << 20` are halved | Existing updates are simple and unsaturated; new tables need explicit bounded/gravity updates and reset semantics. |
| Node traversal | Legal moves are generated, scored, and selected by repeated best-first extraction; child positions are copy-made | The response context is available at the parent/child boundary, but current state does not retain the prior move itself. |
| Existing context | `path` stores hashes for repetition detection, not moves | A candidate must add explicit previous-move stack state; hashes cannot reconstruct a move safely. |
| Search sharing | TT is shared for Lazy SMP; each `Searcher` owns killers/history | Per-search or per-worker continuation tables are safer initially than shared mutable response tables. |

The exact move-ordering routine is at lines 351–378: TT, then SEE captures/promotions, then killer 1, killer 2, then `history[side][from][to]`. The normal child search begins after `pos.make(m)` and the parent records a quiet beta cutoff at lines 808–819. There is no current countermove or continuation table, and no current previous-move field in `Searcher`.

This is an ordering-only opportunity, not a pruning opportunity. It should not be mixed with LMR, futility, RFP, ProbCut, null move, IIR, multi-cut, razoring, or check-extension changes. The adjacent reports show that those mechanisms already have substantial interaction risk; isolating this item is especially important because an ordering change can change the effective LMR move number and therefore the searched tree even though the heuristic itself is not a pruning rule.

## Heuristic design

### One-ply countermove

At a node reached after the opponent played move `p`, store a response statistic indexed by `p` and the candidate quiet move `m`. There are two useful variants:

1. **Countermove map:** `CM[p] = m`, storing the single response that most recently or most strongly caused a cutoff. This is the smallest implementation and mirrors the original countermove heuristic.
2. **Countermove history:** `CMH[p][m]`, storing a bounded score for each response. This is more robust because several responses may be good and because a single tactical cutoff should not permanently replace a useful response.

For this engine, the safer research candidate is **CMH with a compact `[piece][to] -> [from][to]` layout**, or equivalently previous-move `[from][to]` with a quiet-move `[from][to]` score. The source's `Move` representation and existing `[from][to]` history make this straightforward conceptually, but table size matters: a full 64×64 previous index times 64×64 response entries is large. A map-only table is tiny; a piece/to-conditioned table is much smaller than full from/to conditioning and is the standard memory-conscious alternative [1][2]. The diagnostic should compare opportunity and collision rates before choosing a layout.

Update only when all of the following hold: the parent node is a non-root interior node; the cutoff move is quiet and non-promotion; the immediately preceding move is valid, quiet, and non-null; the search was not aborted; and the cutoff is a genuine `score >= beta`, not merely a PV improvement. Record the move as a response to the **opponent's move**, with the side-to-move and move encoding unambiguous. Apply a depth-scaled positive update to the best response and a smaller negative update to quiet alternatives already searched at that node. A gravity-style bounded update is preferable to repeated `+=` followed by occasional halving because it prevents stale saturation and gives failed alternatives useful negative information [2]. Exact bonus and malus constants are hypotheses, not repository-validated numbers.

During scoring, apply the CMH contribution only to quiet moves and only when a valid previous move exists. Keep its contribution below the killer tiers initially, or use a separate additive score that cannot displace TT/SEE ordering. A conservative initial ordering ladder is therefore: TT; winning/equal SEE captures and promotions; killer 1/2; countermove contribution; global quiet history. Do not award a special score to an illegal/stale move; ordinary legal generation remains authoritative.

### Two-ply continuation history

Continuation history generalizes the same idea to the move played two plies earlier (the same side's previous move). A candidate score can combine one-ply and two-ply entries, for example:

`quiet_score = global_history + w1 * CMH(previous_opponent_move, candidate) + w2 * FUH(previous_same_side_move, candidate)`.

The two-ply signal may capture recurring plans that one-ply response conditioning misses, but it needs an explicit move stack and multiplies state and update paths. Current Stockfish goes further with multiple continuation plies, but that is evidence of a mature, tuned implementation—not a reason to import its dimensions or weights into this engine [2][3][4]. Start with one-ply CMH. Add two-ply continuation history only if the one-ply diagnostic shows sufficient conditional signal and no prohibitive memory/overhead cost.

Do not use continuation history in qsearch in the first candidate. Qsearch currently has distinct capture/evasion semantics and should remain unchanged. Do not update on null moves, check evasions, captures, promotions, or aborted nodes. In-check nodes are especially poor generic training examples because the legal response set is forced and tactical; excluding them keeps the first table aligned with its quiet-response hypothesis.

## Lowest-risk diagnostic (no tree change)

The first deliverable should be an **instrumentation-only, default-preserving counterfactual diagnostic**. It must not alter move scores, move selection, alpha/beta, TT writes, LMR move counts, or history updates. For each eligible node, compute the hypothetical CMH/continuation score in a side channel, identify where the corresponding move would rank, and compare that prediction with the incumbent's completed result.

Record at least:

* engine commit, evaluator identity/hash, search parameters, hash size, thread count, UCI limits, and deterministic seed/manifest;
* stable position hash or FEN, side to move, nominal depth, PV/non-PV and cut-node type, in-check status, previous move and same-side predecessor when available;
* TT move presence, killer matches, SEE class, global-history value, hypothetical continuation values, and resulting hypothetical rank;
* whether the incumbent raised alpha, failed low, or beta-cut; selected move, score, bound, PV/mate flags, legal move count, and abort status;
* number of quiet moves searched before cutoff, whether the hypothetical response would have been searched earlier, and node/time counts;
* table dimensions, collision policy, update formula, and per-search reset behavior.

Aggregate by depth, node type, previous-move class, capture/check status, and whether TT/killer already supplied the incumbent. The useful quantities are conditional response opportunity rate, exact-match rate, top-k rank improvement, hypothetical earlier-cutoff rate, overlap with killer/global history, table lookup cost, and estimated nodes saved or added. A high match rate alone is not enough: if the response is usually already the TT move or killer, the feature is redundant. Conversely, if the signal is frequent but only changes late quiet ordering, it may still matter through LMR; that effect must be measured rather than assumed.

The diagnostic should use a compact corpus of opening/middlegame/endgame positions and tactical fixtures, with clean and warmed TT runs and at least two hash sizes. Use fixed node limits where available, because wall time can hide tree changes. Repeat each position sufficiently to expose nondeterminism, but do not call the resulting node reduction an Elo result.

## Conditional opt-in candidate

Only if the diagnostic is clean should an implementation create a separate, **default-off** option or compile-time experiment. It should be a one-feature toggle with the incumbent table and all search parameters unchanged. Compare baseline, telemetry-only, and CMH candidate modes under fresh TT and fixed node/time conditions. Keep the first candidate to one-ply quiet CMH, per-search storage, no qsearch use, and no two-ply table.

The candidate must preserve these invariants:

* legal move generation and legality filtering are unchanged;
* TT priority, SEE ordering, check handling, mate-distance handling, and qsearch are unchanged;
* no continuation score can cause a move to bypass search, alter alpha/beta, or change a reduction directly;
* updates occur only after completed, non-aborted nodes and only on well-defined quiet cutoffs;
* PV and root behavior remain semantically unchanged apart from the ordering-induced search tree;
* disabled mode is equivalent to the current code, including table initialization and history reset behavior;
* each Lazy SMP worker has isolated candidate state initially; no new cross-thread races are introduced.

Because ordering changes move counts, LMR can make the candidate a genuine tree change. That is acceptable only as an explicitly isolated experiment. It is not safe to claim “ordering-only” means “strength-neutral.”

## Safety and promotion gates

| Gate | Required evidence | Stop condition |
|---|---|---|
| Build/tests | Existing Rust build and test suite pass; disabled candidate path has no behavior change in focused tests. | Compile failure, panic, race, or disabled-path divergence. |
| Legality | Every returned move is legal; all checking positions retain legal evasions; promotions, en-passant, castling, and pinned-piece fixtures remain correct. | Any illegal move or lost evasion. |
| Mate/tactical | Forced mates preserve sign and mate distance; checks, sacrifices, zwischenzugs, promotions, recaptures, and quiet refutations agree with incumbent at fixed depths. | Any unexplained tactical or mate disagreement. |
| State/update correctness | Previous-move indexing survives root, iterative deepening, copy-make recursion, null move, and abort paths; no stale entry is read as a valid move. | Assertion, out-of-bounds access, stale-context update, or abort contamination. |
| Determinism | Repeated fixed-depth/node-limited runs reproduce the engine's deterministic contract under fixed TT conditions. | Unexplained PV/score/node instability. |
| Cost | Report lookup/update overhead, memory, nodes, wall time, completed depth, and abort rate by category. A broad unexplained node or time increase blocks promotion. | Net cost is non-positive after overhead, or regressions cluster in a category. |
| Search interaction | Quantify changes in LMR re-searches, move-count distributions, TT cutoffs, killer hits, and aspiration outcomes. | A large unexplained interaction with existing pruning/reduction. |
| Strength | Run a fresh, paired, same-binary SPRT with prescribed openings, fixed options, fresh TT policy, and recorded PGNs before any default change. | No default flip from telemetry, fixed positions, or a small ad hoc match. |

The fixed-position corpus should include start position and quiet middlegames; positions where the best move is a late quiet; forced checks/evasions and mates; promotions and en-passant; pinned/discovered attacks; sacrificial tactics; zugzwang and sparse endgames; repetition/50-move boundaries; and positions with multiple plausible responses to the same prior move. Test both sides to move, shallow and deep boundaries, clean/warm TT, and aborted node-limited searches.

A positive diagnostic result means only that a controlled candidate is justified. If response opportunities are rare, are overwhelmingly already covered by TT/killers, or cost more to maintain than they save, **drop or defer the feature**. If one-ply CMH clears all local gates, it may graduate to Tier 2 planning. Any game-facing integration or default change remains Tier 3-style real-SPRT work under the master brief.

## Recommendation

**Pursue, narrowly.** Begin with telemetry-only measurement of one-ply quiet countermove history. Do not implement two-ply continuation history, import Stockfish constants, share mutable tables across Lazy SMP workers, or combine this work with pruning/reduction changes. The feature is a credible low-risk research candidate because it conditions the existing history signal on a concrete predecessor and leaves the incumbent search intact until explicitly enabled. However, no repository measurements yet establish opportunity, net speed, or playing strength, so the current shipped search and defaults should remain unchanged.

## Verification performed

| Check | Status | Meaning |
|---|---|---|
| Master brief reviewed | **Completed** | Assigned item, Tier 1 scope, reporting format, and SPRT boundary applied. |
| Reinforcement docs `00`–`12` reviewed | **Completed** | Existing preservation, provenance, diagnostic, and promotion conventions applied. |
| Adjacent search reports `14`–`18` and NNUE/null-move report `22` reviewed | **Completed** | Existing interaction and safety cautions incorporated. |
| Local `search.rs`/`tt.rs` audit | **Completed** | Verified current TT/SEE/killer/global-history ordering, cutoff updates, path state, and per-search ownership. |
| External literature research | **Completed** | Read Chessprogramming countermove/history references and current official Stockfish history/search source. |
| Implementation | **Not performed** | Explicitly out of scope. |
| Diagnostic/fixed-position sweep | **Not performed** | No instrumentation was added. |
| Benchmark, match, or SPRT | **Not performed** | No strength or speed claim is made. |

## References

[1]: https://chessprogramming.org/Countermove_Heuristic "Chessprogramming Wiki — Countermove Heuristic"
[2]: https://chessprogramming.org/History_Heuristic "Chessprogramming Wiki — History Heuristic, Counter Moves History, and Continuation History"
[3]: https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/history.h "Official Stockfish — history.h"
[4]: https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/search.cpp "Official Stockfish — search.cpp"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/20-countermove-continuation.md`
**No implementation was made.**
