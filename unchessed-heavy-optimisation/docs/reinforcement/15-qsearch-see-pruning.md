# 15 — Qsearch SEE pruning

**Investigation ID:** `tier1-qsearch-see-pruning`
**Scope:** Determine whether qsearch skips clearly losing captures or merely orders them late; distinguish SEE ordering from SEE pruning; research a safe candidate and fixed-position validation plan.
**Status:** Tier 1 research/design only. **No implementation, parameter change, default flip, match, or SPRT was performed.**

## Executive conclusion

The premise in the master brief is now verified: this engine's qsearch does **both**. It uses SEE to order captures/promotions, but it also skips strictly negative-SEE captures before making them. In `unchessed-core/src/search.rs:352-378`, `move_score` assigns winning/equal captures scores in the `1_000_000 + SEE` band and losing captures scores in the `-1_000_000 + SEE` band. In `qsearch` at lines 435–442, a non-checking move with `scores[i] < -1_000_000` is `continue`d. Since a negative SEE value makes that expression strictly less than `-1_000_000`, **captures with SEE < 0 are pruned outright**; equal-SEE captures are not pruned and are searched in the low band. The low score is therefore not merely “searched last.”

This is a real Qsearch SEE-pruning rule, not a missing feature. The sensible recommendation is **defer/drop as a new Tier 1 implementation item**: do not add a second rule or alter the current threshold without diagnostic evidence. If the existing behavior has not yet been independently regression-tested, the next low-risk action is a read-only/counterfactual telemetry experiment and a fixed-position safety suite. Any tree-changing alteration remains Tier 2/3 work and requires the project's real paired-game SPRT gate.

## Repository inspection

The master brief explicitly distinguishes this item from the already-present SEE capture ordering and asks whether losing captures are skipped or only ordered late. It also states that defaults do not move without a paired-game SPRT and that deterministic tests are evidence, not a substitute for playing-strength evidence.

The relevant control flow is:

| Location | Observed behavior | Classification |
|---|---|---|
| `search.rs:352–378` | For captures, en-passant, and promotions, call `see_with_pins`; SEE >= 0 receives `1_000_000 + sc`; SEE < 0 receives `-1_000_000 + sc`. | **Ordering**: good/equal captures before quiets, losing captures behind quiets. |
| `search.rs:380–410` | At non-check nodes, evaluate standing pat, fail high at beta, then generate captures/promotions; in check, generate all evasions and do not use standing pat. | Qsearch shape and safety boundary. |
| `search.rs:435–442` | At non-check nodes, `scores[i] < -1_000_000` causes `continue`. | **Pruning**: strictly negative-SEE captures/promotions are not searched. |
| `search.rs:444–459` | At non-check nodes, delta pruning skips moves whose victim/promotion material plus 180 cp cannot raise alpha. | A separate, already-present qsearch pruning rule. |
| `search.rs:461–470` | Only after the two pruning gates does the engine make the move, check king safety, update NNUE state, and recurse. | Negative-SEE moves are skipped before legality-after-move checking. |
| `see.rs:100–209` | SEE computes a material swap value with least-attacker exchange simulation, static pin masks, and a backward minimax pass. | The predicate used by both ordering and pruning. |

The qsearch comment at `search.rs:412` still says “MVV-LVA ordering,” but the implementation calls SEE. The module-level comments in `see.rs` correctly describe SEE as the basis for capture ordering and qsearch pruning. This is documentation drift, not evidence that qsearch uses MVV-LVA.

The implementation has important boundaries. SEE pruning is disabled while in check, where qsearch searches all evasions and mate detection requires full generation. The engine also preserves promotions in the capture/promotion path. Delta pruning is separately disabled in check. The current rule is not configurable and has no pruning counter, so this inspection verifies source behavior but cannot quantify its hit rate or node savings.

### SEE is a heuristic predicate, not a proof of the whole position

`see_with_pins` evaluates the material exchange on one destination square. It uses a static pin scan and deliberately does not recompute pins after every simulated exchange; the source documents this as a speed/accuracy tradeoff matching a known Stockfish simplification. SEE therefore answers “what does this local swap likely net?” It does not fully model zwischenzug moves, positional compensation, king attacks outside the exchange square, promotion tactics, mating threats, or the possibility that declining a recapture changes the strategic result. That limitation is exactly why the safety screen must include sacrifices, checks, promotions, x-rays, pins, and mating positions rather than only ordinary defended captures.

## Ordering versus pruning

These mechanisms have different correctness and performance implications:

* **SEE ordering** changes the sequence in which legal moves are searched. It can improve alpha-beta cutoffs while preserving the set of candidate moves. A losing capture can still be searched if all earlier moves fail to produce a cutoff.
* **SEE pruning** removes a move from the qsearch tree. A negative local exchange is treated as unable to improve the tactical result. It can reduce nodes substantially, but a false negative can change the returned score, PV, mate result, or root move.
* **Delta pruning** is neither of the above. It uses the current bound, stand-pat/best score, victim value, promotion gain, and a safety margin to reject a capture that cannot plausibly raise alpha. The existing `+180` margin means the engine already has a second qsearch selectivity gate, and interactions must be measured separately.

The current code's sequence matters: SEE pruning and delta pruning happen before `king_safe_after`, so a candidate rule cannot assume legality checks will rescue an incorrectly pruned tactical move. Conversely, in-check handling is fail-safe because the SEE and delta skips are explicitly bypassed.

## Literature and external evidence

The [Chessprogramming Wiki's Quiescence Search overview](https://chessprogramming.org/Quiescence_Search) explains that qsearch exists to avoid the horizon effect and normally searches a restricted tactical set. Its “Limiting Quiescence” section specifically identifies not trying captures with static exchange evaluation below zero as a qsearch pruning technique, while separately emphasizing capture ordering to control search explosion. Its “Checks” section supports searching all evasions when in check and not using stand-pat there. The page also lists Harris (1975), Kaindl (1982), Beal (1984/1990), Bettadapur (1986), and Bettadapur & Marsland (1988) as historical quiescence/capture-search literature.

The [Static Exchange Evaluation reference](https://chessprogramming.org/Static_Exchange_Evaluation) defines SEE as the likely material result of exchanges on one square. It explicitly separates “Move Ordering” (good and bad captures in different search bands) from “Pruning,” and notes SEE's use in qsearch together with delta pruning. Its didactic swap algorithm uses a max-with-zero continuation choice, illustrating why a losing exchange should generally not be voluntarily continued in a pure capture search, while also making clear that SEE is a static exchange model rather than a full search.

The [Delta Pruning reference](https://chessprogramming.org/Delta_Pruning) describes qsearch delta pruning as testing whether the captured value plus a safety margin can raise alpha; it gives a typical margin around 200 cp and warns that endgame/insufficient-material cases need care. This supports treating the current SEE and delta gates as distinct and testing their interaction rather than assuming additive benefit.

The [official Stockfish terminology documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html) defines qSearch as the tactical search at the end of main search, move ordering as trying likely-best moves first, and pruning as ignoring parts of the tree. Current Stockfish source (`src/search.cpp`, accessed from [the official repository](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)) contains separate SEE-based pruning for captures/checks in main search and a qsearch move-pruning pipeline alongside futility and move-count pruning. This is evidence that modern engines distinguish the concepts; it is not a justification for importing Stockfish constants into this engine.

## Safe candidate and measurement plan (design only)

Because the current rule is already present, the safe candidate should not be “enable SEE pruning.” The only defensible future candidates are narrowly scoped comparisons such as:

1. **Incumbent:** retain the current strict `SEE < 0` qsearch skip.
2. **Control:** order losing captures late but search them, isolating pruning from ordering.
3. **Conservative threshold arm:** retain only a small negative threshold (for example, prune `SEE < -T`), but calibrate `T` from this engine's own fixed-position evidence rather than importing a Stockfish value.
4. **Interaction arms:** test incumbent SEE pruning with current delta pruning separately from a combined change; do not attribute savings to SEE when delta pruning is doing the work.

A diagnostic build should count generated captures/promotions, negative/equal/nonnegative SEE classifications, SEE skips, delta skips, legal moves reached, qsearch nodes, and score/PV/mate differences. It should also record whether the skipped move was a checking move, a promotion, an en-passant move, pinned, or a recapture. Counters should be diagnostic-only and default-off. A useful result is net nodes after SEE computation, not just the number of `continue`s.

Run fresh-TT fixed-position comparisons at several shallow and moderate depths with identical evaluator, options, hash, and commit. Record best move, score, mate distance, PV, completed iteration, node count, qsearch node count, and all skip counters. Flag any tactical score/PV/mate regression as a reject for that candidate. Agreement on a small suite is a safety screen, not an Elo claim.

## Fixed-position safety suite

The repository already has focused SEE tests in `unchessed-core/src/see.rs` that are good starting fixtures: an undefended rook-takes-knight gain; a rook capture recaptured by a pawn; a pinned recapturer; and a three-ply x-ray exchange. They verify SEE arithmetic, not qsearch behavior, so the qsearch harness must place each position behind a controlled search call and compare incumbent/control results.

The additional fixed-position corpus should include:

| Class | Required coverage | Failure to watch for |
|---|---|---|
| Plain losing captures | QxP defended by a pawn; rook takes a defended minor; equal exchanges. | Candidate confuses ordering with pruning or fails to skip only strict negatives. |
| Tactical sacrifices | Sacrificial captures that open a file/diagonal, deflect a defender, or force a queen win next ply. | SEE-negative local swap is globally best. |
| Checks and evasions | Side in check with capture and non-capture evasions; checking captures at qsearch root and descendants. | In-check move gets pruned or stand-pat is used. |
| Promotions | Capturing promotions, non-capturing promotions, underpromotions, and a promotion whose immediate SEE is misleading. | Promotion piece gain/SEE and qsearch generation disagree. |
| En passant | EP capture with unusual occupancy/x-ray effects. | Victim-square/value calculation differs from SEE or legality. |
| Pins and x-rays | Pinned recapturer, discovered attacker, long slider exchange, king-adjacent captures. | Static pin approximation causes a false negative or illegal line. |
| Mate and king safety | Captures delivering mate, avoiding mate, and positions where a losing capture is the only defense. | Pruning changes mate score or misses the only legal defense. |
| Endgames | Sparse material, pawn races, insufficient-material transitions, 50-move-sensitive positions if supported by the harness. | Material-only SEE/delta assumptions hide a decisive promotion or draw issue. |
| Boundaries | SEE exactly 0; SEE -1; alpha just reachable/unreachable under current `+180` delta; beta in mate range. | Off-by-one threshold or interaction with delta pruning. |

The strongest fixed-position tests are not just “best move stayed equal.” They should assert that forced mates remain mates with the same distance, legal evasions remain available, and a control that searches losing captures can expose any score/PV discrepancy. Include both PV and non-PV/null-window contexts because pruning is most consequential at narrow bounds.

## Verification performed

| Check | Result |
|---|---|
| Master brief | **Read**; scope, existing SEE ordering, standing SPRT rule, and Tier 1/Tier 2 boundary applied. |
| Reinforcement docs 00–12 | **Reviewed** for numbering, evidence conventions, fixed-position safety expectations, and the no-default-flip/real-SPRT rule. |
| Qsearch source inspection | **Completed**; strict negative-SEE captures/promotions are skipped at `search.rs:435–442`; equal SEE is retained. |
| SEE source/tests inspection | **Completed**; `see_with_pins` and four focused exchange tests reviewed. |
| Web research | **Completed** using the Chessprogramming Wiki qsearch, SEE, and delta-pruning references; official Stockfish terminology and source were also read. |
| Focused Rust test execution | **Blocked before compilation**: installed Cargo/rustc is 1.75.0 while this checkout's `Cargo.lock` is version 4 and Cargo reports `requires -Znext-lockfile-bump`. Existing test code was inspected but no passing-test claim is made. |
| Implementation | **Not run / not made.** |
| Fixed-position qsearch sweep | **Not run**; no diagnostic toggle/harness was added under this research-only task. |
| Cutechess match/SPRT | **Not run**; required for any playing-strength or default decision. |

## Recommendation and gate

**Recommendation: drop the item as a missing-feature proposal; defer any threshold or rule change.** Qsearch SEE pruning is already implemented, alongside SEE ordering and delta pruning. The primary actionable issue is documentation clarity: rename the stale qsearch ordering comment from MVV-LVA to SEE and, if this area is revisited, add counters and focused qsearch regression tests. Those clarifications do not change behavior.

If a future experiment proposes less or more aggressive pruning, graduate it only after the fixed-position screen shows no tactical, mate, legality, or endgame regression and after a fresh-TT node study demonstrates net savings after SEE cost. Keep the candidate isolated from other pruning changes. A surviving candidate still requires a real paired-game cutechess SPRT before any default or playing-strength conclusion; fixed positions and node counts alone cannot establish Elo.

## References

[1]: https://chessprogramming.org/Quiescence_Search "Chessprogramming Wiki — Quiescence Search"
[2]: https://chessprogramming.org/Static_Exchange_Evaluation "Chessprogramming Wiki — Static Exchange Evaluation"
[3]: https://chessprogramming.org/Delta_Pruning "Chessprogramming Wiki — Delta Pruning"
[4]: https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html "Official Stockfish Docs — Terminology"
[5]: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp "Official Stockfish — search.cpp"

**No implementation was made.**
