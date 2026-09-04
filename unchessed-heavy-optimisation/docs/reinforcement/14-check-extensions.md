# 14 — Check extensions: verification, additive value, and the cheapest safety diagnostic

**Investigation ID:** `tier1-check-extensions`
**Tier:** 1 (research/design only)
**Status:** Complete; no implementation, default change, tuning run, or game match was performed.

## Executive finding

The master brief’s statement that check extensions are “confirmed absent entirely” is **stale for this branch**. The current search already contains two check-related depth safeguards:

1. At negamax entry, a node in check is forced to search at least one full ply: `depth = if in_chk { depth.max(1) } else { depth }` (`unchessed-core/src/search.rs:525–529`). This is the standard check-evasion safety behavior, although it is expressed as a floor rather than as an explicit per-move extension.
2. After a legal move is made, a move that gives check receives `ext = 1`, and the child depth is `depth - 1 + ext` (`search.rs:705–719`). This is an explicit **one-ply checking-move extension**.

Therefore, implementing another generic “check extension” would duplicate existing behavior and is not justified as a Tier 1 candidate. The useful next step is a cheap **fixed-position safety audit** of the existing mechanism, particularly the interaction with pruning, LMR, quiescence, mate scores, and node budgets. The honest recommendation is **defer new implementation; verify and instrument only if a later owner approves a correctness/observability change**. Any changed gate or extension policy remains subject to a real paired-game SPRT before promotion.

## What I inspected and what was verified

I read the master brief, the existing reinforcement reports `00` through `12`, the search implementation, move generation, and the prior search/performance notes. The repository is on branch `manus/research-facilities`. No source file was modified other than this report.

| Question | Verified result |
|---|---|
| Is there an explicit checking-move extension? | **Yes.** `in_check(&next)` is computed after legality checking; `ext` is `1` when the move gives check, and added to child depth. |
| Is there an evasion safeguard? | **Yes.** An in-check node is raised to at least depth 1 before the depth-zero qsearch transition. |
| Are check extensions applied before move-level pruning? | The `gives_check` flag is computed before the per-move futility/SEE gates. Plain futility explicitly excludes checking moves (`!gives_check`); the capture/check branch has separate handling. |
| Can LMR reduce a checking move? | The LMR condition explicitly excludes `gives_check`; a checking move is not reduced by that branch. |
| Does NMP/RFP avoid checked nodes? | Both null-move and reverse-futility paths are gated by `!in_chk`. |
| Does qsearch generate quiet checking moves? | `generate(pos, true, ...)` is documented as “captures + promotions.” Thus qsearch does not generally generate quiet checks. However, negamax forces in-check nodes to depth at least one before entering qsearch. This boundary must be tested rather than assumed safe for every caller. |
| Are singular/recapture extensions present? | No corresponding implementation was found in the inspected search path. The master brief’s claim that these are absent remains consistent with the code search. |
| Was a strength result obtained? | **No.** No SPRT, self-play match, or tuning campaign was run. |

The prior `01-search.md` report already documented check extensions as present and cited `search.rs:483–825`; the repository’s own research note `docs/research-notes-vrzina-engine-thesis.md` also records “+1 when giving check (`search.rs:709).” These existing documents conflict with the “absent” sentence in the master brief. The source code is the controlling evidence.

## Terminology and literature context

A check extension is selective search depth: extend a forcing checking line, commonly by one ply, so the search does not stop immediately before the forced reply or tactical consequence. The established distinction is between extending a move that **gives check** and ensuring a side that is **evading check** is searched deeply enough. The Chessprogramming Wiki describes both forms and gives one ply as the typical extension, while warning that selective extensions can cause search explosion and that some engines reduce or reject unsound checks [1].

The motivation is plausible here. Checks constrain replies, are tactically forcing, and can expose horizon-effect errors. The current engine already has substantial selectivity in the opposite direction: reverse futility pruning, null-move pruning, LMR, ProbCut, plain futility, SEE-based capture pruning/order, and aspiration windows. A check extension can be additive because its exclusions protect the very tactical moves most likely to be harmed by those reductions/pruning rules. In particular, this code avoids RFP/NMP while in check, avoids plain futility on checking moves, and avoids LMR on checking moves.

The literature does **not** support assuming that an extension is universally additive. Beal and Smith evaluated extension rules on a fixed position set and measured node counts; their abstract reports that some rules were strongly advantageous in isolation but disadvantageous in combination, notably singular extensions becoming detrimental when check extensions, recaptures, and null moves were already present [2]. This directly supports an incumbent-protected fixed-position comparison rather than stacking another extension. Anantharaman’s earlier *Extension Heuristics* paper describes implemented extension heuristics and experimental evaluation in Deep Thought [3]. Historical engine reports likewise use check evasion and other forcing-line extensions, but those results are not transferable Elo evidence for this engine.

The relevant conclusion for this repository is narrower: **the policy category is standard, but its marginal value depends on the existing tree and evaluator**. Since the branch already has the basic policy, the research question is not “should we add check extensions?” but “are the existing extensions correctly bounded and are there tactical positions where their current form fails?”

## Interaction with the current search stack

### RFP and null move

Both reverse futility and null move are intentionally disabled while the side to move is in check. This is essential: checking positions are forcing and the null-move assumption is invalid or unsafe there. The existing in-check depth floor then ensures a checked node cannot immediately disappear into ordinary qsearch.

### LMR and futility

The explicit checking move is excluded from the LMR condition. Plain futility also excludes checking moves, while captures/checks enter a separate branch with SEE-based capture/check pruning. This is a sensible safety shape, but it creates a measurable cost: a checking move can retain full depth and an additional ply even when it is a late or losing check. SEE-based pruning may still discard a losing check under its margin. That is not automatically wrong—many checks are unsound sacrifices—but it is exactly why a diagnostic must include sacrificial checks and mate-threat positions, not only clean mating puzzles.

### ProbCut and aspiration

ProbCut can return early above beta before the ordinary move loop. A check extension cannot repair a position that ProbCut has already cut, so tactical safety testing must include positions where a shallow tactical check competes with the ProbCut threshold. Aspiration windows can trigger re-searches when a forcing line’s score falls outside the window; extensions may improve the final line but increase fail-high/fail-low work. Node counts and completed-depth behavior must therefore be recorded under both wide and normal aspiration settings if telemetry is later added.

### Quiescence

Quiescence is capture/promotion based and applies SEE and delta pruning outside check. It does not enumerate quiet checks by design. The negamax entry guard (`depth.max(1)` while in check) is the compensating safety mechanism. A future change must not remove that guard unless qsearch is changed to search all legal evasions. The cheapest audit should explicitly test checked positions with quiet king moves, interpositions, and captures, because a qsearch-only entry would otherwise have an incomplete evasion set.

### Lazy SMP and transposition tables

The extension is local search state, not shared mutable state, so it should be naturally thread-safe. Nevertheless, differing depths and TT entries can make extension behavior appear inconsistent across Lazy SMP helpers. The fixed-position diagnostic should use a clean TT for each run and compare single-thread and configured multi-thread execution only as a determinism/safety check, not as a strength claim.

## Cheapest fixed-position safety diagnostic (design only)

The lowest-cost diagnostic is a deterministic Rust test or an opt-in offline harness around the existing `go` entry point. It should not add a UCI default, alter search behavior, or launch games. Use `Limits::depth(D)` at small depths (for example, 3–8), a clean TT per case, HCE first and the shipped NNUE second if the test environment can load it, and a fixed `SearchParams` snapshot.

Construct a compact FEN suite with labelled categories:

| Category | Required fixtures |
|---|---|
| Mate finding | Mate in 1–3, both colors, checking move at the nominal horizon, and a position with a quiet mate threat. |
| Check evasion | Single legal evasion, multiple king moves, capture of the checker, interposition, double check, and a position with no legal evasion (mate). |
| Checking sacrifices | A winning check with negative immediate SEE, a losing “spite check,” and a check where the only win is to decline the check. |
| Horizon/tactical boundary | A forcing check followed by a forced recapture or promotion; positions where a shallow check appears attractive but fails one reply deeper. |
| Pruning boundaries | Non-PV and cut-node positions near RFP/NMP/ProbCut/futility thresholds, with and without check. |
| Regression controls | Start position, a quiet middlegame, an endgame with no checks, and existing mate tests from `search.rs`. |

For each FEN, run two conceptual modes: **current policy** (existing check extension and in-check floor) and a diagnostic baseline with the extension contribution disabled only in the harness or a temporary local experiment. Because no implementation is requested here, this report specifies the comparison rather than performing it. Both modes must use the same legal move generator, evaluator, TT policy, depth, and node limit. If a clean toggle cannot be introduced without changing code, compare current fixed-depth results against an independent reference/perft/mate oracle and report only current-policy safety; do not simulate a false baseline.

Record, per position and depth: best move, score, mate distance where applicable, PV, completed iteration depth, node count, abort status, and whether the expected legal evasion/mate result was found. The safety output should be machine-readable and include the exact FEN, parameter snapshot, evaluator identity/hash, engine commit, and TT size. A useful first screen is 20–50 positions, not a large corpus.

## Gates and stop conditions

**Gate 0 — repository correctness.** The existing focused search tests and the three named NNUE parity tests must pass in a toolchain that can parse the current lockfile. In this sandbox, `cargo test -p unchessed-core --release --lib --no-default-features` was attempted but blocked before compilation because the installed Cargo rejected lockfile version 4 (`-Znext-lockfile-bump`); this is an environment/toolchain limitation, not a test pass. Do not report it as passing.

**Gate 1 — legality and mate safety.** Across every fixture and depth, no legal move may disappear; every checked node must either find a legal evasion or return mate; known mate positions must retain the correct mate sign and distance. Any regression is an immediate stop.

**Gate 2 — deterministic incumbent behavior.** Repeating a fixed-depth run with a clean TT must reproduce the same result, PV, and node count within the engine’s documented deterministic mode. HCE and NNUE runs must not produce numerical crashes or invalid mate scores. Multi-thread checks are supplementary and must not be used to excuse single-thread nondeterminism.

**Gate 3 — bounded cost.** Measure overhead rather than selecting an arbitrary “good” node count. A candidate policy should show a tactical benefit without an unexplained broad node explosion; as a provisional diagnostic guard, flag any category with more than 25% node growth and require explicit review. This is a review threshold, not a proven universal limit and not a promotion criterion.

**Gate 4 — incumbent-protected game decision.** Fixed positions, mate suites, perft, unit tests, and node counts can justify a safety decision or reject a broken candidate; they cannot establish Elo. Any change to extension eligibility, depth, interaction with pruning, or defaults requires the project’s real paired-game cutechess SPRT and must preserve all defaults until that result completes. The master brief separately classifies search-extension campaigns as Tier 3 work requiring explicit approval.

## Recommendation

**Do not implement a second check-extension mechanism.** Mark the master brief’s “absent entirely” claim as corrected: checking moves already receive a one-ply extension, and checked nodes already receive an at-least-one-ply evasion floor. Research value remains in a cheap, default-preserving safety audit, not in adding a redundant feature. If an owner later wants a code change, the only reasonable Tier 2 candidate is an opt-in diagnostic toggle/telemetry path that proves the current tree and boundaries; it is not permission to flip behavior. Do not pursue singular or recapture extensions as part of this item: the cited literature warns about interaction effects, and those are separate candidates requiring their own evidence.

## Verified versus assumed

**Verified:** the source locations and conditions listed above; current check-giving extension; in-check depth floor; LMR/futility/RFP/NMP exclusions; capture/promotion-only qsearch generation; prior reports’ conflicting “already present” documentation; absence of a completed test run in this sandbox due to Cargo lockfile incompatibility; no implementation or SPRT performed.

**Assumed/design-only:** that the current extension is strength-positive; that its node overhead is acceptable; that every qsearch caller is protected by the in-check depth floor; that Lazy SMP gives equivalent tactical outcomes; and that the proposed 20–50-position suite predicts game strength. None of these assumptions supports a default change.

## References

[1]: https://chessprogramming.org/Check_Extensions "Chessprogramming Wiki — Check Extensions"
[2]: https://doi.org/10.3233/ICG-1995-18403 "D. F. Beal and M. C. Smith, Quantification of Search-Extension Benefits, ICGA Journal 18(4), 1995"
[3]: https://doi.org/10.3233/ICG-1991-14202 "T. S. Anantharaman, Extension Heuristics, ICGA Journal 14(2), 1991"
[4]: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/blob/manus/research-facilities/unchessed-core/src/search.rs "Repository search implementation"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/14-check-extensions.md`

**No implementation was made.**
