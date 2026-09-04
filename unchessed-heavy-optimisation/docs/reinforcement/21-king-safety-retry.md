# 21 — Validated king-safety retry

**Investigation ID:** `21-king-safety-retry`
**Scope:** One Tier 1 item: determine whether king safety is worth retrying from validated comparable-engine magnitudes, rather than repeating the repository's blind SPSA attempt.
**Repository:** `/home/ubuntu/Unchessed-UCI-Engine`, branch `manus/research-facilities`
**Status:** Research/design only. No source change, evaluator change, SPSA run, fixed-position campaign, match, or SPRT was performed.

## Decision

> **Recommendation: pursue conditionally, but do not start Tier 2 yet.**

A king-safety retry is technically plausible, but the evidence supports only a narrowly scoped, provenance-first diagnostic. Comparable engines show that king safety is not normally one arbitrary scalar: it is a gated, nonlinear combination of shelter/storm, king-zone attacks, safe checks, defenders, weak squares, and material/phase scaling. The repository should therefore not add a guessed “king safety bonus,” copy one foreign constant, or rerun the failed SPSA harness. The next decision requires an implementation whose feature semantics and unit conversion are explicitly matched to a validated reference, followed by deterministic safety and activation evidence. Only after those gates pass may a Tier 2 candidate be planned; a real paired-game SPRT remains mandatory before any default or strength conclusion.

The result is deliberately narrower than “king safety improves chess.” The literature establishes mature design patterns and real parameter magnitudes in other engines. It does **not** establish that those magnitudes transfer to Unchessed, whose shipped default is NNUE and whose hand-crafted evaluation is a fallback path.

## What was inspected

I read the master brief, the existing reinforcement reports `00`–`12` where present, the current evaluation/search/UCI code, and `docs/parameter-calibration-audit.md`. The repository audit records that the prior king-safety attempt was reverted after a real SPRT failure and states the resulting discipline plainly: this engine's SPSA harness could not discover a term's magnitude from scratch; future work must start from validated real numbers. The reinforcement synthesis and search-bandit report add the required safeguards: exact node-limit behavior and default-off search telemetry should precede tuning, and no game-facing candidate is promoted without a fresh paired-game SPRT.

The current `unchessed-core/src/eval.rs` contains material, pawn, mobility, rook-file/7th, passed-pawn, and knight-outpost HCE terms, with `EvalParams` percentages. A repository-wide search found no implemented HCE king-safety/shelter/storm/king-zone term. The default UCI path loads the shipped NNUE; the HCE percentages are therefore inert in normal NNUE evaluation. This is a crucial scope fact: an HCE king-safety patch would not affect default play unless the evaluator architecture is deliberately changed, and changing that path would require a separate, explicit candidate definition.

No files were modified other than this report. The worktree already contained unrelated changes and untracked reports; none were altered.

## Evidence from comparable engines

### Stockfish: attack-unit danger is a nonlinear, gated score

The Stockfish 15 source provides a concrete, authoritative reference implementation, rather than a folklore scalar. It defines piece-type king-ring attack weights of **knight 76, bishop 46, rook 45, queen 14** (Stockfish internal score units), safe-check tables including, for example, rook **805/1292**, queen **730/1128**, bishop **650/984**, and knight **1071/1886** for one/multiple checks, and then accumulates several additional danger terms. The danger expression includes attack-weight count, weak king-ring squares at **183** each, unsafe checks at **148** each, king blockers at **98** each, adjacent king attacks at **69** each, a quadratic king-flank attack term, a **−873** no-enemy-queen adjustment, a **−100** own knight/king shelter-related adjustment, a shelter-score interaction, and a small constant. These are not independent centipawn bonuses: they feed Stockfish's later nonlinear king-danger/safety scaling and its own internal score representation [1].

The operational lesson is more important than copying the numbers. Stockfish counts attacks in a king ring, counts multiple attacked squares, distinguishes safe from unsafe checks, adds blockers and weak squares, and applies interaction/scaling. A direct port of `76` or `183` into an HCE centipawn term would be a unit error and would omit the gates that make the reference meaningful.

### Ethereal: independently tuned shelter/storm and safety magnitudes

Ethereal's public evaluation source gives a second, independent reference with explicit middlegame/endgame score pairs. Its king evaluation includes `KingDefenders`, `KingPawnFileProximity`, large square/file/rank-indexed `KingShelter` and `KingStorm` tables, and safety weights **knight 48/41, bishop 24/35, rook 36/8, queen 30/6**. It also defines `SafetyAttackValue 45/34`, `SafetyWeakSquares 42/41`, `SafetyNoEnemyQueens −237/−259`, safe-check terms **queen 93/83, rook 90/98, bishop 59/59, knight 112/117**, and `SafetyAdjustment −74/−26` [2]. Here `S(mg,eg)` is an Ethereal score pair, not a license to treat every entry as a flat centipawn term. The tables are combined through Ethereal's king-safety computation and phase behavior.

This is useful evidence of **magnitude class and structure**: individual safety components are commonly tens of evaluation units, safe checks can be roughly 59–117 units, no-queen/shelter interactions can reach roughly 200–260 units, and indexed shelter/storm entries can be larger in particular squares. It is not evidence that an Unchessed value of `45` or `−237` is correct.

### Historical Glaurung/Stockfish-style safety curve: attack count saturates

The ChessProgramming Wiki's king-safety survey documents a representative attack-unit design: minor-piece attacks count as **2** units, rook attacks as **3**, queen attacks as **5**, with safe queen contact checks adding **6** and safe rook checks a smaller amount. It reproduces a Glaurung 1.2 safety table that rises from zero through an S-shaped curve and saturates near **650**, and a Stockfish-derived rescaled table that saturates near **500** [3]. The same source describes a sample square-attack scheme using **20/20/40/80** weights for knight/bishop/rook/queen attacked squares and an attacker-count multiplier of **0, 50, 75, 88, 94, 97, 99 percent** for one through seven attackers.

Those tables are historical explanatory material, not current Stockfish defaults. They nevertheless establish why a single linear term is a poor model: one attacker is often intentionally worth little, several coordinated attackers are worth disproportionately more, and the danger is capped. A retry must choose one reference model and reproduce its gates, not mix the historical table with modern constants.

### Tuning evidence: king safety can be tuned, but only with a suitable objective

In Andrew Grant's account of Ethereal tuning, he reports that linear evaluation tuning produced approximately **+10 Elo**, a king-safety-specific patch approximately **+3.4 Elo**, and a combined linear/safety/complexity experiment approximately **+2.3 Elo**. He attributes the method to an AdaGrad/Texel-style differentiable treatment of nonlinear safety and complexity, not to a tiny game batch blindly perturbing a scalar [4]. This is a report by the engine author and a useful methodological reference, not a transferable Unchessed result. It supports the narrower conclusion that nonlinear safety can be tuned when its derivatives/data/objective are handled deliberately; it does not justify a new default here.

## Contrast with the repository's failed blind SPSA attempt

The repository's `docs/reinforcement/09-search-bandit.md` and calibration audit describe the historical failure. The checked-in SPSA scripts used `3+0.03`, only **12 games per match**, and either 40 or 200 iterations. A two-sided 200-iteration campaign would nominally require **4,800 games** before confirmation, yet the available log stops at iteration 93 and is not a completed campaign. Early 12-game scores range through **0.333, 0.375, 0.500, and 0.625**; the vector stays effectively near `[50.0, 50.0]` and later moves only around `[49.9, 50.1]`. Absolute paths for the engine, book, cutechess, and output are not available in this sandbox. There is no completed SPRT, stable optimizer trajectory, or reproducible strength conclusion.

The failure is especially unsuitable as a king-safety calibration method for five reasons:

1. **The coordinate was not anchored to a validated semantic magnitude.** A foreign engine's attack units, an Ethereal score pair, and Unchessed HCE centipawns are different coordinate systems. SPSA cannot repair an undefined feature/unit mapping.
2. **Twelve games cannot resolve a small gradient.** The observed win rates are compatible with large sampling noise; a plus/minus result is not a measured derivative at that sample size.
3. **The objective was discontinuous and gated.** King safety changes only in positions where shelter, attack-zone, safe-check, or attacker-count predicates activate. A scalar perturbation can appear inert in most games and then cross a tactical threshold.
4. **The proposed vector lacked activation telemetry.** Without counts for king-ring attacks, safe checks, shelter/storm states, scaling, and score contribution, one cannot tell whether a result reflects king safety or a collateral change in move choice/tree shape.
5. **There was no incumbent protection or independent confirmation.** A noisy plus/minus winner could become the next baseline. The project policy requires a real paired-game SPRT, not an optimizer's interim score.

Therefore the correct interpretation is **“the blind protocol failed to produce evidence,” not “king safety is disproven.”** Conversely, comparable-engine magnitudes are priors, not proof that a new Unchessed implementation will help.

## Proposed validated retry (Tier 1 boundary)

The retry should remain a design and evidence exercise until the following choices are frozen. The first implementation candidate should be the smallest coherent reference model, not a blend of terms:

| Design decision | Required treatment | Prohibited shortcut |
|---|---|---|
| Reference | Select either the Stockfish attack-unit/king-danger structure or the Ethereal shelter/storm/safety structure and cite the exact source revision. | Mixing Stockfish attack weights with Ethereal shelter tables and historical Glaurung scaling. |
| Units | Declare the source unit, phase interpolation, and conversion to Unchessed HCE score units. Calibrate conversion from a known reference position set, not from game outcomes. | Copying `76`, `183`, `45`, or `−237` as Unchessed centipawns. |
| Scope | Start with one coherent mechanism, preferably king-ring attacks plus safe-check/attacker gating or shelter/storm, not every king feature at once. | A single flat “king safety” scalar or a multi-term bundle with no attribution. |
| Activation | Define legal king ring, attacker counting, blocked/weak squares, safe-check legality, castling/uncastled handling, side-to-move perspective, phase, and material gates before coding. | Informal proximity or “pieces near king” without attack legality and blockers. |
| Evaluator path | Prove whether the candidate is HCE-only and therefore default-inert under NNUE, or explicitly define a separate NNUE integration hypothesis. | Claiming default strength from an HCE-only term while the shipped default is NNUE. |
| Safety | Keep mate scores, in-check handling, sign conventions, and score clamps unchanged; add targeted tests for castling, missing shield, sacrificial attack, queenless positions, and endgame scaling. | Allowing king safety to override legal/tactical correctness or mate handling. |

A default-off, diagnostic-only implementation can be considered Tier 1 only after its source provenance and unit mapping are recorded. It should expose a trace or offline report with per-component contributions and activation counts. Existing default-off search statistics should be extended only if that can be done without changing the disabled path; otherwise use a separate fixed-position evaluator harness. No UCI default should move.

## Evidence required before Tier 2

The following gates are **necessary evidence**, not work performed in this report:

1. **Reference and provenance gate.** Pin the exact Stockfish/Ethereal source revision, license-compatible excerpts or reimplementation, feature definitions, source units, phase/scaling formula, and a conversion note. A reviewer must be able to recompute every imported magnitude.
2. **State/legality gate.** Build a deterministic FEN/EPD fixture corpus covering castled and uncastled kings, intact and broken shields, pawn storms, open/semi-open files, one versus multiple attackers, safe and unsafe checks, queenless positions, opposite-side attacks, and sparse endgames. Verify attack maps against an independent legal reference. Include side-to-move and color symmetry cases.
3. **Numerical/parity gate.** For each fixture, compare per-component output and total score against an independently calculated reference implementation within a declared integer tolerance. Test monotonic sanity properties only where the reference model guarantees them; do not assume every pawn move should monotonically worsen safety.
4. **Default-preservation gate.** With the candidate disabled, fixed-depth searches must retain line, score, and node totals under fresh TT states for both HCE and the shipped NNUE path. If the term is HCE-only, explicitly record that normal NNUE play is unchanged.
5. **Activation/telemetry gate.** On a fixed corpus, report counts and distributions for each feature, phase, material gate, score contribution, and saturation/clamp event. Repeat the incumbent and candidate with identical settings to establish deterministic output. A candidate with near-zero activation or extreme saturation is rejected or re-scoped before games.
6. **Tactical safety gate.** Existing mate-track, legality, check, repetition, and search tests must pass. Add reject-only fixtures for back-rank mate, sacrificial attacks, checking moves, queen trades, and king moves. No fixed-position pass is an Elo claim.
7. **Candidate-isolation gate.** Change one coherent mechanism and one declared unit conversion at a time. Hold binary, evaluator/model, thread count, hash, book, hardware, and time control fixed. Do not tune a 13-dimensional search vector alongside king safety.
8. **Real-game gate.** Only after the above pass may a Tier 2 plan specify a small reject-only paired-opening screen and then a fresh real paired-game SPRT using the repository convention (`Threads=1`, fixed hash/book/evaluator, `5+0.05`, `elo0=0`, `elo1=5`, α=β=0.05, raw PGN/log and hashes retained). An inconclusive or negative result keeps the default unchanged; no early point estimate is a promotion.

Tier 2 must not begin merely because the code compiles, the fixtures look plausible, the term resembles Stockfish, or a fixed-position score changes. It begins only when the reference, unit, activation, default-preservation, and safety evidence is complete and a posted plan is acknowledged under the master brief's credit rule.

## Verified versus assumed

| Category | Finding |
|---|---|
| **Verified in this investigation** | Master brief and repository documents/code were read; current branch is `manus/research-facilities`; no HCE king-safety term was found in the inspected evaluation code; current HCE term parameters are separate from the shipped NNUE path; the prior SPSA details and incomplete log are recorded in repository reports; comparable source constants and formulas were read from the cited public sources. |
| **Verified from repository records, not rerun here** | The prior king-safety attempt failed its real SPRT; the SPSA run was underpowered/incomplete; current search-tuning evidence standards require telemetry, incumbent protection, and a fresh paired-game SPRT. |
| **Assumed/design-only** | Any proposed reference selection, unit conversion, fixture set, telemetry schema, and candidate screen. No implementation or measurement validates them yet. |
| **Not run** | No SPSA, source modification, compile/test run for this item, evaluator benchmark, fixed-position screen, cutechess match, or SPRT. |

## Conclusion

King safety merits a **validated retry as a research question**, not an immediate feature or tuning campaign. Stockfish, Ethereal, and the documented Glaurung/Stockfish family show real, bounded magnitudes and nonlinear interactions, while Ethereal's reported tuning result shows that a properly modelled safety function can produce a measurable but modest gain. None of those results supplies an Unchessed constant. The failed blind SPSA attempt lacked the semantic anchoring, replication, activation observability, and confirmation controls needed to answer this question. Preserve all current defaults, implement no arbitrary term, and require the evidence gates above before any Tier 2 plan or game-facing work.

## References

[1]: https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_15/src/evaluate.cpp "Stockfish 15 evaluate.cpp (official source)"

[2]: https://raw.githubusercontent.com/AndyGrant/Ethereal/master/src/evaluate.c "Ethereal evaluate.c (public engine source)"

[3]: https://chessprogramming.org/King_Safety "ChessProgramming Wiki — King Safety"

[4]: https://talkchess.com/viewtopic.php?t=74877 "Andrew Grant, Evaluation & Tuning in Chess Engines (TalkChess)"

[5]: https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html "Stockfish Fishtest Mathematics"

[6]: https://arxiv.org/abs/2205.15602 "Ivec and Vojnović, Bayesian statistics approach to chess engines optimization"

[7]: https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF "James C. Spall, Implementation of the Simultaneous Perturbation Algorithm"

[8]: https://chessprogramming.org/Automated_Tuning "ChessProgramming Wiki — Automated Tuning"
