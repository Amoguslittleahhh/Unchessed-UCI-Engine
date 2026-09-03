# 28 — Insufficient-material and dead-position draws

**Investigation ID:** `tier1-dead-position-draws`
**Tier:** 1 (protocol completeness / infrastructure)
**Status:** Research and design only; no implementation, default change, match, or SPRT was performed.

## Executive conclusion

The engine does not currently detect insufficient-material or general dead-position draws. This is a correctness and interoperability gap, but it is not a promising playing-strength improvement. A conservative, exact detector for a small set of material classes would be cheap and useful at the search boundary; a complete dead-position detector is substantially harder because the FIDE rule is about **all possible legal continuations**, not merely the pieces currently on the board. The recommended disposition is **defer implementation unless the owner wants protocol/rule correctness**, while keeping a narrowly scoped, default-preserving design ready. Do not add an aggressive material heuristic that can incorrectly convert a legally possible win into a draw.

The practical priority is therefore lower than search extensions, move-ordering, or other candidates that can improve game strength. If pursued, first implement only formally safe terminal recognition for bare-king, king-versus-lone-bishop, and king-versus-lone-knight classes (with kings present and no pawns, rooks, queens, or additional pieces), then add fixed-position tests. Do not claim that this solves general dead-position recognition.

## Rule definition and terminology

The authoritative FIDE Laws distinguish stalemate from a dead position. Article 5.2.1 says stalemate is a draw when the player to move has no legal move and is not in check. Article 5.2.2 says a position is dead when **neither player can checkmate the opponent's king with any series of legal moves**, and that it immediately ends the game [1]. This is deliberately broader than the colloquial phrase “insufficient material.”

> A dead position is a position where neither player can mate the opponent’s king with any series of legal moves. [1]

“Insufficient material” is best treated as a conservative, easily recognized subset of dead positions. For example, king versus king, king and bishop versus king, and king and knight versus king cannot produce checkmate. By contrast, a material census is not a complete dead-position proof: fortress-like or blocked positions can contain several pieces while still making checkmate impossible, and some unusual material configurations require reasoning about legal moves, cooperation, promotions, and captures. The python-chess documentation makes the same qualification: its `has_insufficient_material()` check considers material and bishop square colors, but not piece positions; it can therefore miss fortress or forced-line dead positions [2].

This distinction matters for safety. A detector that says “draw” merely because the position looks drawish is an evaluation heuristic, not a rules detector. A rules detector may terminate only when the proposition is proved.

## Repository inspection

The master brief confirms that item 20 is absent and asks whether the evaluator already scores these positions near zero before treating the item as a priority [3]. I inspected the board, move generation, search, UCI, and hand-crafted evaluation paths.

| Area | Verified repository behavior |
|---|---|
| Board state | `unchessed-core/src/board.rs` stores bitboards/mailbox state, side to move, castling, en-passant, halfmove, fullmove, and Zobrist hash. There is no material-draw or dead-position predicate. |
| Move application | `Position::make()` updates pieces, captures, promotions, castling, en-passant, halfmove clock, side, and hash. It does not test insufficient material. |
| Legal moves | `unchessed-core/src/movegen.rs::legal()` generates legal moves. |
| Search terminal checks | `unchessed-core/src/search.rs::negamax()` checks `pos.halfmove >= 100` and repetition before normal search. It does not check material or dead positions. At a no-legal-move node it returns mate if in check, otherwise `self.draw(ply)` (stalemate). |
| Root UCI behavior | `unchessed-core/src/uci.rs::run_go()` computes legal moves and prints `bestmove 0000` when there are none. A legal-move dead position is searched normally, rather than immediately reported as a draw. |
| Evaluation | `unchessed-core/src/eval.rs::evaluate()` applies material, piece-square tables, tempo, bishop pair, mobility, passed-pawn, rook-file, and other terms. There is no terminal override for dead positions. The NNUE path likewise receives ordinary position features, including a halfmove bucket, rather than a rules-derived draw result. |
| Existing tests | There is a stalemate test (`stalemate_position_ignores_root_hints`) and repetition tests, but no insufficient-material or dead-position fixtures. |

The search’s existing halfmove test is a different rule. It treats 100 halfmoves as a draw score, corresponding to the claimable fifty-move threshold in the engine’s search convention; it is not proof that neither side can ever mate. FIDE also defines automatic fivefold repetition and the 75-move rule separately in Article 9.6 [1]. This report does not propose changing those existing mechanisms.

At the root, `bestmove 0000` is a useful signal for checkmate/stalemate handling but is not a UCI result announcement. In a GUI game, the GUI or game manager normally determines the result from the position and rules. A future draw detector must therefore specify whether it is only a search score/terminal-node facility or also a protocol-result facility; silently inventing a new UCI result line would be a separate compatibility decision.

## Practical importance

For normal engine play, a missing dead-position detector usually has small Elo impact. These positions are rare in serious games, and a sufficiently deep search often discovers that a bare-minor ending cannot force a win through the evaluation and horizon. The engine may nevertheless spend time searching a game that is already over under the Laws, emit nonzero centipawn scores from material/PST/tempo terms, and choose arbitrary legal moves. That is undesirable in analysis tools, tournament adjudication, game replay, and GUI integration even when it does not materially change results.

There is also a difference between **insufficient winning material** and **dead position**. A conservative insufficient-material detector can be constant-time or near-constant-time from bitboards. General dead-position recognition is closer to a reachability problem over legal move states. It must account for whether either side can create a mating position through captures, promotions, discovered checks, or cooperative legal play. Attempting a broad detector in the search hot path risks both CPU cost and false-positive rule errors.

The practical engine precedent supports restraint. A public Stockfish discussion shows a position that a specialized “unwinnability” analyzer identifies as dead while Stockfish evaluates Black as winning. The discussion explains that Stockfish is optimized for playing strength, not complete dead-position analysis, and that integrating complex recognition could weaken gameplay if it is wrong [4]. This is not a controlled benchmark, but it is a useful warning about scope: strong gameplay engines commonly do not attempt complete FIDE dead-position proof.

The evaluator should not be treated as a rules substitute. Even if the scalar score is near zero in canonical bare-king/minor positions, “near zero” is not “draw,” and a nonzero score is not evidence that checkmate is possible. PST asymmetries, tempo, bishop-pair logic, mobility, and contempt/draw-score handling can all produce a score despite an objectively terminal draw. Conversely, forcing every near-zero endgame to zero would destroy useful win/draw discrimination.

## Fixed-position test design

No source implementation was made. The following fixtures are cheap, deterministic tests for a future implementation and for manual engine/UCI probing. They should be parsed with full FEN semantics and checked both with White and Black to move where applicable.

| Category | FEN | Expected rule status | Purpose |
|---|---|---|---|
| Bare kings | `8/8/8/8/8/8/4k3/4K3 w - - 0 1` | Dead draw | Minimal guaranteed class; both sides have only a king. |
| Lone bishop | `8/8/8/8/8/8/4k3/2B1K3 w - - 0 1` | Dead draw | White has king+bishop, Black king. Mirror colors and side to move. |
| Lone knight | `8/8/8/8/8/8/4k3/2N1K3 w - - 0 1` | Dead draw | White has king+knight, Black king. Mirror colors and side to move. |
| Material is not enough to prove general deadness | `8/1k5B/7b/8/1p1p1p1p/1PpP1P1P/2P3K1/N3b3 b - - 0 1` | Reported by a specialized analyzer as dead; requires careful independent validation | Regression fixture for the boundary between simple insufficient material and general dead-position analysis; it must not be hard-coded without a proof and legality review [4]. |
| Ordinary winning material control | `7k/5Q2/6K1/8/8/8/8/8 w - - 0 1` | Not an insufficient-material draw | Ensures a detector does not classify queen-and-king positions as dead; the side to move has legal mating continuations. |
| Stalemate control | `7k/5Q2/6K1/8/8/8/8/8 b - - 0 1` | Stalemate draw | Separates “no legal move” from dead-position logic and preserves existing test behavior. |

The first three fixtures are appropriate unit tests for a narrow detector. Tests should assert the detector result independently of the static evaluator, search depth, contempt, and move ordering. Additional negative fixtures should include king plus a pawn, king plus rook, king plus queen, and bishop-versus-bishop configurations that are not covered by the chosen proof. A future implementation must document every accepted material class and prove that each is sufficient for the claim “no legal sequence can checkmate.”

A cheap executable probe was not completed because the sandbox’s Cargo 1.75.0 cannot parse this checkout’s Cargo.lock version 4 (`lock file version 4 requires -Znext-lockfile-bump`); no test pass is claimed. No alternate toolchain, tablebase files, game generation, or engine match was used.

## Recommended design if reopened

The safest location is a pure board-level predicate, for example `Position::is_insufficient_material()` or a module-level function, with no mutation and no dependence on search state. It should use bitboard counts and bishop square-color masks. The initial accepted set should be deliberately narrow:

1. Exactly one king per side and no other pieces.
2. One side has exactly one bishop and its king; the other has only its king.
3. One side has exactly one knight and its king; the other has only its king.

The implementation should not infer general deadness from “no pawns” or “total non-king value below X.” It should not classify king plus two knights versus king, arbitrary bishop-versus-bishop positions, or blocked multi-piece constructions without a separately justified proof. The FIDE criterion is possibility of *any* legal mate sequence, not ability to force mate against best defense; this is why intuitive “cannot force mate” shortcuts are unsafe.

Call the predicate early in `negamax`, alongside the existing halfmove and repetition terminal checks, so all legal-move positions receive a neutral draw score and do not consume search nodes. A root-level check could avoid unnecessary search, but it must preserve the current UCI behavior contract and decide whether a terminal draw should print `bestmove 0000`, a legal placeholder, or a result extension. The least disruptive first step is search-only terminal scoring plus tests; do not change normal `info score` formatting or add a new result protocol without an explicit UCI design.

Draw contempt needs an explicit policy. Existing search code represents a draw through `self.draw(ply)`, and the root passes `draw_score` derived from options/adaptation. A rules-terminal draw should use that same established pathway only if the project intentionally wants contempt to influence move selection in an already drawn game. For correctness and analysis predictability, a future implementation should prefer neutral zero for a rules-terminal draw, or at minimum document why contempt remains active. It must never turn a legally terminal draw into a nominal win/loss merely to avoid draws.

Required tests before integration would include color/side-to-move symmetry, FEN parsing, all accepted and rejected material classes, search returning a draw score at depth one, no regression in checkmate/stalemate, and preservation of repetition/50-move behavior. If a detector is later expanded beyond simple material classes, it requires a proof-backed corpus and adversarial legality tests, not only evaluator agreement.

## Recommendation and gates

**Recommendation: defer as a strength project; pursue only as a narrowly scoped correctness improvement.** The opportunity is real for protocol fidelity and wasted-search reduction, but the expected playing-strength gain is small and complete dead-position detection is not a cheap general feature. Preserve all defaults. Do not add a broad heuristic or tablebase dependency; the master brief explicitly places tablebase acquisition/hosting in Tier 3 [3].

If the owner elects to reopen this item, the next Tier 2 plan should be limited to the three exact material classes above, with no UCI output changes and no evaluation/training changes. The plan must include the fixed-position unit tests, benchmark the terminal check’s cost, and verify that the accepted classes are rules-safe. Any change affecting game-facing move selection still requires the master brief’s real paired-game gate; unit tests and fixed positions are not a substitute for SPRT. General dead-position analysis should remain **deferred or dropped** unless a maintained, authoritative algorithm and a sufficiently broad legal-position test corpus become available.

## What was verified versus assumed

| Item | Status |
|---|---|
| Master brief and existing reinforcement docs 00–12 read | Verified. |
| Draw/repetition/halfmove search handling inspected | Verified in `search.rs`. |
| Root legal-empty handling inspected | Verified in `uci.rs`; it prints `bestmove 0000`. |
| No insufficient-material/dead-position predicate found | Verified by source search and inspection. |
| Evaluator has no terminal draw override | Verified for hand-crafted evaluation; no rules predicate was found in the NNUE path either. |
| FIDE definition and 50/75-move/repetition distinction | Verified from the official FIDE Laws [1]. |
| python-chess material-only limitation | Verified from current documentation [2]. |
| Complete dead-position detector would be complex and false-positive risk is material | Literature/engineering inference, supported by the Stockfish discussion [4]; not a measured Unchessed benchmark. |
| Current evaluator score on the proposed FEN fixtures | Not run; no score or near-zero claim is made. |
| Playing-strength impact | Not measured; no Elo, cutechess, or SPRT claim is made. |

## References

[1]: https://handbook.fide.com/chapter/e012023 "FIDE Laws of Chess, Articles 5 and 9"
[2]: https://python-chess.readthedocs.io/en/latest/core.html "python-chess core API and outcome documentation"
[3]: ../../upload/pasted_content_6.txt "Unchessed AI master research and engine-strength brief"
[4]: https://github.com/official-stockfish/Stockfish/discussions/3916 "Dead positions CHA can see but Stockfish not"
[5]: https://chessprogramming.org/Draw "Chessprogramming Wiki: Draw"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/28-dead-position-draws.md`
