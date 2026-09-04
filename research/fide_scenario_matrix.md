# FIDE Laws of Chess Scenario Matrix

**Authority:** FIDE Laws of Chess taking effect 1 January 2023, supplied by the user at [https://handbook.fide.com/chapter/e012023](https://handbook.fide.com/chapter/e012023). The captured source is preserved at `research/fide_source/fide_laws_2023.md`.

## Interpretation and coverage boundary

This matrix treats a chess engine as a **position/move legality and game-state implementation**, normally exercised through FEN, UCI `position`, and UCI move input. It does not pretend that an engine can literally implement physical-touch rules, clocks, arbiter judgement, scoresheets, venue conduct, or tournament penalties. Those clauses are listed explicitly as **not engine-testable** rather than silently omitted. “Exhaustive” below means exhaustive coverage of the rule clauses and representative adversarial constructions for the engine-testable subset; it does not mean enumeration of every reachable chess position.

| ID | FIDE clause | Scenario / oracle | Class | Planned evidence |
|---|---|---|---|---|
| F01 | 1.2 | White moves first; side-to-move alternates after every accepted move | Engine | Start position and short legal sequence |
| F02 | 1.4.1, 3.9 | A move leaving own king in check is rejected; capturing the opposing king is never accepted | Engine | Pinned-piece and king-capture probes |
| F03 | 1.4.1, 5.1.1 | Checkmate ends the game immediately and is not reported as a draw due to the halfmove clock | Engine | Forced mate with halfmove 100+ |
| F04 | 1.5, 5.2.2 | Dead positions where neither side can ever mate are drawn | Engine/state | K vs K; K+B vs K; K+N vs K; same-colour bishops where applicable |
| F05 | 2.1–2.4 | Board is 8×8; coordinates, colors, files/ranks/diagonals map correctly | Core | Perft and square/attack tests |
| F06 | 2.2–2.3 | Standard initial placement and 16 pieces per side | Core | Start FEN and piece inventory |
| F07 | 3.1 | Cannot move onto own piece; capture removes opposing piece | Engine | Occupancy and capture probes for every piece type |
| F08 | 3.2 | Bishop moves only diagonally | Engine | Empty, blocked, off-axis, capture, and edge probes |
| F09 | 3.3 | Rook moves only on rank/file | Engine | Empty, blocked, off-axis, capture, and edge probes |
| F10 | 3.4 | Queen moves rank/file/diagonal only | Engine | Empty, blocked, off-axis, capture, and edge probes |
| F11 | 3.5 | Sliding pieces cannot jump intervening pieces | Engine | One blocker, same-color blocker, enemy blocker |
| F12 | 3.6 | Knight moves to the eight L-shaped destinations, ignoring intervening pieces | Engine | Center, edge, corner, occupancy probes |
| F13 | 3.7.1 | Pawn single advance requires empty destination and correct direction | Engine | White/black, blocked, edge-rank invalid cases |
| F14 | 3.7.2 | Pawn double advance is allowed only from original rank and requires both squares empty | Engine | Original/non-original rank; one/two blockers |
| F15 | 3.7.3 | Pawn captures only diagonally forward onto an opposing piece | Engine | Empty diagonal, backward, lateral, own-piece cases |
| F16 | 3.7.3.1–3.7.3.2 | En passant is legal only immediately after an adjacent opposing two-square pawn advance | Engine/state | Valid EP; delayed EP; wrong pawn; phantom target |
| F17 | 3.7.3.3–3.7.3.5 | Promotion on furthest rank must become exactly Q/R/B/N of same color with immediate effect | Engine | All eight promotion choices, quiet/capture, both colors |
| F18 | 3.8.1 | King moves one adjacent square, not onto attacked square or occupied own piece | Engine | All king-neighborhood attack/occupancy permutations |
| F19 | 3.8.2 | Castling moves king two squares toward rook and rook to crossed square as one move | Engine | White/black K/Q castling from valid positions |
| F20 | 3.8.2.1 | Castling right is lost permanently after king moves | State | Move-away-and-back then castle attempt |
| F21 | 3.8.2.1 | Castling right is lost permanently after the relevant rook moves | State | Rook-away-and-back then castle attempt |
| F22 | 3.8.2.2(3) | Castling is forbidden if king start, transit, or destination square is attacked | Engine | Attack each of the three squares, including constrained attackers |
| F23 | 3.8.2.2(4) | Castling is forbidden when any interposed piece exists | Engine | Each between-square occupancy and both sides |
| F24 | 3.9.1 | A king is in check from an attacking piece even when that attacker is pinned to its own king | Engine | Pinned attacker/check detection |
| F25 | 3.9.2 | No move may expose or leave own king in check | Engine | Absolute pins, discovered checks, interposition, king capture |
| F26 | 3.10.1–3.10.2 | Legal/illegal move classification matches Articles 3.1–3.9 | Engine/API | Differential move legality against reference library |
| F27 | 3.10.3 | Malformed/unreachable positions are rejected or explicitly scoped by FEN parser | Parser | Invalid kings, pawns on back rank, bad rights/EP, side-in-check constraints |
| F28 | 5.2.1 | Side to move with no legal move and king not in check is stalemate/draw | Engine/state | Multiple stalemate constructions |
| F29 | 5.2.2 | Dead position is drawn immediately, distinct from merely low material | Engine/state | Dead and non-dead minor-piece positions |
| F30 | 5.2.3 | Draw agreement is a protocol/game-management action after both players have moved | Protocol/state | If exposed by UCI; otherwise mark unsupported |
| F31 | 6.1–6.12 | Clock operation, flag fall, default time, time controls, interruptions, and clock defects | Not engine-testable | Document interface absence; no false pass |
| F32 | 7.2 | Incorrect initial setup/board orientation/colour reversal restoration | OTB/arbiter | Not representable through ordinary UCI state; document limitation |
| F33 | 7.4, 7.6 | Displaced pieces and restoration to prior position | OTB/arbiter | Not engine-testable through UCI |
| F34 | 7.5.1 | Completed illegal move restoration and replacement legal move | Protocol/state | Test only if engine exposes illegal-move history/recovery; otherwise unsupported |
| F35 | 7.5.2 | Pawn reaching last rank without replacement is corrected to queen under competition procedure | OTB/protocol | Document unsupported unless explicit interface exists |
| F36 | 7.5.3–7.5.4 | Pressing clock without move or using two hands is penalized as illegal move | OTB/clock | Not engine-testable |
| F37 | 7.5.5 | First/second completed illegal move penalties and loss/draw exceptions | OTB/arbiter | Not engine-testable |
| F38 | 8.1–8.7 | Scoresheets, recording, draw symbols, incomplete reconstruction, signed results | OTB/arbiter | Not engine-testable; SAN parser tested separately where present |
| F39 | 9.1 | Draw offers and event restrictions | OTB/protocol | Not engine-testable unless UCI exposes offer state |
| F40 | 9.2.1–9.2.2 | Threefold claim by intended move or current position with claimant to move | Protocol/state | Test if repetition history/claim API exists |
| F41 | 9.2.3 | Same position requires same side, same piece placement, and same legal possibilities | State | Differential repetition-key tests |
| F42 | 9.2.3.1 | En-passant availability makes positions different even if no EP capture is ultimately possible under some interpretations | State | EP-key cases with immediate-history control |
| F43 | 9.2.3.2 | Castling rights make positions different; rights lost only after king/rook moves | State | Same board with rights/no-rights and move-away-return |
| F44 | 9.3.1–9.3.2 | Fifty-move claim at 50 full moves without pawn move/capture, including intended move form | State/protocol | Halfmove boundary with quiet moves and resets |
| F45 | 9.4–9.5 | Touching a piece, pausing clock, correct/incorrect claims, and penalties | OTB/arbiter | Not engine-testable unless protocol explicitly models them |
| F46 | 9.6.1 | Automatic draw at fivefold repetition | State | Construct five occurrences with equal state keys |
| F47 | 9.6.2 | Automatic draw at 75 moves each without pawn move/capture, unless last move is checkmate | State | Halfmove 150 draw and mate-precedence pair |
| F48 | 10.1–10.2 | Tournament points and score limits | Tournament | Not engine-testable by the UCI engine |
| F49 | 11.1–11.12 | Player conduct, venue, devices, appeals, and arbiter assistance | OTB/arbiter | Not engine-testable |
| F50 | 12.1–12.9 | Arbiter duties, intervention, penalties, and fair-play administration | OTB/arbiter | Not engine-testable |
| F51 | Appendix A | Rapid chess timing, recording, penalties, setup and illegal-position handling | OTB/clock | Not engine-testable; document unsupported |
| F52 | Appendix B | Blitz timing, recording, and rapid-law fallback | OTB/clock | Not engine-testable; document unsupported |
| F53 | Appendix C | Algebraic notation generation/parsing, disambiguation, check/mate, castling, promotion, capture | Parser/protocol | Test SAN module if exposed |
| F54 | Appendix D | Blindfold/disabled-player notation adaptations | OTB | Not engine-testable |

## Scenario expansion requirements

For each engine-testable row, the execution set must include both colors where applicable, every piece type where applicable, legal and illegal controls, edge/corner squares, occupied and empty destinations, pinned and unpinned attackers, and state-reset cases. Repetition and fifty/seventy-five-move cases must vary castling rights, en-passant state, side to move, captures, pawn moves, checks, and mate precedence. Any reference-library disagreement must be preserved with the exact FEN, move sequence, expected oracle, engine output, stderr, exit status, and source location for follow-up.

## Traceability status

This is a test-design artifact, not a result. Coverage claims are **unverified until runtime evidence is collected**. The final report must distinguish clauses tested at source-inspection level, automated-test level, and runtime level, and must list every OTB-only clause as out of scope rather than treating it as passed.
