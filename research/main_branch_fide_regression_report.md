# Independent Re-evaluation of Updated `main`

**Repository:** `Amoguslittleahhh/Unchessed-UCI-Engine`  
**Engine baseline:** remote `main`, commit `81a41818690afdf6f80b7ae3d822538dd795476f`  
**Commit date:** 4 September 2026 (UTC+8)  
**Research branch for artifacts:** `manus/research-facilities`  
**Engine under test:** `/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter` built from the fresh `main` clone  
**Primary authority:** FIDE Laws of Chess effective 1 January 2023 [1]

## Executive conclusion

This evaluation was restarted from a fresh clone of remote `main`. The previous research branch, its PDF, its added tests, and its prior engine artifacts were not used as inputs to the engine under test. They were considered only for post hoc comparison after the new `main` snapshot had been independently built and exercised.

The updated `main` snapshot passes the repository’s Python test surface, the default Rust suite, the ignored Rust suite, optimized release tests, canonical perft counts, a complete seven-position runtime mate corpus, and a 1,000-position reachable-state UCI differential test. The fresh conformance harness reported valid UCI handshake and readiness, **1,004/1,004 non-terminal valid positions with oracle-legal bestmoves**, and **10/10 terminal positions returning `bestmove 0000`** under the expanded `python-chess` terminal oracle. A separate move-list/state-sequence harness reported **252/252 legal-or-terminal results**.

The main branch therefore materially improves on the prior observed engine behavior: the earlier dead-position gap is no longer reproduced, and all seven verified forced mates pass when the harness waits for `bestmove`. Nevertheless, the repository’s own `scripts/build-and-test.sh` returns nonzero because its two-second pipe race expects the depth-six mate before sending `quit`; the exact same position returns `a1a8` when the caller waits for the real `bestmove`. This is best classified as a **smoke-test synchronization defect**, not a demonstrated search-rule failure.

A separate issue remains: malformed FEN inputs produce a parser error but the UCI process still emits a `bestmove` afterward. The result may be `0000` or a move from the previously installed position, and the protocol does not expose a structured rejected-position state. This remains a fail-closedness hazard for clients that do not correlate parser errors with subsequent search commands.

## Current authoritative rules baseline

The current FIDE Handbook still lists **FIDE Laws of Chess taking effect from 1 January 2023** as the Laws document. The official Rules Commission page provides the same 2023 Laws PDF and changes table [2]. FIDE’s current 2026 General Rules and Regulations became effective on 1 March 2026 [3], but they are complementary technical and administrative regulations; they do not replace the 2023 Laws for board movement, legal positions, checkmate, stalemate, dead positions, repetition, or the 50/75-move rules.

The testing methodology uses canonical perft data from the Chess Programming Wiki [4], the open UCI protocol context described by Shredder Chess [5], and `python-chess` core APIs as an independently maintained executable oracle [6]. These sources have distinct roles: FIDE is normative; perft is a move-generator reference; UCI defines the engine/GUI boundary; and `python-chess` provides independently implemented legal-move and status computations.

## Fresh baseline and complete rescan

The new clone contains 1,453 tracked files and four Rust workspace members: `unchessed-core`, `unchessed-adapter`, `unchessed-reviewer`, and `unchessed-datagen`. The current `main` commit is `81a41818690afdf6f80b7ae3d822538dd795476f`, titled “Close out the audit's build/CI tail: linker preflight, release tests, CI.” The source rescan covered Rust board, FEN, move generation, perft, search, SAN, UCI, data-generation, and reviewer modules, as well as Python tools and tests, CI, build scripts, benchmark corpora, manifests, and documentation.

The codebase now includes explicit tests for the seven-position `matetrack.epd` corpus, UCI position resend behavior, observed-plies carry/reset behavior, root-hint mate protection, stalemate handling, and performance/runtime gates. The updated search terminal path includes a halfmove guard that checks for checkmate precedence before returning the draw score. The parser still returns `Option<Game>` and uses `fen::parse(...).ok()?`; the outer UCI loop logs an error and continues rather than installing an explicit rejected-position state.

## Existing test and build reproduction

The declared build script was run exactly as committed on `main`. Its compile and Rust-test stages completed, but the script exited with status 1 at the forced-mate UCI smoke check. The script sends `go depth 6`, sleeps two seconds, then sends `quit` and requires `bestmove a1a8`. In the exact reproduction, the engine emitted `bestmove a1b1` before the premature quit. A corrected caller that waits for `bestmove` produced the expected mate sequence through depth six and returned `bestmove a1a8`.

| Surface | Result | Interpretation |
|---|---:|---|
| `scripts/build-and-test.sh` | **failed at smoke** | Build/tests pass; depth-six smoke races `quit` after two seconds |
| Default Rust workspace tests | **123 passed, 0 failed, 6 ignored** | Internal debug test surface passed |
| Ignored Rust tests | **6 passed, 0 failed** | Deep perft and runtime gates passed |
| Python suite | **385 passed, 22 skipped, 347 subtests** | Complete `tools` test surface passed |
| Release Rust tests | **123 passed, 0 failed, 6 ignored** | Optimized test surface passed |
| Release ignored tests | **6 passed, 0 failed** | Optimized deep/runtime gates passed |
| Release build | **passed** | Optimized workspace compiled |

The smoke failure is not silently converted into a pass. It is retained as a current repository-gate defect, while the correctly synchronized runtime result is recorded separately.

## Expanded FIDE and runtime tests

### Canonical perft

A standalone branch-local Rust checker linked against the fresh `main` build evaluated the online canonical positions. The start position matched depths 1--6, including 119,060,324 nodes at depth six. Kiwipete matched depths 1--5, including 193,690,690 nodes at depth five. Positions 3, 4, 5, and 6 matched their published depth ranges. Every observed count equaled the expected count.

| Position | Depth range | Result |
|---|---:|---|
| Start position | 1--6 | **All pass** |
| Kiwipete | 1--5 | **All pass** |
| Position 3 | 1--5 | **All pass** |
| Position 4 | 1--4 | **All pass** |
| Position 5 | 1--4 | **All pass** |
| Position 6 | 1--4 | **All pass** |

### Seven-position verified mate corpus

The fresh runtime harness independently parsed the repository’s seven-position `matetrack.epd` file with `python-chess`, converted each expected SAN mate to UCI, launched the updated main executable from its own build directory, waited for `bestmove`, and compared the result. All seven passed at depth ten: both colors’ back-rank mates, the full-shield back-rank mate, smothered knight, queen-supported king mate, corner queen mate, and two-rook ladder mate.

### Differential reachable-state UCI testing

The fresh harness sent 14 fixed FIDE-relevant positions and 1,000 deterministic reachable positions generated from the initial position with seed `20260904` and up to 60 legal plies. Book and adaptive behavior were disabled to isolate rules and protocol behavior. The harness checked `uciok`, two `readyok` responses, returned move syntax, and membership in the exact legal-move set from `python-chess 1.11.2`.

| Category | Count | Result |
|---|---:|---|
| Fixed positions | 14 | Included castling, blocked castling, en passant, promotion, check, pin, stalemate, dead material, and halfmove boundaries |
| Reachable positions | 1,000 | Deterministic legal-play samples |
| Valid positions | 1,014 | All parsed and searched |
| Non-terminal positions | 1,004 | **1,004/1,004 oracle-legal bestmoves** |
| Terminal positions | 10 | **10/10 `bestmove 0000`** |
| Malformed FEN cases | 5 | All produced parser-error text and also emitted a bestmove |

The terminal oracle included checkmate, stalemate, insufficient material, and automatic 75-move draw status. This corrected an earlier harness classification mistake in which dead positions were treated as non-terminal. The updated main engine’s `0000` response for king versus king, king-and-bishop versus king, and king-and-knight versus king is now consistent with the expanded terminal oracle.

### Move-list and state-sequence testing

A second harness generated 250 legal move-list sequences from the initial position, then added explicit en-passant and castling sequence cases. It sent `position startpos moves ...` followed by `go depth 1`, and checked the returned move against the resulting oracle position. All 252 cases passed. A separate malformed-FEN sequence confirmed that a parser error is visible but that two bestmoves can still be emitted in the same process, so client-side command/result correlation remains necessary.

## Findings against the latest main snapshot

### Dead-position behavior improved and prior gap no longer reproduced

FIDE Article 5.2.2 requires a draw when neither player can possibly checkmate the opponent by any series of legal moves [1]. The fresh main executable returned `bestmove 0000` for king versus king, king and bishop versus king, and king and knight versus king. The earlier research-branch finding that these positions returned ordinary moves is therefore **not reproduced on current main**. This is a confirmed improvement for the tested dead-material classes, not proof of every possible dead-position construction.

### Mate precedence remains correct when the caller waits for completion

The updated main search source checks checkmate precedence before the halfmove draw guard. The corrected runtime probe returned `a1a8` at depth six for the forced mate, and the seven-position mate corpus passed at depth ten. This supports FIDE Article 9.6.2 behavior for the tested constructions. The committed build script still has a timing-sensitive smoke assertion that can terminate the search before the result is emitted.

### Build-script smoke race is a reproducible repository defect

The script’s comments state that a real GUI waits for `bestmove`, but the script itself sleeps for a fixed two seconds and sends `quit`. On this main snapshot, the exact command produced `bestmove a1b1`, while a synchronized five-second probe produced depth-one through depth-six mate information and `bestmove a1a8`. The contradiction is within the repository gate: the engine’s synchronized result passes, but the declared smoke gate is not robust to the new timing behavior. This should be fixed by reading until `bestmove` with a timeout rather than sleeping and quitting.

### Malformed FEN still emits a post-rejection bestmove

Five malformed cases were tested: missing black king, two white kings, invalid castling rights, occupied en-passant square, and a pawn on the back rank. All five generated `could not parse` text, and all five also generated a `bestmove`. FIDE Article 3.10.3 defines an illegal position as one that cannot have been reached by any series of legal moves [1]. The parser correctly rejects several malformed inputs at the text level, but the UCI state machine does not expose a structured failure that prevents subsequent search output from being mistaken for a response to the rejected FEN. The back-rank pawn remains a policy-boundary case because a promotion-ready FEN may be intentionally accepted as a permissive analysis input.

### Fifty- and seventy-five-move status remains only partially exposed

The engine returns legal moves for valid non-check positions at high halfmove clocks rather than emitting a game-result or claim-status message. FIDE Article 9 distinguishes the 50-move claim from the 75-move automatic draw, and Article 9.6.2 preserves checkmate precedence [1]. The expanded terminal oracle classified automatic 75-move positions as terminal, but the UCI output contract does not expose a structured draw result or claim state. This remains a protocol/API coverage limitation rather than a demonstrated illegal move.

## Proof boundaries

The result is valid for the fresh `main` commit, the recorded Rust 1.98.1 environment, the debug and release binaries built there, the finite FEN corpus, and the exact commands and scripts preserved on the research branch. The 1,000 reachable positions are deterministic samples rather than an enumeration of the game graph. Perft establishes complete leaf counts only for selected positions and depths. A legal selected move does not prove that every legal move was enumerated by the search path.

The FIDE Laws cover over-the-board play. The audit cannot establish one-hand movement, touch-move, physical displacement, clock pressing, flag falls, scoresheets, draw offers, arbiter intervention, penalties, appeals, venue/device conduct, rapid/blitz administration, or tournament scoring through a UCI engine. Current 2026 FIDE general regulations are administrative/technical context, not substitutes for those physical competition tests [3].

## Reproducibility artifacts

All new artifacts are committed only on `manus/research-facilities`:

| Artifact | Purpose |
|---|---|
| `research/main_branch_fide_regression_report.md` | This updated report |
| `research/main_branch_fide_conformance.py` | Fresh-main fixed/reachable/malformed-FEN harness |
| `research/main_branch_fide_conformance_results.json` | Raw 1,014-case results and transcripts |
| `research/main_branch_fide_conformance_v5.log` | Summary of corrected run |
| `research/main_branch_matetrack_runtime.py` | Seven-position synchronized mate harness |
| `research/main_branch_matetrack_runtime_results.json` | Raw seven-case mate results |
| `research/main_branch_matetrack_runtime_v3.log` | Seven-for-seven summary |
| `research/main_branch_uci_sequence_conformance.py` | 252 move-list/state-sequence harness |
| `research/main_branch_uci_sequence_results.json` | Raw sequence results |
| `research/main_branch_perft.rs` | Standalone perft checker linked to fresh-main library |
| `research/main_branch_perft.log` | Canonical perft output |
| `main_build_test.log` | Exact committed build-script result |
| `main_ignored_tests.log` | Debug ignored-test result |
| `main_python_tests.log` | Complete Python suite |
| `main_release_tests.log` | Optimized Rust suite |
| `main_release_ignored_tests.log` | Optimized ignored suite |
| `main_exact_smoke.log` | Exact two-second smoke reproduction |
| `main_exact_smoke_wait5.log` | Synchronized depth-six mate reproduction |
| `main_code_surface.txt` | Updated code and test inventory |
| `main_failure_inspection.txt` | Source-level failure inspection |
| `main_online_authorities.md` | Refreshed official sources and effective dates |

## References

[1] [FIDE Handbook, FIDE Laws of Chess taking effect from 1 January 2023](https://handbook.fide.com/chapter/e012023).  
[2] [FIDE Rules Commission, 2023 Laws of Chess and changes table](https://rcc.fide.com/2023-laws-of-chess/).  
[3] [FIDE Handbook, General Rules and Regulations effective from 1 March 2026](https://handbook.fide.com/chapter/GeneralRulesAndRegulations032026).  
[4] [Chess Programming Wiki, Perft Results](https://chessprogramming.org/Perft_Results).  
[5] [Shredder Chess, Universal Chess Interface](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html).  
[6] [python-chess Core documentation](https://python-chess.readthedocs.io/en/latest/core.html).
