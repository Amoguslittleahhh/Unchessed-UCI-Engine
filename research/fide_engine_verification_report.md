# Unchessed-UCI-Engine: FIDE Rules Verification Report

**Author:** Manus AI  
**Audit date:** 4 September 2026 (UTC+8)  
**Repository:** `Amoguslittleahhh/Unchessed-UCI-Engine`  
**Audited commit:** `9c4f1f8e82dca8e4c729f275e8c15cb0e87e7b74`  
**Working branch:** `manus/research-facilities`  
**Primary authority:** FIDE Laws of Chess, effective 1 January 2023 [1]

## Overall conclusion

The current checkout **builds and executes**, and the observed automated suites provide strong evidence for the implemented Rust and Python test surfaces: the repository’s default Rust suite reported **123 passed, 0 failed, and 6 ignored**; the separately executed ignored Rust suite reported **6 passed, 0 failed**; and the complete Python suite reported **385 passed, 22 skipped, and 347 subtests passed**. The release workspace also built successfully. These are Level 4 automated-test results, with Level 5 runtime evidence for the UCI executable.

The engine also returned a legal move for all **1,011 non-terminal valid positions** in the expanded differential harness, including fixed castling, en-passant, promotion, pin/check, and 1,000 deterministic reachable positions checked against `python-chess 1.11.2`. A forced mate at halfmove clocks 100 and 150 was found, consistent with FIDE Article 5.1.1 taking precedence over the 50/75-move rules. This is meaningful evidence of move-generation/runtime soundness, not proof of complete FIDE compliance.

The broader claim “test the engine against literally all FIDE scenarios” is **not confirmed**. Three material boundaries remain. First, the engine returns a move in dead positions such as king versus king, king and bishop versus king, and king and knight versus king, whereas FIDE Article 5.2.2 requires an immediate draw when neither side can possibly checkmate. Second, the UCI parser reports malformed FEN as “could not parse” but continues searching the previous position and emits a normal `bestmove`; this is a protocol-level recovery behavior that can conceal rejected input from a caller. Third, the FIDE Laws include clock, physical-touch, scoresheet, arbiter, venue, penalty, rapid, blitz, and tournament-administration rules that a UCI chess engine cannot establish through ordinary position/search commands. Those clauses were explicitly catalogued as out of scope rather than counted as passes.

| Subclaim | Evidence level | Verdict | Supported scope |
|---|---:|---|---|
| Workspace compiles in debug and release profiles | 4 | **confirmed** | Current commit, current sandbox, Rust 1.98.1 |
| Built-in Rust tests pass | 4 | **confirmed** | 123 default tests plus 6 ignored tests |
| Built-in Python tests pass | 4 | **confirmed** | 385 passed, 22 skipped, 347 subtests |
| UCI returns legal moves on tested valid positions | 5 | **partially confirmed** | 1,011 non-terminal cases, including 1,000 deterministic reachable positions |
| Forced mate beats halfmove draw guard | 5 | **confirmed** | Tested white mate-in-one at halfmove 100 and 150 |
| Complete FIDE move/state rules are implemented | 3–5 | **refuted for the full claim** | Dead-position adjudication is absent in observed behavior; other clauses are not engine-testable |
| Full FIDE Laws, including OTB competitive rules, were tested | 2–5 | **could not verify** | Matrix is comprehensive by clause, but OTB-only rules require an arbiter/clock/scoresheet environment |

## Claim checked and threat model

The checked claim was: **“The main directory of the selected GitHub repository can be scanned read-only, the engine can be downloaded/built, and it can be tested against all scenarios represented by the supplied FIDE Laws link.”** The repository state, commit, environment, input positions, command lines, outputs, and generated artifacts are recorded in this branch.

The principal failure modes were illegal move generation, king-safety errors, castling-right loss or attack-through-square errors, en-passant timing errors, promotion errors, incorrect terminal detection, repetition/halfmove boundary errors, malformed-position crashes or silent fallback, and false confidence from tests that do not exercise the real UCI executable. The matrix therefore separates source inspection, core automated tests, runtime differential checks, and rules that require physical competition procedures.

## Exact method and evidence inventory

The repository was cloned with GitHub CLI into `/home/ubuntu/unchessed_audit/repo`. The initial remote branch was `main`; the audited commit was `9c4f1f8e82dca8e4c729f275e8c15cb0e87e7b74`, whose message was “Fix checkmate scored as a draw at halfmove clock 100”. A local branch named `manus/research-facilities` was created before adding research artifacts. No source files were changed.

The complete tracked-file inventory, code-surface inspection, source symbol map, and FIDE source capture are preserved in the `research/` directory. The workspace contains four Rust members: `unchessed-core`, `unchessed-adapter`, `unchessed-reviewer`, and `unchessed-datagen`. The core source includes `board.rs`, `fen.rs`, `movegen.rs`, `perft.rs`, `san.rs`, `search.rs`, `uci.rs`, and related evaluation/runtime modules. The repository also contains Python tools and tests, large PGN archives, model artifacts, configuration, and documentation. The large data archives were inventoried but were not all replayed as engine games because they are historical training/reference data rather than a FIDE legality oracle.

| Evidence artifact | Purpose |
|---|---|
| `research/repository_inventory.txt` | Tracked and discovered repository files |
| `research/code_surface_inspection.txt` | Manifests, READMEs, build instructions, and test markers |
| `research/static_rule_trace.txt` | Static rule-symbol and source trace |
| `research/fide_source/fide_laws_2023.md` | Captured user-supplied FIDE source |
| `research/fide_scenario_matrix.md` | Clause-by-clause scenario matrix and scope boundary |
| `research/build_test_smoke_v2.log` | Repository build, default Rust tests, and UCI smoke output |
| `research/ignored_tests_v2.log` | Six ignored Rust tests |
| `research/python_pytest_v2.log` | Complete Python test suite |
| `research/fide_uci_differential.py` | Reproducible runtime differential harness |
| `research/fide_uci_differential_results_v3.json` | Raw per-case FENs, outcomes, transcripts, and oracle fields for the expanded run |
| `research/fide_rigorous_perft.log` | Branch-local canonical perft integration test |
| `research/python_chess_perft_crosscheck_results.json` | 24 independent python-chess perft checks |
| `research/online_testing_authorities.md` | Online sources and method notes |
| `research/malformed_fen_completion.log` | Individual malformed-FEN reproductions |
| `research/halfmove_boundary_valid_probes.log` | Valid halfmove 100/150 probes |
| `research/individual_terminal_invalid_reproduction.log` | Dead-position and invalid-FEN reproductions |
| `research/anomaly_implementation_excerpt.txt` | Source excerpts for parser and halfmove behavior |

The FIDE authority states that the Laws cover over-the-board play, comprise Basic and Competitive Rules, and that the English text is authentic [1]. Accordingly, Articles 1–5, relevant position/state portions of Articles 7 and 9, and Appendix C were treated as directly or partially engine-testable. Articles concerning physical actions, clocks, scoresheets, arbiter decisions, conduct, penalties, rapid/blitz administration, and tournament points were retained in the matrix but not represented as engine passes.

## Runtime results

The repository script `scripts/build-and-test.sh` was run after installing Rust 1.98.1 because the system Cargo 1.75.0 could not read the repository’s lockfile version 4. The script’s captured output ends with `OK: build + tests + smoke passed`, including `bestmove d2d4` from start position and `bestmove a1a8` for the forced back-rank mate. The direct rerun of the ignored test command returned exit status 0. The optimized release build returned exit status 0 and produced `target/release/unchessed-adapter` at approximately 1.1 MB.

The complete Python suite was run with the dependencies declared by `tools/requirements-dev.txt`. The observed result was `385 passed, 22 skipped, 347 subtests passed in 22.33s`. In addition, a branch-local integration test independently reasserted canonical online perft counts across six positions, including the deeper Kiwipete depth-5 checkpoint, and reported `1 passed`. A separate pure-Python `python-chess 1.11.2` oracle recomputed 24 tractable canonical counts through depth 4 and reported `all_pass: true`; deeper counts were intentionally left to the Rust test because pure Python expansion to 193,690,690 Kiwipete nodes is not a tractable execution method. The skips are preserved in the raw log and were not converted into passes.

The expanded differential harness sent 1,022 cases through a current debug UCI executable: 16 fixed valid positions, 1,000 deterministic reachable positions, and 6 malformed-FEN inputs. For valid non-terminal cases, the returned UCI move was checked against the legal move set from `python-chess 1.11.2`. The summary reported `1,011` legal-bestmove passes and `0` non-terminal legal-move failures. Terminal handling was analyzed separately because a terminal position may legitimately return `0000` rather than a legal move.

| Runtime group | Count | Observation |
|---|---:|---|
| Fixed valid positions | 16 | Includes castling, en passant, promotion, pins, mate, stalemate, and material cases |
| Deterministic reachable positions | 1,000 | Seed `20260904`, up to 60 legal plies each |
| Valid non-terminal legal-bestmove passes | 1,011 | No observed illegal bestmove |
| Stalemate | 1 | Returned `bestmove 0000`; consistent with no legal move |
| Dead positions | 3 | Returned ordinary moves; inconsistent with immediate Article 5.2.2 draw adjudication |
| Malformed FEN inputs | 6 | Parser emitted an error for invalid king/rights/EP cases but continued with prior state; back-rank pawn was accepted |

## Material findings and anomalies

### Dead-position adjudication is not established and is contradicted by runtime behavior

FIDE Article 5.2.2 provides that a position is drawn when neither player can checkmate the opponent by any series of legal moves [1]. The engine was given king versus king, king and bishop versus king, and king and knight versus king. In each case the current executable returned `bestmove a1b2` at depth 1 instead of an immediate terminal draw signal. Source inspection found explicit halfmove and repetition draw handling in `unchessed-core/src/search.rs:519–533`, but no corresponding dead-material adjudication path. The observed result is a **Level 5 refutation of complete automatic dead-position handling**, although it does not prove that every possible dead position is mishandled.

### Checkmate precedence over the halfmove draw guard is supported

FIDE Article 9.6.2 states that the 75-move automatic draw applies unless the last move resulted in checkmate, which takes precedence [1]. The engine returned `bestmove a1a8` with `score mate 1` for the forced mate position at both halfmove clocks 100 and 150. Source inspection shows the guard at `unchessed-core/src/search.rs:521–530`, where checkmate is tested before the halfmove draw return. This subclaim is **confirmed at Level 5 for the tested construction**, not generalized to every checkmate/clock configuration.

### The UCI parser logs rejected FEN but leaves the previous position active

The UCI loop prints `could not parse` when `parse_position` returns `None` (`unchessed-core/src/uci.rs:380–386`). The parser calls `fen::parse(...).ok()?` (`unchessed-core/src/uci.rs:895–915`), so the invalid position is rejected. However, the outer loop does not terminate the game or emit a structured error; it retains the prior `Game`, then a subsequent `go` searches that prior state. For a missing king and two-king FEN, the executable emitted the parser error followed by `bestmove d2d4`, the move from the prior start position. For invalid castling rights and an occupied en-passant square, the same pattern occurred. This is **confirmed at Level 5** as a silent-fallback hazard for UCI clients. It is not a crash, but callers must inspect `info string` output or the engine must expose stronger error semantics to avoid treating the `bestmove` as a response to the rejected FEN.

A pawn on the back rank was accepted by the FEN parser and the engine returned `a7a8q`. FIDE Article 3.7.3.3 describes a pawn reaching the furthest rank as requiring immediate promotion [1], while Article 3.10.3 defines an illegal position as one unreachable by legal moves [1]. The parser’s acceptance of such a FEN may be intentional permissiveness for a promotion-ready state, but the distinction is not documented as a formal input contract. It remains an **unresolved scope issue**, not a claimed bug without a defined FEN policy.

### Halfmove 50/75 claims are not game-result outputs

On valid non-check FENs with halfmove clocks 100 and 150, the engine continued with `bestmove a1b1`. FIDE Articles 9.3 and 9.6 distinguish a 50-move claim from a 75-move automatic draw [1]. A UCI engine may reasonably leave claim and arbiter control to the GUI, but this executable does not emit a result or a documented claim-status response in the tested path. Therefore, complete FIDE game-result compliance for these clauses is **could not verify**, while move legality itself remained valid.

## Boundary of proof

The runtime evidence establishes behavior only for the audited commit, the installed Rust 1.98.1 environment, the debug/release binaries built there, the listed commands, and the finite FEN corpus. The 1,000 reachable positions were deterministic samples, not an enumeration of the game graph. Perft and unit tests provide strong internal evidence but do not prove all UCI paths, all model/runtime configurations, all memory sizes, all thread counts, or all long-running searches.

The audit did not and cannot establish physical one-hand movement, touch-move obligations, clock pressing, flag falls, default time, scoresheets, draw offers, arbiter intervention, penalties, venue/device conduct, appeals, rapid/blitz event administration, or tournament scoring through a normal UCI engine interface. These are explicitly marked out of scope in `research/fide_scenario_matrix.md`.

## Suggested remediations and follow-up checks

The following are suggestions, not verified fixes. Add explicit dead-position detection, or expose a clearly documented adjudication layer, and add tests for all dead-material classes plus composed positions where mating material exists but no mate is reachable. The acceptance test is that the engine reports an immediate draw/no-move result for king versus king, king and bishop versus king, and king and knight versus king, while preserving non-dead minor-piece positions.

Consider changing invalid `position fen` handling so a rejected FEN produces a structured UCI error and prevents `go` from searching stale state, or document the stale-state contract and make clients reject any `bestmove` following a parse error. The acceptance test is an isolated process that sends an invalid FEN after start position and confirms no bestmove is attributed to the rejected position.

Add explicit game-state/result tests for threefold and fivefold repetition, 50-move claim, 75-move automatic draw, and mate precedence, including castling-right and en-passant differences in the repetition key. The acceptance test should compare both state/result outputs and legal moves against an independently implemented oracle, not only search scores.

## References

[1]: https://handbook.fide.com/chapter/e012023 "FIDE Handbook — FIDE Laws of Chess taking effect from 1 January 2023"
[2]: https://chessprogramming.org/Perft_Results "Chess Programming Wiki — Perft Results"
[3]: https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html "Shredder Chess — Universal Chess Interface (UCI)"
[4]: https://python-chess.readthedocs.io/en/latest/core.html "python-chess — Core documentation"
