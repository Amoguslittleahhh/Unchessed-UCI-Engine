# FIDE Scenario Review and Old-versus-New Results

## Scope and evidence discipline

The updated engine was evaluated from a fresh remote `main` clone at commit `81a41818690afdf6f80b7ae3d822538dd795476f`. The earlier research branch, its PDF, its added test files, and its previous executable were excluded from the engine-under-test path. Previous artifacts were used only as historical comparison data after the new main snapshot had produced independent results.

The normative baseline remains the official FIDE Laws of Chess taking effect 1 January 2023 [1]. The current FIDE Handbook also lists 2026 general regulations effective 1 March 2026 [2]; those are complementary technical and administrative regulations, not a replacement for the 2023 Laws governing legal movement and game-state adjudication. Canonical perft counts came from the Chess Programming Wiki [3], protocol context from the UCI overview [4], and legal-move/status checks from `python-chess` 1.11.2 [5].

## Newly reviewed FIDE scenarios

| Scenario family | FIDE clauses | Adversarial treatment | Current result |
|---|---|---|---|
| Standard initial state | Articles 1–2 | Start FEN, side-to-move, standard placement, perft depths 1–6 | Canonical counts pass |
| Sliding movement | Articles 3.2–3.5 | Bishops, rooks, queens with empty squares, same-color blockers, enemy blockers, and edge geometry | Covered by perft and legal-bestmove oracle |
| Knight movement | Article 3.6 | Edge/corner and unconstrained-jump cases | Covered by perft and oracle |
| Pawn movement | Article 3.7.1–3.7.3 | Single/double pushes, blockers, captures, both colors | Covered by perft and fixed positions |
| En passant | Articles 3.7.3.1–3.7.3.2 | Valid target, immediate-history requirement, wrong/phantom targets, explicit move-list sequence | Valid move-state sequence passes |
| Promotion | Articles 3.7.3.3–3.7.3.5 | Quiet and capture-ready promotions, promotion-ready FEN, legal returned promotion | Runtime case returns legal promotion |
| Castling | Articles 3.8.2–3.8.2.2 | Both rights, blocked path, attacked start/transit/destination, rights encoded in FEN | Perft and fixed legal-move checks pass |
| Check and pins | Articles 3.9–3.10 | In-check side, absolute pin, king safety, no king capture | Fixed legal-move checks pass |
| Checkmate/stalemate | Articles 1.4, 5.1, 5.2.1 | Forced mates, stalemate positions, no-move terminal response | Seven-mate corpus passes; stalemate returns `0000` |
| Dead positions | Articles 1.5, 5.2.2 | K vs K, K+B vs K, K+N vs K, insufficient material oracle | Current main returns `0000` for all tested classes |
| Halfmove boundaries | Articles 9.3, 9.6 | High halfmove quiet states and mate precedence | Mate wins when synchronized; result/claim API remains implicit |
| FEN rejection | Article 3.10.3 | Missing king, two same-color kings, invalid rights, occupied EP, back-rank pawn | Error text appears, but a bestmove still follows |
| UCI state sequencing | Protocol boundary | 252 `position startpos moves ...` sequences, explicit EP/castling sequence | 252/252 legal-or-terminal passes |
| OTB-only rules | Articles 4, 6–12, Appendices A/B/D | Touch-move, one-hand move, clocks, scoresheets, arbiter penalties, rapid/blitz procedures | Not engine-testable through ordinary UCI |

The adversarial verifier handled edge cases by first separating engine-observable rules from physical competition rules, then using independent oracles and exact input/output capture. It corrected a potential false positive in the terminal classification: dead positions were initially treated as non-terminal when only checkmate/stalemate were considered; the corrected oracle included insufficient material and automatic 75-move status. It also detected and removed a harness synchronization error in which sending `quit` immediately after `go` could associate an old or incomplete result with a new case. The final synchronized mate harness waits for `bestmove` before terminating the process.

## Headline comparison

| Metric | Earlier audited commit `9c4f1f8` | Updated main `81a41818` | Difference / interpretation |
|---|---:|---:|---|
| Default Rust tests | 123 pass, 0 fail, 6 ignored | 123 pass, 0 fail, 6 ignored | Same count; new main retains pass status |
| Ignored Rust tests | 6 pass, 0 fail; 54.39 s | 6 pass, 0 fail; 52.76 s | 1.63 s faster, approximately 3.0% lower observed wall time; hardware/load not controlled |
| Python suite | 385 pass, 22 skip, 347 subtests; 22.33 s | 385 pass, 22 skip, 347 subtests; 21.54 s | 0.79 s faster, approximately 3.5%; not a controlled benchmark |
| Differential cases | 1,022 | 1,014 fresh fixed/reachable valid cases plus 5 invalid cases | Denominators differ; do not compare as a single rate |
| Non-terminal legal bestmoves | 1,011 pass, 0 fail | 1,004 pass, 0 fail | New run uses a corrected terminal oracle and a different fixed/invalid corpus |
| Dead positions | 3/3 returned ordinary moves | 3/3 returned `0000` | Prior gap no longer reproduced on current main |
| Verified mate corpus | No independent seven-case runtime result | 7/7 pass when synchronized | New direct runtime coverage |
| Move-list/state sequences | Not previously run | 252/252 pass | New protocol/state coverage |
| Canonical perft | Prior branch integration passed | Fresh main linked checker passed all 28 recorded counts | No observed regression in tested perft positions |
| Build-script smoke | Passed in prior capture | Failed at fixed two-second mate smoke | Current script races `quit`; synchronized depth-six probe returns `a1a8` |
| Malformed FEN | Error followed by stale bestmove observed | Error followed by bestmove in all 5 cases | Hazard persists, though exact output/state must be interpreted per case |

The perft comparison counts 28 fresh-main checks: start position at six depths, Kiwipete at five, position 3 at five, and positions 4–6 at four each. Every observed count matched the online expected value. The “performance” deltas above are test wall times, not engine strength or nodes-per-second improvements. They are descriptive only because CPU scheduling, cache state, and background load were not controlled.

## Updated verdicts

**Build and test surface — partially confirmed.** The source compiles in debug and release profiles; default and ignored Rust tests, Python tests, release tests, and release ignored tests pass. The repository’s own script still fails at its forced-mate smoke assertion because its fixed sleep/quit protocol is not a reliable completion barrier.

**Move generation — confirmed within tested scope.** All fresh-main canonical perft counts pass, all 1,004 fresh-main non-terminal differential positions return oracle-legal moves, all 252 move-list/state sequences pass, and all seven verified mates pass when the caller waits for completion. This does not prove exhaustive correctness over the entire game graph.

**Dead-position adjudication — partially confirmed.** Current main returns `0000` for the three previously failing dead-material classes. This confirms the observed improvement for those constructions, not every possible dead position.

**Malformed-position handling — refuted as fail-closed behavior.** Every malformed case produced parser-error text and also emitted a bestmove. The executable does not expose a structured rejected-position state that prevents a client from attributing that bestmove to the malformed FEN.

**Full FIDE compliance — could not verify.** Touch-move, physical movement, clocks, scoresheets, arbiter judgment, penalties, rapid/blitz administration, and tournament procedures are outside a normal UCI engine’s observable interface. They are not counted as passes.

## Reproduction artifacts

The raw evidence is stored on `manus/research-facilities` in `research/`: `main_branch_fide_conformance_results.json`, `main_branch_matetrack_runtime_results.json`, `main_branch_uci_sequence_results.json`, `main_branch_perft.log`, `main_build_test.log`, `main_release_tests.log`, `main_release_ignored_tests.log`, `main_python_tests.log`, `main_exact_smoke.log`, `main_exact_smoke_wait5.log`, and `old_new_metrics_rederived_latest.txt`.

## References

[1] [FIDE Handbook, Laws of Chess taking effect from 1 January 2023](https://handbook.fide.com/chapter/e012023).  
[2] [FIDE Handbook, General Rules and Regulations effective from 1 March 2026](https://handbook.fide.com/chapter/GeneralRulesAndRegulations032026).  
[3] [Chess Programming Wiki, Perft Results](https://chessprogramming.org/Perft_Results).  
[4] [Shredder Chess, Universal Chess Interface](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html).  
[5] [python-chess Core documentation](https://python-chess.readthedocs.io/en/latest/core.html).
