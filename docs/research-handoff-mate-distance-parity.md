# Unchessed Evaluation Research Handoff

**Branch:** `manus/research-facilities`
**Repository:** `Amoguslittleahhh/Unchessed-UCI-Engine`
**Purpose:** Preserve the complete context of the clean-room evaluation and parity research before experimentation stops.

## Executive status

The project has a functional homemade eval-bar, a loaded original NNUE path, a quantized incremental residual experiment, leakage-safe real-game manifests, black-box Stockfish 19 comparison tooling, and an opt-in teacher-calibrated presentation mode. The strict and stable **100/100 parity target has not been achieved**. The correct final state is an honest research handoff, not a redefined score.

The strongest measured ordinary-position result is **99.523 cp held-out MAE** on 999 non-mate test positions from a broader 5,000-position real-game corpus. The all-position result remains **155.049 cp held-out MAE** because mate-like targets and tactical terminal values are not represented adequately by a static score head. A regenerated 120-position depth-10 direct comparison of the calibrated loaded-NNUE evalbar measured **422.575 cp MAE**, median absolute error **106 cp**, and maximum error **29,887 cp**. The large maximum is a mate-score mismatch, not an ordinary-position regression.

## Clean-room boundary

Stockfish 19 and the supplied Stockfish archive were used as external references and black-box UCI targets. The repository was audited after the supplied archive exposed copied source under historical `docs/source-evidence/` paths. Those copies were removed and replaced by provenance-only documentation. The Unchessed implementation does not import Stockfish source, weights, feature rows, or implementation fragments.

The accepted methodology is:

1. Use public research papers and upstream documentation for architectural understanding.
2. Use the official Stockfish executable only through UCI for black-box labels and comparisons.
3. Implement independent Rust/Python features, models, state updates, and calibration logic.
4. Preserve failed experiments and negative results rather than silently promoting them.

## Major implementation milestones

| Commit | Result |
|---|---|
| `6db56c1` | 300,000-position leakage-safe manifest, fixed-point residual state, exact move-delta updates, and IEEE scale-up paper |
| `03c6ad4` | Removed copied Stockfish source evidence and tightened strict parity gates |
| `6f6b108` | Recorded the affine calibration ceiling |
| `766a249` | Added the practical-readiness rubric; this was not strict SFNNv16 parity |
| `dd8a0a8` | Added the original 5,000-label teacher-calibrated evalbar experiment |
| `47939ba` | Added the broader all-position manifest and progress report |

## Current evaluator surfaces

The default search evaluator remains the existing Unchessed NNUE/HCE architecture. The experimental presentation-only modes are exposed through UCI:

```text
position startpos
evalbar homemade
evalbar parity
evalbar calibrated
```

`evalbar calibrated` applies original affine and interaction terms over the loaded evaluator score and eleven independently authored bitboard-derived signals. It is explicitly opt-in and does not alter the main search evaluator.

The current calibrated path also includes an original bounded forced-mate probe up to three plies. It uses Unchessed legal move generation and check detection to recognize immediate and short forced mates. It is not called by the main search evaluator. Focused evalbar tests and the full core suite pass after this change.

## Data and training evidence

Two important 5,000-position real-game studies were completed.

| Study | Description | Result |
|---|---|---:|
| Quiet corpus | 5,000 positions from leakage-safe game-level splits, primarily stable positions | Combined original calibration: **53.474 cp** held-out MAE on 999 non-mate tests |
| Broader corpus | 5,000 positions without the quiet filter, including tactical and endgame positions; 3,965 train / 493 validation / 542 test | Retrained original v4 NNUE: **99.523 cp** held-out MAE on 999 non-mate tests; **155.049 cp** on all 1,000 tests |

The broader corpus contained 5 mate-like labels above 4,000 cp. The regression trainer clipped those extreme targets for the static network experiment; this improved ordinary-position behavior but cannot solve mate distance by itself. The next proper architecture should use separate score/WDL/mate-distance targets rather than forcing all terminal outcomes into a single centipawn regression head.

## Speed and reliability

The incremental residual-state work improved the nonlinear/HCE NPS ratio from approximately **0.8622** to approximately **0.8896**, but it still fails a strict ≤5% overhead gate. The full core regression suite most recently passed with **150 tests passed, 6 ignored**. The calibrated UCI smoke command returned a legal result and explicit provenance label.

The forced-mate probe is intentionally outside the search hot path. If it is ever promoted into search, it must be replaced with an incremental tactical state or a tightly budgeted proof cache; recursively generating legal moves at every node would be an unacceptable performance regression.

## Why strict 100/100 is still open

The remaining gaps are substantive:

| Gate | Status |
|---|---|
| Ordinary non-mate score fidelity | Crossed on one 999-position held-out stratum at 99.523 cp MAE |
| All-position score fidelity | Not crossed: 155.049 cp held-out MAE |
| Mate-distance recognition | Only a bounded three-ply presentation probe exists |
| Deep tactical recognition | Not yet trained as a separate target/head |
| Search overhead | Not within the strict ≤5% target |
| Powered fixed-control match | Not yet established with sufficient sample size |
| Stable 100/100 claim | **Not achieved** |

The direct 120-position comparison should not be summarized without its distribution. Its median error was 106 cp, while a single mate-like position contributed nearly 30,000 cp. Both ordinary and terminal strata must be reported separately in future work.

## Recommended continuation

The next research iteration should construct a stratified teacher corpus containing quiet, tactical, checking, capture, promotion, endgame, and terminal positions with game-level splits. Each position should receive at least three targets: a bounded static score, WDL or expected score, and mate distance or terminal class. A robust multi-task original model should train these heads jointly, with explicit loss weighting and a held-out tactical set.

At inference time, mate-distance recognition should be a shallow, cached proof mechanism rather than a full recursive scan inside the ordinary evaluator. Tactical features should include legal checking-move pressure, forcing-capture availability, pinned-piece exposure, king escape count, promotion race features, and quiescence-derived bounded swing signals. Any new path must be benchmarked against the current 0.8896 NPS ratio and must not be enabled by default without a powered match.

Image recognition is useful only for board-state ingestion, UI verification, screenshot debugging, and detecting coordinate or rendering mistakes. It is not a substitute for exact FEN/bitboard state in evaluator training because images do not reliably encode side to move, castling rights, en-passant state, repetition history, or exact mate distance.

## Reproducibility pointers

| Artifact | Purpose |
|---|---|
| `unchessed-eval-bar/README.md` | Eval-bar modes and direct comparison methodology |
| `unchessed-eval-bar/results/all_position_progress.md` | Broader corpus and mate-gap results |
| `unchessed-eval-bar/results/strict_100_gate.md` | Strict gate definition and previous ceiling |
| `unchessed-eval-bar/results/combined_teacher5000_nonmate_analysis.json` | Quiet/non-mate calibration result |
| `unchessed-eval-bar/results/teacher5000_nonmate_analysis.json` | Loaded-NNUE non-mate result |
| `unchessed-eval-bar/results/stockfish19_teacher_calibrated_loaded_120.md` | Direct Stockfish 19 comparison |
| `tools/build_parity_dataset.py` | Leakage-safe manifest creation |
| `tools/label_stockfish_batch.py` | Persistent black-box UCI labeling |
| `tools/labels_to_nnue_records.py` | Original label-to-training-record conversion |
| `tools/compare_sfnnv16_homemade.py` | Direct score comparison |

## Next work: execution plan

The following sequence is the recommended continuation. It is deliberately ordered so that each expensive step produces evidence needed by the next step. Do not skip directly to a 100/100 claim: the current failure is concentrated in terminal and tactical positions, while the ordinary non-mate score gate is already close to its target.

### Stage 0 — Freeze the baseline

Create a baseline record before changing model code. Record the branch commit, compiler version, CPU description, binary checksum, evaluator file checksum, Stockfish reference checksum, corpus manifest checksums, and the exact command lines. Re-run the full core suite and the direct 120-position comparison. Preserve the resulting files under `unchessed-eval-bar/results/baseline/` rather than overwriting prior reports.

The baseline acceptance record must include the current values: 150 core tests passed with 6 ignored; ordinary held-out MAE 99.523 cp on the broader-corpus non-mate subset; all-position held-out MAE 155.049 cp; calibrated 120-position depth-10 direct MAE 422.575 cp with median absolute error 106 cp; and nonlinear/HCE NPS ratio approximately 0.8896. If these numbers move unexpectedly before implementation, stop and diagnose the environment or corpus first.

### Stage 1 — Build a stratified tactical and terminal corpus

Extend `tools/build_parity_dataset.py` or add a new script rather than manually editing label files. Preserve game-level splitting: no positions from one game may cross train, validation, and test partitions. Add explicit strata and report counts for quiet, checking, capture, promotion, low-material endgame, king-exposed, high-mobility, and terminal or mate-like positions.

Use the persistent black-box labeler in `tools/label_stockfish_batch.py`. Store the teacher protocol, depth, hash, threads, binary checksum, and date in a sidecar manifest. Keep the raw labels outside the normal source commit if they are large; commit the manifest, checksums, summaries, and small reproducibility samples. At minimum, reserve a game-disjoint tactical test set of 2,000 positions and a terminal test set of every available mate-like example.

The data gate is a minimum of 100,000 labeled positions for a serious retrain, with the tactical and terminal strata reported separately. A 5,000-position experiment is useful for iteration but is not sufficient evidence for stable parity.

### Stage 2 — Define separate targets instead of one raw centipawn head

Do not force mate scores such as 30,003 cp into an ordinary static regression target. Convert each teacher result into separate targets: bounded static score, expected score or WDL class, terminal class, and mate distance where the teacher reports a forced mate. Use a robust score loss such as Huber or clipped absolute loss for ordinary positions, a classification loss for terminal status, and a distance loss only on positions with reliable mate-distance labels.

The model should remain original and compact. A practical first design is the existing incremental NNUE score head plus a small quantized tactical head over independently authored features: checking-move pressure, legal checking destinations, forcing-capture count, king escape count, pinned or overloaded-piece indicators, promotion-race distance, and bounded quiescence swing. The tactical head should be gated by a confidence or terminal probability, not added unconditionally to every quiet position.

The training split must be by game, and hyperparameters must be chosen only on validation games. The test set must remain untouched until the model, calibration, and threshold are frozen.

### Stage 3 — Implement mate-distance recognition safely

Keep the current `forced_mate_in` probe as a correctness reference only. It recursively generates legal moves and is unsuitable for the ordinary search hot path. Use it to generate labels and unit-test positions, not as a per-node evaluator call.

For runtime, implement a bounded proof cache or incremental tactical state. A safe progression is: first recognize checkmate and stalemate exactly; then add mate-in-one; then add a depth-limited proof only when the position is already in check, has a checking move, has very low material, or has a high tactical-confidence trigger. Cache by position hash, side to move, remaining proof depth, and rule state. Do not run the proof on every quiet node.

Required correctness tests include both colors, checking captures, promotions, underpromotions, interpositions, double check, stalemate, castling rights, en-passant legality, and positions where a tempting checking move is not mate. Every new test should compare the fast path with the slow legal-move reference.

### Stage 4 — Optimize before enabling broadly

Benchmark the tactical path independently and inside search. Measure nodes per second, evaluator calls per second, p50/p95/p99 evaluation latency, allocations, cache hit rate, and branch-trigger frequency. The strict speed target is no more than 5% overhead against the same binary and time-control settings. If the tactical path exceeds that threshold, keep it opt-in and reduce its trigger rate or move more state into incremental updates.

Avoid floating-point work, repeated full-board feature scans, and repeated legal-move generation on quiet positions. Reuse move lists where the architecture permits, use compact integer features, and keep the WDL projection outside search. The presentation evalbar may be more expensive than the search evaluator, but its cost must be documented separately.

### Stage 5 — Calibrate and validate on untouched data

After freezing the model and thresholds, evaluate the untouched game-disjoint test set. Report overall MAE, non-mate MAE, tactical MAE, terminal classification accuracy, mate-distance exact accuracy, mate-distance within-one accuracy, sign accuracy, median absolute error, and worst-case error. Always show the number of examples in every stratum.

The strict score gate should require all-position MAE at or below 100 cp on a predeclared test set, not merely on a filtered non-mate subset. The terminal gate should require no missed mate-in-one cases in the dedicated suite and an explicitly reported mate-distance tolerance. The speed gate should require no more than 5% overhead. The reliability gate remains a full green core test suite and deterministic repeated-run output.

### Stage 6 — Run a powered engine match

Only after the score, terminal, speed, and reliability gates pass should the tactical head be allowed to influence the playing evaluator. Run a fixed-control match against the same Stockfish 19 reference and against the previous Unchessed baseline. Fix openings, threads, hash, time controls, adjudication, and random seeds where possible. Use enough games for an actual confidence interval; the earlier smoke tournaments are not statistically powered.

Report wins, draws, losses, average score, estimated Elo difference with uncertainty, color balance, termination reasons, average depth, nodes per second, and time forfeits. Revert the feature from default-on if it loses strength, creates instability, or fails the speed gate even when its offline MAE improves.

### Stage 7 — Documentation and release discipline

Update the IEEE paper and this handoff with the frozen dataset checksum, model checksum, exact training command, target definitions, test split, all metrics, and match configuration. Add a negative-results section for every failed model. Keep Stockfish source and weights out of the repository; black-box binaries and raw labels are reference artifacts, not implementation dependencies.

The final 100/100 claim may be made only when every predeclared gate passes on untouched data and the powered match is complete. If any gate fails, report the achieved gate vector instead of converting it into a single inflated score.

### Suggested command sequence

The next researcher should begin with commands of this form, adapting paths to the available corpus:

```sh
cd /home/ubuntu/Unchessed-UCI-Engine
sha256sum unchessed-nnue.bin target/release/unchessed-adapter
cargo test -p unchessed-core --lib
python3 tools/build_parity_dataset.py --help
python3 tools/label_stockfish_batch.py --help
python3 tools/compare_sfnnv16_homemade.py --help
```

Use a temporary compatible `Cargo.lock` only when the installed Rust toolchain requires it, and always restore the tracked lockfile before committing. Keep all changes on `manus/research-facilities`; never edit or merge the protected main branch for this research stream.

## Final interpretation

This branch contains meaningful original engineering progress: incremental residual state, leakage-safe corpus construction, black-box calibration, an opt-in teacher-calibrated evalbar, and a bounded forced-mate probe. It does not contain stable SFNNv16 parity. The repository is ready for the next researcher to continue from measured evidence without repeating the audit, data construction, provenance correction, or failed calibration experiments.
