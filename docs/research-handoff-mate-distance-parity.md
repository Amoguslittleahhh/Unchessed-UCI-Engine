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

## Final interpretation

This branch contains meaningful original engineering progress: incremental residual state, leakage-safe corpus construction, black-box calibration, an opt-in teacher-calibrated evalbar, and a bounded forced-mate probe. It does not contain stable SFNNv16 parity. The repository is ready for the next researcher to continue from measured evidence without repeating the audit, data construction, provenance correction, or failed calibration experiments.
