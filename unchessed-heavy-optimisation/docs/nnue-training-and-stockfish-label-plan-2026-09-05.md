# NNUE Training, Board-State Capture, and Stockfish-Quality Labeling Plan

**Project:** `unchessed-heavy-optimisation`  
**Date:** 5 September 2026  
**Author:** Manus AI

## Executive conclusion

The next Unchessed NNUE improvement should not begin by copying a Stockfish `.nnue` file or by changing the network width. Unchessed already uses a Stockfish-compatible HalfKAv2_hm-style feature geometry, training-time virtual factorization, and incremental accumulators. The highest-confidence weakness is **label and data provenance**, not the basic feature-transformer concept.

The recommended sequence is to preserve the incumbent network, create a legally complete and provenance-rich record format, relabel the same frozen positions with stronger and better-documented teachers, train controlled objective/loss ablations, and promote only candidates that pass fixed-position checks followed by a real paired-game SPRT. This sequence is more likely to reduce effective centipawn error and increase playing strength than an unvalidated architecture transplant.

A lower offline loss is not equivalent to a stronger engine. The local project already documents a 108M-record v4 run with approximately 47.8 centipawns of validation MAE that nevertheless lost approximately 155.6 Elo to the shipped v3 network. That result is a direct warning that **validation loss is a health metric and selection signal, not the release criterion**.

## What the established engines actually teach us

| System | What is well supported by public sources | What should be transferred to Unchessed |
|---|---|---|
| Stockfish NNUE | Sparse king-relative features, incremental accumulators, low-precision CPU inference, factorized training features, searched evaluation targets, and game-based network selection. The current trainer supports calibrated score/WDL targets and fake quantization. [1] [2] [3] | Transfer the training discipline and quantization-aware methodology. Do not copy Stockfish weights or current-master features into the incompatible `UNCHNNUE` format. |
| Ethereal / NNTrainer | Public historical networks use mirrored king-relative factorized features, blended search-score/WDL targets, and repeated large-scale teacher-data regeneration. The exact current commercial recipe is not fully public. [4] [5] | Run controlled teacher-regeneration experiments and preserve full provenance. Do not copy undocumented optimizer constants or assume the public trainer exactly describes current releases. |
| Berserk | A documented historical NNUE used mirrored king buckets, two perspective accumulators, a blended score/WDL target, and billions of self-play positions. Its current architecture is different from the historical release. [6] [7] | Preserve separate side-to-move and non-side-to-move accumulation semantics and use self-play only with immutable holdouts and explicit teacher metadata. |
| AlphaZero / Lc0 | Policy targets are soft search distributions, while value targets are game outcomes. Lc0 records rich board, policy, value, visit, and provenance information. [8] [9] | Use soft teacher distributions for a separate move-ordering model if desired. Do not replace Unchessed alpha-beta search with MCTS merely because the data format is richer. |

Stockfish’s current feature sets, such as threat and pawn-pair features, are legitimate research directions. They are not low-risk upgrades. Each requires a new feature index, accumulator update path, serialization format, trainer/export implementation, numerical parity tests, new training data, and a fresh SPRT.

## How to reduce effective CP-value loss

### 1. Fix the target before tuning the loss

Unchessed currently trains in probability space using a project-specific target of approximately:

```text
0.7 × sigmoid(search_cp / 400) + 0.3 × (wdl / 2)
```

with a power-2.5 error. This is a reasonable hybrid target, but the `/400` calibration and 70/30 mix are not universal Stockfish constants. Stockfish’s public trainer also maps search scores into calibrated WDL-like probabilities and supports a blend with game outcomes, but its exact scale, target mix, and feature configuration are independently configurable. [2] [3]

The first experiment should therefore compare, on the **same positions and same train/validation split**:

| Candidate | Search-score component | Game-outcome component | Purpose |
|---|---:|---:|---|
| Baseline | 0.70 | 0.30 | Existing Unchessed objective |
| Calibration A | 0.50 | 0.50 | Historical Ethereal/Berserk-style hypothesis |
| Calibration B | 0.85 | 0.15 | More faithful to searched evaluation |
| Calibration C | Fitted scale | 0.30 | Fit the logistic CP scale on held-out teacher labels |

The scale must be fitted only on training data and then frozen before validation. Otherwise, the calibration set becomes part of the training decision.

Report at least validation MAE in centipawns, probability error, WDL calibration error, rank correlation with teacher scores, sign-flip rate, and error by phase, material, tactical status, and score magnitude. A single aggregate MAE can hide catastrophic tactical errors.

### 2. Improve labels before increasing model capacity

The current NNUE data generator uses a fixed shallow local HCE search at approximately 5,000 nodes and stores a compact record. This is cheap, but it creates a ceiling if the target is noisy or systematically biased. The existing local stronger-label study correctly proposes relabeling the **same positions** with deeper HCE or self-distillation instead of comparing two differently filtered corpora.

Create three teacher sidecars over the same frozen position IDs:

1. **Baseline teacher:** current 5,000-node HCE, reproduced exactly.
2. **Deeper local teacher:** a larger fixed-node or fixed-depth search with identical engine code and explicit options.
3. **Independent teacher:** a pinned Stockfish release and explicitly pinned Stockfish NNUE, used only if licensing, distribution, and runtime constraints are acceptable.

The independent teacher should not be called “ground truth.” It is another approximation. The useful question is whether it produces labels that improve Unchessed’s game strength after retraining.

Every label must carry the teacher binary hash, network hash, engine commit, evaluator mode, thread count, hash size, search limit, score perspective, completion status, and timestamp or deterministic generation identifier. A label without this metadata is not suitable for a strength experiment.

### 3. Use hard-example weighting carefully

After the first controlled teacher comparison, consider a bounded weighting scheme. Increase sampling for positions where teachers disagree, where the side-to-move has a forcing move, or where the evaluation crosses zero. Do not simply overweight every tactical position. A noisy teacher disagreement can otherwise dominate training.

A safe first weighting table is:

| Position class | Initial weight |
|---|---:|
| Quiet, teacher agreement within 15 cp | 1.0 |
| Teacher disagreement 15–75 cp | 1.25 |
| Sign disagreement or best-move disagreement | 1.5, capped |
| Mate, tablebase, illegal, incomplete, or corrupted label | Reject or isolate |

The weights must be applied identically to train and evaluation metrics, and the candidate must still pass a normal game SPRT. Weighting is a hypothesis, not a guarantee of lower playing error.

### 4. Stop exporting the last checkpoint

The local trainer already contains the correct direction: early stopping and best-checkpoint export. Keep a patience threshold based on validation MAE, but save multiple checkpoints around the minimum. The final winner must be chosen by games among checkpoints, because a lower validation error can correspond to a weaker evaluator.

For each training condition, use at least three seeds when resources permit. Retain the incumbent as the control. Do not promote a candidate because it wins one small match or because its loss curve looks smoother.

### 5. Quantize only as a retraining project

Stockfish’s speed comes partly from int16 accumulators, int8 activations and weights, and int32 dense accumulation. Unchessed currently uses f32 inference and its own `UNCHNNUE` format. The local audit measured a large cache and throughput motivation for quantization, but it also found that the current weights exceed a naive int8 range and that post-hoc conversion fails numerical parity.

The safe quantization project is:

1. Add fake activation and weight quantization during training.
2. Define clipping and scale metadata in the model format.
3. Export an explicitly versioned quantized file.
4. Compare Rust scalar and SIMD inference against a high-precision reference.
5. Measure CP drift over a fixed corpus and tactical suite.
6. Benchmark accumulator update and forward-pass latency.
7. Run a separate paired-game SPRT.

Do not round the existing f32 file and call it a production quantized net.

## What board information is required to obtain a Stockfish-quality label?

A board diagram alone is not sufficient. It can reproduce a static evaluation, but it cannot always reproduce legal moves, repetition rules, fifty-move behavior, or a teacher search exactly. The record must distinguish **position state**, **game-history state**, and **teacher metadata**.

### Position-state fields

The minimum complete position state is equivalent to a six-field FEN:

| Field | Required information | Why it matters |
|---|---|---|
| Piece placement | All 64 squares, or 12 piece bitboards | Legal move generation and NNUE features |
| Side to move | White or Black | Score perspective, legal moves, search root |
| Castling rights | White king/queen side and Black king/queen side | Legal castling and rook/king rights after prior moves |
| En-passant target | Exact square, not only the file | Legal en-passant capture and NNUE state |
| Halfmove clock | Plies since the last pawn move or capture | Fifty-move draw claims and rule-aware search |
| Fullmove number | Move number | Reproducibility and audit identity; it usually does not change chess evaluation directly |

Unchessed’s current compact NNUE record stores piece bitboards, score, and WDL, but the legacy layout is not sufficient for legally exact re-searching because it does not preserve all state needed by the rules. The new record version must not discard castling, exact en-passant state, or the halfmove clock.

### Game-history fields

For a strict Stockfish-equivalent search, retain either the complete move prefix or a repetition-safe history representation. At minimum, preserve the previous positions’ repetition keys since the last irreversible move. A position’s board diagram can be identical while the repetition claim state differs.

Also retain the game result and adjudication state if using outcome supervision. A resignation, tablebase adjudication, maximum-length draw, or unfinished game must not be silently converted into an ordinary win, draw, or loss.

### Teacher-search fields

To know what produced a label, record:

| Category | Fields |
|---|---|
| Engine identity | Engine name, exact commit or release, binary SHA-256 |
| Evaluation identity | `EvalFile` path, NNUE SHA-256, evaluator mode, policy mode |
| Search resources | Threads, Hash, node/depth/time limit, MultiPV, ponder state |
| Search result | Root score, score POV, WDL if available, depth, seldepth, nodes, nps, completion flag |
| Move information | Best move, MultiPV alternatives, per-move scores or root visit probabilities, principal variations |
| Environment | CPU tier, compiler flags, operating system, deterministic seed, opening/source ID |

The teacher must emit a label only after it has returned a valid completion signal. A timeout, crash, malformed UCI line, illegal move, or missing evaluator file must produce a rejected record, not a guessed score.

## How to collect the best move rather than only a scalar evaluation

The best move is a search result, not a property that can be inferred from the board tensor alone. The reliable pipeline is:

1. Reconstruct the complete legal position from the new record.
2. Generate and count legal root moves independently.
3. Start the pinned teacher with explicit `EvalFile`, threads, hash, and search limit.
4. Request either a fixed node limit or fixed depth. Do not mix limits within one comparison.
5. Request `MultiPV=N` for a chosen small N, or run root-restricted searches for every legal move when a full distribution is required.
6. Parse only completed `bestmove` and `info` records whose position token, search token, and evaluator hash match the request.
7. Verify that every returned move is legal in the reconstructed position.
8. Normalize scores to the side-to-move perspective and convert them to a documented probability scale.
9. Store the best move, alternative moves, scores, depths, nodes, and principal variations.
10. Reject incomplete or contradictory labels.

`MultiPV` is efficient for a top-ranked subset, but it does not guarantee a score for every legal move. If the policy model needs a probability for every legal move, use a two-stage method: request a broad MultiPV shortlist, then run root-restricted searches for the remaining legal moves under a fixed per-move budget. The result is expensive, but it produces an interpretable teacher ranking rather than pretending that a one-hot best move is a complete policy target.

A practical soft target for move ordering is:

```text
q_i = sigmoid((score_i - score_best) / temperature)
pi_i = q_i / sum_j(q_j)
```

The score difference must be computed from the same side-to-move perspective. The temperature must be selected on a validation set and then frozen. For a stronger teacher, preserve the raw scores as well as the normalized target so the target can be recalibrated later.

This target is **not** equivalent to AlphaZero’s MCTS visit distribution. AlphaZero and Lc0 use searched visit counts, while an alpha-beta teacher normally provides scores and principal variations. The correct description is “score-derived soft root policy,” not “AlphaZero policy.” [8] [9]

## Recommended Unchessed record version

Do not mutate the existing 104-byte record. Introduce a versioned record or container with a manifest. A practical initial schema is:

| Section | Suggested contents |
|---|---|
| Fixed position | 12 piece bitboards, side to move, castling mask, exact en-passant square, halfmove clock, fullmove number |
| History | Repetition keys or a compressed move prefix since the last irreversible move |
| Identity | Game ID, source shard, position index, ply, opening ID, generator seed |
| Outcome | WDL, adjudication type, result perspective, plies to termination if known |
| Teacher | Binary/net hashes, engine commit, evaluator mode, threads, hash, limit, score and WDL |
| Move labels | Best move, legal move list, MultiPV alternatives, scores, PVs, completion flags |
| Integrity | Schema version, record length, CRC or hash, manifest SHA-256 |

The format should have a compiler-free verifier that checks legal state, move legality, score perspective, finite numeric values, probability normalization, and file-level checksums before training begins.

## Training and evaluation gates

The following gates should be mandatory for every new NNUE candidate:

| Gate | Pass condition |
|---|---|
| Schema gate | Every record is complete, legal, checksummed, and provenance-bound. |
| Feature gate | Rust runtime features and the training/export feature extractor agree on a corpus of positions, including castling, en-passant, promotions, and king-bucket changes. |
| Label gate | Teacher completion, score POV, engine/net hashes, and search limits are present. Invalid or timed-out labels are rejected. |
| Offline gate | Candidate improves or preserves held-out MAE, calibration, sign accuracy, tactical buckets, and special-rule subsets. |
| Fixed-position gate | Candidate does not introduce unexplained score drift, illegal PVs, or material tactical regressions on a fixed suite. |
| Speed gate | Evaluation latency, accumulator update cost, memory footprint, and NPS are measured on portable and x86-64-v3 builds. |
| Game gate | Candidate passes a predeclared paired-game SPRT against the incumbent with fixed options, colors, openings, and time control. |
| Stability gate | No crashes, illegal moves, UCI protocol failures, evaluator-load failures, or persona-transition violations occur in the full match log. |

The game gate must remain the release authority. A small match, a lower CP loss, or a Stockfish-looking architecture is not enough.

## Concrete execution order

**Phase 1: provenance and state correctness.** Implement the new complete-state record and verifier. Add tests for castling, exact en-passant, halfmove clock, repetition, promotions, and side-to-move score normalization. Do not train yet.

**Phase 2: fixed-position relabeling.** Select a frozen, representative corpus stratified by phase, material, tactics, and rating/source. Generate baseline, deeper-local, and independent-teacher sidecars. Measure disagreement and label completion rate.

**Phase 3: objective ablation.** Train baseline, 50/50, 85/15, and fitted-scale candidates with identical data, seeds, batch size, and export path. Save multiple checkpoints and compare offline metrics.

**Phase 4: search-policy dataset.** If move-ordering supervision is desired, collect legal move lists and score-derived soft distributions. Keep this model separate from the NNUE evaluator and keep it off by default until calibration and speed gates pass.

**Phase 5: promotion testing.** Run a reject-only fixed-position screen, then a paired short-control SPRT, then a slower confirmation control. Test one material change at a time. Preserve the incumbent if the result is inconclusive.

**Phase 6: quantized branch.** Only after a stronger f32 candidate exists, develop fake-quantized training and a new runtime format. This is a separate candidate branch, not a prerequisite for improving the current evaluator’s label quality.

## Final recommendation

The best near-term bet is **not** a larger network. It is a stronger and more reproducible teacher-label pipeline with complete board state, leakage-resistant validation, explicit score calibration, and game-based promotion. That approach directly addresses low CP-value loss, brute-force compatibility, and stability at the same time.

The phrase “get the best move from Stockfish evaluation” should be operationalized as: **run a pinned Stockfish search on a complete legal state, collect completed root scores and legal moves under explicit resource limits, and preserve enough metadata to reproduce the result**. The board tensor supplies the position. The search supplies the move. The manifest proves which search supplied it.

## References

[1]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html "Stockfish NNUE documentation"
[2]: https://github.com/official-stockfish/nnue-pytorch "Official Stockfish NNUE PyTorch trainer"
[3]: https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_architecture.h "Stockfish NNUE architecture source"
[4]: https://github.com/AndyGrant/NNTrainer "Ethereal public NNTrainer repository"
[5]: https://github.com/AndyGrant/Ethereal/releases/tag/v14.00 "Ethereal 14 release notes"
[6]: https://github.com/jhonnold/Berserk/releases/tag/9 "Berserk 9 release notes"
[7]: https://github.com/Luecx/CudAD "CudAD training framework"
[8]: https://arxiv.org/html/1712.01815v1 "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
[9]: https://lczero.org/dev/overview/ "Leela Chess Zero development overview"
[10]: https://github.com/official-stockfish/fishtest "Stockfish Fishtest repository"
[11]: https://github.com/official-stockfish/fishtest/wiki/Creating-my-first-test "Fishtest test creation guide"
[12]: https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/tree/manus/research-facilities/unchessed-heavy-optimisation "Unchessed heavy-optimization branch"
