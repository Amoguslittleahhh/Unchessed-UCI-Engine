# 27 — Tier 1: Calibrated WDL output

**Investigation ID:** `tier1-calibrated-wdl-output`
**Scope:** Research and design only. No engine behavior, UCI default, training job, label generation, or model file was changed.
**Status:** Complete; recommendation is **defer implementation pending a small outcome-labeled calibration study**.

## Decision summary

The repository has enough information to expose a *candidate* WDL mapping, but not enough evidence to call that mapping calibrated. The NNUE emits a single side-to-move scalar evaluation in the repository's own centipawn-like units. Training converts that scalar to `sigmoid(score / 400)` and blends it with a three-valued game-result field. That is a useful bounded training target, not a three-class probability distribution and not an outcome-calibrated UCI prediction.

A defensible WDL feature therefore requires a separate, held-out fit against actual game outcomes. The cheapest suitable experiment is to replay fresh, game-disjoint engine games while retaining full positions, search scores, material, move number, result, time control, and opponent identity; fit a monotone material-conditioned score-to-outcome model; and report reliability and proper-scoring diagnostics. Until that succeeds, do not add `wdl` to UCI output or describe a sigmoid of the current score as a win/draw/loss probability. Recommendation: **pursue the diagnostic, defer shipping and retraining**.

## What the repository actually does

### Evaluator and score units

`unchessed-core/src/nnue.rs` defines `SCALE = 400.0` and clamps the exported NNUE result to `[-3000, 3000]`. The runtime chooses an output bucket from piece count, computes the side-to-move-oriented network output, multiplies by 400, and returns an integer score. This is an engine evaluation scale, not a statistically established probability scale. The symmetry tests establish evaluator perspective behavior, not outcome calibration.

The UCI loop currently formats a completed search as `info ... score cp <integer>` (or `score mate <n>`), in `unchessed-core/src/uci.rs`. There is no `UCI_ShowWDL` option and no `wdl` field in the emitted `info` line. The natural protocol shape, if eventually justified, is the standard UCI extension `wdl <win-per-mille> <draw-per-mille> <loss-per-mille>`, with the order explicitly documented and values summing to 1000. It must be attached to the searched root score, not to a raw static NNUE call hidden from the GUI.

### Training target

`tools/train_nnue.py`, `_features_and_target`, currently computes:

```python
target = 0.7 * torch.sigmoid(score_f / 400.0) + 0.3 * (wdl.to(torch.float32) / 2.0)
```

The binary record schema in `tools/check_nnue_data.py` is 12 little-endian `u64` bitboards, an `i16 score`, one `u8 wdl`, and five padding bytes. The checker labels the byte values as `wdl L/D/W = 0/1/2`, so the result contribution is respectively 0, 0.5, or 1. The model is consequently trained toward one scalar in [0, 1]: 70% transformed teacher score and 30% game-result scalar.

That target is **not** three probabilities. In particular, a scalar expected score does not identify a unique `(P(win), P(draw), P(loss))` triple: many distributions have the same `P(win) + 0.5 P(draw)`. The training blend also does not guarantee that its output is calibrated to outcomes, because the score component is a teacher-evaluation transformation, the result component may reflect a particular data distribution, and the loss optimizes a blended regression target rather than a multiclass likelihood. Repurposing the scalar as “win probability,” or treating `1 - scalar` as loss and inferring draw by subtraction, would be an unsupported relabeling.

## What “calibrated WDL” should mean

For a declared population and condition set, a prediction is calibrated when cases assigned probability `p` have the corresponding empirical event frequency. This is conditional on the opponent population, engine strength, time control, adjudication/draw rules, and position-selection policy. It is not an intrinsic property of a centipawn number. Stockfish's official WDL model makes this dependence explicit: it fits engine-vs-engine fishtest games, uses a logistic win-rate model, and conditions the fitted parameters on material. Its documentation says that 100 centipawns is calibrated as a 50% win probability for its stated self-play condition, while its FAQ warns that opponent and time control change win/draw rates.

The appropriate output is three mutually exclusive probabilities from a single model, for example:

```text
W(x,m) = sigmoid((x - a(m)) / b(m))
L(x,m) = W(-x,m)
D(x,m) = 1 - W(x,m) - L(x,m)
```

Here `x` is the score whose semantics are being calibrated, `m` is a declared covariate such as material, and `a(m), b(m)` are fitted on outcome data. The symmetry is a modeling choice that should be tested, not silently assumed. The fitted parameters must preserve `b(m) > 0` and nonnegative probabilities; if the unconstrained model violates those conditions, use a constrained fit or a multiclass model instead.

A simpler first diagnostic can fit one global symmetric logistic mapping, then compare it with a material-conditioned mapping. A more flexible alternative is multinomial logistic regression on score plus material features. Isotonic calibration is a possible later comparator, but it needs substantially more data and can overfit; a strictly monotone sigmoid is preferable for a first cheap diagnostic because it preserves score ranking. These are calibration maps, not claims that the underlying search score is a ground-truth win rate.

## Proposed cheap calibration diagnostic (no implementation in this item)

### Data and provenance

Use fresh games generated or replayed with the exact candidate engine, evaluator net hash, search settings, opponent setting, and time control that the eventual output is meant to describe. Record a full legal position at every sampled ply, side-to-move, root search score, depth/nodes/time, material count, game/move number, game result from the side-to-move perspective, and termination/adjudication reason. Do not reconstruct positions from the archived NNUE binary records: prior repository work documents that those 104-byte records do not retain castling rights, en-passant state, or a complete replayable position.

Sample at a modest fixed interval and stratify or reweight by material and phase so the diagnostic is not dominated by opening positions. Split by **game**, never by position, so positions from one game cannot appear in both fitting and evaluation. Reserve a final untouched holdout. If using existing PGNs, bind the exact PGN and engine/evaluator metadata with hashes and report the selection policy.

The target is the terminal outcome, encoded as one-hot `(W,D,L)` from the relevant side's perspective. Exclude positions after resignation/termination only according to a predeclared rule; do not silently treat a resignation as a forced engine win. Mate scores require a declared policy: either map them to saturated probabilities only for a separate terminal/mate analysis, or exclude them from the finite-score fit and report their coverage. Never feed a sentinel mate integer through the ordinary sigmoid.

### Fits and baselines

Fit and compare the following on the training split only:

1. **Empirical material-bin baseline:** outcome frequencies by score bin and material bucket, smoothed or held out as appropriate.
2. **Global symmetric logistic:** `W(x)=sigmoid((x-a)/b)`, `L(x)=W(-x)`, `D=1-W-L`.
3. **Material-conditioned symmetric logistic:** cubic or low-degree functions for `a(m)` and `b(m)`, following the structure documented by Stockfish, with constrained positivity.
4. **Current-score/blend proxy:** report the existing `sigmoid(score/400)` and the training scalar `0.7*sigmoid(score/400)+0.3*result/2` only as scalar baselines. Do not call either a WDL model. A three-way proxy may be shown only as an explicitly labeled heuristic and must not be promoted.

The diagnostic should first test whether the current NNUE score has useful monotone information. If score ordering is poor, fitting a more elaborate WDL curve is cosmetic. If ordering is reasonable but reliability is poor, calibration may still be useful for display while leaving search and training untouched.

### Metrics and acceptance gates

Report per-class log loss, multiclass Brier score, and a reliability diagram for each class. Use equal-count bins with bin counts and confidence intervals, plus expected calibration error (ECE) as a descriptive summary. Report class frequencies, calibration by material/phase, and a held-out confusion table at maximum-probability class. Brier and log loss combine calibration, discrimination, and outcome uncertainty; therefore do not use a lower aggregate score alone as proof of calibration. Compare against the empirical baseline and include bootstrap confidence intervals clustered by game.

Cheap acceptance gates for a future implementation are:

| Gate | Requirement | Meaning |
|---|---|---|
| Data integrity | No illegal/reconstructed positions; game-disjoint split; complete result perspective; immutable hashes | The measurement is interpretable |
| Probability validity | Every output is finite, nonnegative, and sums to 1000 per mille after rounding | UCI-safe WDL |
| Held-out calibration | Reliability curves are materially closer to the diagonal than the uncalibrated proxy in the dominant score/material strata; no severe class-specific failure | Mapping adds honest information |
| Proper scoring | Held-out multiclass log loss and Brier score beat the current proxy and empirical baseline within uncertainty | Improvement is not visual-only |
| Stability | Parameters and metrics remain similar across game/time-control or opening folds | Not a narrow artifact |
| Scope | Fit condition, opponent, time control, and score source are documented in the option/report | Users know what probability means |

There is no universal numeric ECE threshold that makes a chess WDL model true. A candidate that only wins on a random, leakage-prone split should be rejected. Passing this diagnostic still does not establish playing strength and does not justify changing the NNUE training target.

## Implementation boundary if the diagnostic passes

A later Tier 2 implementation should be deliberately small: add an opt-in `UCI_ShowWDL` boolean defaulting to false, store a versioned calibration table and provenance, calculate WDL from the final root score and current position's material, and append `wdl W D L` to normal `info` lines. It should not alter search comparisons, aspiration windows, pruning, NNUE inference, training labels, or score reporting. Mate and tablebase scores need explicit protocol behavior and tests. The option's documentation must state that probabilities apply to the calibration population and are not universal odds against an arbitrary human or engine.

The first implementation should prefer a compact pre-fitted table or low-degree formula over run-time fitting. Add unit tests for score sign, color-flip symmetry, material clamping, probability bounds, per-mille rounding/sum, and mate handling. Validate UCI clients with an opt-in smoke test. Only after a real paired-game comparison shows no unintended engine behavior should the feature be considered for a default change; display output itself is normally strength-neutral, but incorrect WDL can mislead users and tools.

## Literature and repository evidence

The official [Stockfish WDL model repository](https://github.com/official-stockfish/WDL_model) describes fitting logistic win rate from large engine-vs-engine game collections, with material-dependent parameters and the symmetry formulas used above. The [Stockfish FAQ](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html) explicitly qualifies the curves by opponent and time control and distinguishes near-certain draws at zero from a generic “50% win” interpretation. The [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) explains converting CP-like outputs to WDL space with a sigmoid and blending evaluation and game-result losses; this supports the distinction between a training target and an outcome-calibrated three-class prediction. The [scikit-learn probability-calibration guide](https://scikit-learn.org/stable/modules/calibration.html) defines reliability diagrams and cautions that Brier/log loss mix calibration with discrimination and uncertainty; it also describes sigmoid versus isotonic calibration and isotonic overfitting risk.

Repository evidence is limited but clear: `tools/train_nnue.py:287–300` contains the scalar blend; `tools/check_nnue_data.py` defines the record and WDL byte convention; `unchessed-core/src/nnue.rs:78,596–616` defines the scale, bucketed evaluation, and clamp; and `unchessed-core/src/uci.rs:1307–1318` emits only `score cp`/`score mate`. Reports `docs/reinforcement/02-nnue.md`, `08-label-noise.md`, and `11-tier1-synthesis.md` consistently require explicit score/POV/teacher provenance, warn that score agreement is not calibration or playing strength, and prohibit inventing labels from incomplete archived state. Those constraints apply directly here.

## Work performed and not performed

I read the master brief, reinforcement reports `00`–`12`, the NNUE and label-noise documentation, the training target and binary schema, the NNUE evaluator, and the UCI formatter. I fetched and read the official Stockfish WDL model, Stockfish FAQ, Stockfish NNUE documentation, and scikit-learn calibration guidance. I did **not** generate games, fit parameters, compute calibration metrics, modify source code, add a UCI option, retrain a net, run an Elo match, or claim calibrated numbers. No defaults or artifacts other than this report were changed.

**Final recommendation:** pursue the bounded, game-disjoint calibration diagnostic; **defer implementation and any training-target change** until it demonstrates held-out reliability and proper-score improvement under a declared opponent/time-control condition. If suitable outcome data cannot be produced with full legal state and provenance, drop WDL output rather than shipping a repurposed loss scalar.

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/27-wdl-output.md`

## References

[1]: https://github.com/official-stockfish/WDL_model "official-stockfish/WDL_model"
[2]: https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html "Stockfish FAQ"
[3]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html "Stockfish NNUE documentation"
[4]: https://scikit-learn.org/stable/modules/calibration.html "scikit-learn probability calibration"
[5]: ../nnue-label-noise-real-measurement.md "Repository NNUE label-noise measurement"
[6]: ./08-label-noise.md "Reinforcement label-noise report"
[7]: ./11-tier1-synthesis.md "Tier 1 synthesis"
