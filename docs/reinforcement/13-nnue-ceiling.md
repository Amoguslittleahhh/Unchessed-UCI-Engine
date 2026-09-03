# 13 — NNUE ceiling: capacity versus reachable data

**Investigation ID:** `13-nnue-ceiling`
**Tier:** 1 (cheap research/design)
**Repository/branch:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/research-facilities`
**Decision:** **Closed for piece-count resampling/reweighting at this scale.** Capacity versus broad reachable-data support remains untested; neither is claimed as the cause of the ceiling. No implementation, retrain, default change, Tier 2/3 work, cloud spend, commit, or push was started.

## Executive answer

The new paired label measurement materially changes the ceiling analysis. Fresh labels generated during PGN replay differed from the existing labels by only **17–22 cp MAE**, with **Pearson 0.93–0.98** across four PGN sources and 10x/20x depth multipliers ([real measurement][1]). Thus the old simulated 50–56 cp noise-floor argument is no longer a defensible primary explanation.

The repository evidence does **not** isolate architecture from data. The two new real experiments close the piece-count intervention axis: hard resampling improved **4 of 7 rare/mid buckets by 5.7%–37.2% relative MAE**, while the two common buckets (57% of real games) regressed **10.9% and 13.9%**; clipped inverse-frequency soft weighting **[0.25×, 20×]** then made bucket 2 **19.5% worse**. The rare pools had only **44 and 1,030 examples out of 98,000**, consistent with overfitting rather than learning generalizable rare-position structure.

Capacity remains plausible but is not the best first bet. The v4 feature scheme is a checked implementation of Stockfish-style `HalfKAv2_hm`, with a 256-wide accumulator, SCReLU, two perspectives, factorized training table, and eight piece-count output heads. The exported model has **5,767,937 parameters** and the architecture audit found the feature mapping correct. Yet no full-scale width/depth ablation exists, and v4 did not beat v3 despite being structurally more modern. That negative result is ambiguous: v4 was trained from scratch on the available corpus, uses a different feature/head/training recipe, and was not a matched-capacity experiment against a fixed held-out reachable-position set.

**Working ranking:** (1) effective coverage/distribution and sampling efficiency, (2) train/evaluation distribution mismatch and remaining teacher bias, (3) architecture capacity. The ranking should be reversed only if a matched width/feature ablation shows a clear held-out gain on the same reachable-position strata while coverage is held constant.

## Evidence ledger

| Question | Verified repository evidence | What it supports | Limit |
|---|---|---|---|
| Did more records help? | 0.959M, 9M, 27M and 108M v4 results improved in that order; 27M→108M improved the SPRT gap by about 89 Elo [2]. | Data is not irrelevant; the old corpus was underpowered. | It is raw record count, not deduplicated/effective coverage; each point also has recipe/checkpoint caveats. |
| Are returns linear? | 10x (0.959M→9M) produced a much larger gain than 3x (9M→27M); the report calls the trend diminishing returns [3]. | More of the same distribution is increasingly inefficient. | Only four data points and not a controlled coverage measurement. |
| Is architecture obviously too small? | v4 uses the reference-like `HalfKAv2_hm`, 32 king buckets, 256 accumulator, factorized training table and 8 piece-count heads; audit confirms the table [4]. | No known feature-index or bucket defect explains the plateau. | Correctness is not adequacy; there is no matched width sweep at 108M. |
| Was the old label-noise floor real? | Real paired labels: 17–22 cp MAE and Pearson 0.93–0.98 [1]. | The former 50–56 cp simulated floor must be retired. | Agreement with this alternative teacher does not prove label correctness or Elo benefit. |
| Is data quality/distribution relevant? | The generator applies quiet filters; on the measured corpus M1 rejected 21.5%, M2 rejected 16.5% of eligible positions, and 43,275/79,267 candidates were accepted [5]. The committed corpus has no TimeControl headers and required disabling the base-time gate. | Selection materially changes support and can discard tactical/reachable regions; provenance matters. | No end-to-end retrain comparing coverage strata has been run. |
| What does NNUE practice say? | Official Stockfish documentation describes feature-factorization and multiple architecture families, while its training-dataset guidance says dataset quality is empirical, more positional data may help, and the best practice mixes datasets because coverage/concepts matter [6][7]. | Neither parameter count nor raw records alone is a reliable predictor. | Stockfish's results and hardware are not local Elo evidence. |

## Capacity analysis

The v4 trainer is not a toy model. Its exported feature transformer is `22528 × 256`, followed by two 256-element perspective activations and an eight-row output head; the repository records 5.77M parameters. The input representation is sparse and king-bucketed, so capacity is not well described by only the final linear layer: it depends on whether frequently visited feature combinations can share useful weights and whether rare buckets receive adequate gradient. The training-time virtual/factorized table explicitly addresses sparse-row gradient starvation, which weakens (but does not eliminate) the case that v4 simply cannot fit the data.

The architecture audit found the `HalfKAv2_hm` mapping byte-identical to the Stockfish reference for the relevant half of the king-bucket table. It also records that Stockfish has historically increased hidden-layer widths and explored feature sets such as `Full_Threats`; therefore capacity changes can matter in principle. But those historical changes are accompanied by a new training corpus and empirical testing. They are not evidence that widening this net, in isolation, will recover the missing 108–156 Elo.

The 108M run also reached **47.8 cp validation MAE**, below the prior modeled 50–56 cp noise floor. That is important, but validation MAE is on the trainer's random 2% split and may share the same generator distribution. A low random-split error can coexist with poor performance on underrepresented reachable states. It therefore cannot distinguish capacity from coverage.

## Effective-data and reachable-position analysis

Raw positions are not independent observations. A self-play shard can contain long adjacent trajectories, repeated openings, repeated material/king configurations, or many quiet positions that differ only slightly. For a sparse NNUE, the relevant question is whether the training stream supplies enough independent variation for each important feature interaction, not whether the file contains enough bytes. A useful mental model is **effective coverage**: the mass of distinct, reachable, decision-relevant strata represented after filtering and deduplication, weighted by deployment frequency and tactical importance.

The repository's generation path samples from PGNs, caps positions per game (`NNUE_MAX_PER_GAME=12`), filters by ply and acceptance, and applies M1/M2 quietness checks. These are sensible safeguards, but they can create selection bias. In particular, a static evaluator cannot represent every tactical value visible only after search; excluding unstable positions reduces label conflict but may under-cover exactly the boundary positions where search makes decisions. Conversely, a random split of neighboring positions can overstate generalization. The absence of TimeControl headers in the committed corpus also means the base-time provenance gate had to be disabled, another reason not to equate record count with quality.

The official Stockfish NNUE dataset guidance is unusually direct: “We don't really know”; datasets with more positional evaluations may be better; better evaluations do not always produce better results; and datasets are chosen empirically. It also says that training usually benefits from more than one dataset and that Lc0-derived data can improve coverage/concepts, while training solely on such data can be worse from scratch. This is consistent with a coverage hypothesis, but it does not license importing a Stockfish corpus or assuming its distribution transfers to Unchessed.

## Why the label result does not settle the question

The 17–22 cp paired-label MAE and high Pearson correlation show that the tested alternative labels are close to the existing labels. They do not show that either teacher is the true static value, nor that a net trained on them will play better. A label can be highly correlated yet systematically biased in a strategically important stratum. Also, the measurement used fresh PGN replay and does not retroactively establish labels for the unavailable 108M source records. The result removes a formerly dominant explanation; it does not identify the replacement.

## Cheap falsifiable diagnostic

Run a **matched reachable-coverage/capacity matrix** on a small, reproducible subset (for example 1–5M records or the largest CPU-feasible subset), with no deployment change and no cloud spend:

1. Build a canonical position key including all state needed for legal identity (board, side to move, castling and en-passant where available). Keep game-disjoint train and test games. Do not call byte-level record equality “position diversity.”
2. Produce a coverage report for the existing stream: exact/near duplicate rate, unique-position count, occupancy by king bucket, total piece-count band, material imbalance, game phase/ply band, and label/evaluation quantiles. Report both raw counts and inverse-frequency-weighted deployment-style counts. If full state is unavailable, mark the key as approximate and do not claim legal deduplication.
3. Construct two equal-record training sets: **A**, the existing random stream; **B**, a coverage-balanced sample stratified by king bucket × piece-count band × phase/material/eval quantile, with a held-out set sampled from fresh games and the same strata. Keep optimizer, steps seen, seed policy, label source, and record count fixed.
4. Train two capacity variants on each set: the current v4 and a deliberately narrower model (for example half accumulator width, with a separately documented ABI/export path). A wider model is optional; do not spend on it until the narrow/current comparison is informative. Evaluate per-stratum MAE, not only aggregate random-split MAE.
5. Predeclare the decision rule. **Coverage supports the hypothesis** if B improves held-out error on rare/deployment-weighted strata by a practically meaningful margin (suggested threshold: ≥5% relative MAE improvement in at least two predeclared strata) while aggregate common-stratum error does not materially regress, and if the gain persists across seeds. **Capacity supports the hypothesis** if the current model has a clear train–test gap or the wider/narrower matched comparison changes held-out MAE consistently across strata at fixed coverage, with the gain not explained by altered sampling. If neither occurs, defer both interventions and revisit labels/feature representation.

This test is cheap because it uses existing training code, CPU-feasible subsets, and offline metrics. It is falsifiable: a balanced set that does not improve rare-stratum held-out error rejects “coverage is the immediate bottleneck,” while a width change that does not improve a fixed-coverage test rejects “capacity is the immediate bottleneck.” It is not a substitute for a playing-strength test.

A stronger but still cheap variant is a **learning-curve by unique-position mass**: train current v4 on 0.25×, 0.5×, 1× and 2× *deduplicated/coverage-balanced* sample mass, using a fixed held-out fresh-game set. A continuing slope in weighted held-out MAE argues for data; a flat curve with a train–test gap argues for capacity or representation. The curve must use steps/optimization budget normalized explicitly; fixed epoch counts are not comparable across data sizes.

## Closure of the piece-count axis

The hard-resampling result is **not a clean win**: improving four rare/mid material buckets by 5.7%–37.2% relative MAE came at the expense of common buckets representing 57% of real games. The soft-weighting follow-up was negative for the intended rare-bucket rescue: **bucket 2 regressed 19.5%**, with rare pools of only **44 and 1,030 examples out of 98,000**. Per the predeclared rule in the supplied soft-reweighting design, **piece-count-bucket resampling and reweighting are closed at this data scale**. These results do not prove that the model lacks capacity or that more raw data is the answer.

The three referenced experiment files (`docs/nnue-coverage-capacity-matrix-178m.md`, `docs/nnue-soft-reweighting-result.md`, and `docs/reinforcement/32-nnue-soft-reweighting.md`) are absent from the working tree and reachable Git history in this checkout. The exact figures above are verified against `pasted_content_7.txt`, not independently recomputed here; no missing bucket names, absolute MAEs, seeds, or logs are invented.

## Mandatory real-world check

I ran the existing release engine on real positions with the shipped NNUE:

```text
printf 'uci\nsetoption name EvalFile value /home/ubuntu/Unchessed-UCI-Engine/unchessed-nnue.bin\nisready\nposition startpos\ngo depth 3\nposition fen r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 2 5\ngo depth 3\nquit\n' | timeout 30s target/release/unchessed-adapter 2>&1
```

Exact relevant output:

```text
info string [Unchessed] NNUE loaded from '/home/ubuntu/Unchessed-UCI-Engine/unchessed-nnue.bin'
readyok
bestmove e2e4
info depth 1 multipv 1 score cp 12 nodes 42 nps 42000 hashfull 0 time 0 pv d2d3
info depth 2 multipv 1 score cp 23 nodes 311 nps 311000 hashfull 0 time 1 pv a2a4 d7d6
info depth 3 multipv 1 score cp 13 nodes 1445 nps 722500 hashfull 0 time 2 pv d2d3 d7d6 a2a3
bestmove f3h4
```

This verifies local NNUE loading and real-position search, **not** strength, MAE, or experimental-net comparison. The full gate was attempted with `bash scripts/build-and-test.sh` and blocked before compilation: `lock file version 4 requires -Znext-lockfile-bump` (exit 101). No lockfile/toolchain edits were made.

## Recommendation and gate

**Recommendation: close this thread and redirect effort.** Do not pursue another piece-count sampling/reweighting variant.

**Defer:** any architecture widening, `Full_Threats` feature-set change, 178M/500M “more of the same” retrain, or cloud spend. The 108M result is still more than 100 Elo behind the shipped default at the optimistic confidence bound, and the project has no evidence that another raw-volume point will close that gap. A capacity experiment should not be started as a blind full retrain; it should follow the matched diagnostic.

**Do not drop:** the question. If a balanced reachable-position set fails and a matched width sweep succeeds, capacity becomes the next justified experiment. If both fail, prioritize a teacher/representation audit rather than buying more records. Any candidate that affects evaluation still requires a real paired-game SPRT; lower validation MAE, high Pearson, and a report-only diagnostic cannot change the shipped default.

## What was actually inspected and not run

I read the master brief, reinforcement reports `00`–`12` where present, the NNUE architecture audit, quiet-filter documentation, v4 scaling/results reports, the v4 trainer, and the relevant NNUE runtime/generation references. I used authoritative Stockfish documentation and its training-dataset guidance, plus the recent dataset study. I did **not** run a new training job, generate labels, compute deduplication/coverage statistics, or run an Elo match; the referenced experiment artifacts are absent here. I **did** run the real UCI/NNUE position check above and the repository build gate, which exposed the Cargo lockfile blocker. Existing reported measurements are supplied project evidence, not newly reproduced here.

## References

[1]: `docs/nnue-label-noise-real-measurement.md` "Real paired NNUE label-noise measurement"
[2]: `docs/nnue-v4-108m-recipe-result.md` "NNUE v4: 108M-record recipe-validated SPRT"
[3]: `docs/nnue-v4-retrain-data-scaling-finding.md` "NNUE v4 retrain: data-scaling root cause"
[4]: `docs/nnue-architecture-audit.md` "Auditing our NNUE against the Stockfish reference"
[5]: `docs/nnue-dataset-quiet-filters.md` "NNUE dataset quiet-position filters"
[6]: [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
[7]: [Stockfish NNUE training datasets](https://github.com/glinscott/nnue-pytorch/wiki/Training-datasets)
[8]: [Tan and Watkinson Medina, *Study of the Proper NNUE Dataset* (arXiv:2412.17948)](https://arxiv.org/abs/2412.17948)
[9]: [Stockfish, “Introducing NNUE Evaluation”](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)

## Status

**No source code or defaults changed. No Tier 2/3 work or compute spend started.** **CLOSED — piece-count resampling/reweighting at this scale; capacity remains open only as an untested hypothesis.**

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/13-nnue-ceiling.md`
