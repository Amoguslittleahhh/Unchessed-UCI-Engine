# 07 — Move and piece prediction objective bottleneck

**Investigation ID:** `07-objective`  
**Tier:** 1 (one item only)  
**Repository:** `/home/ubuntu/Unchessed-UCI-Engine`  
**Branch inspected:** `manus/rustc-bootstrap-trial`  
**Decision:** **Pursue a bounded objective experiment; defer multi-ply as the primary objective and drop any game-facing change at this stage.**

## Executive summary

The current Maia-style objective is single-move supervised classification: given a position and rating/features, predict the recorded legal move with cross-entropy. That is the correct baseline for Maia-style human imitation, and it is also the objective for which this repository has the most direct evidence. The checked-in policy trainer uses a 780-feature board/state input and 4,096 from-to logits, while the newer Unarchitectured pretraining path uses legal-only policy cross-entropy with dual-Elo conditioning. Authoritative Maia work explicitly defines the task as predicting the human move actually taken, rather than optimizing game outcome [1][2]. Chessformer/Maia-3 likewise trains supervised models to predict played moves (or engine playout distributions) and reports strong move-matching without search [3].

A multi-ply continuation objective is scientifically plausible, but it is not a free improvement. It changes the target from a directly observed human action to a joint trajectory likelihood. At minimum it needs teacher-forced legal state transitions, history/state alignment, masking at every ply, a sequence loss/weighting policy, and an evaluation that still measures the first move. It can improve consistency and representation of short-term plans, but it can also average over uncertain future replies and dilute the rating/style signal that Maia is intended to model. The literature reviewed here supports the value of history and auxiliary outcome information, not a claim that multi-ply prediction beats next-move classification for human move matching.

The repository's measured bottleneck is more immediate: **forward-pass cost and cost placement**, not demonstrated inability of single-move classification. The deployed Unarchitectured forward measured 2.43 ms at exit 2/128, 5.21 ms at 4/192, and 12.78 ms at 8/256 on a two-thread sandbox benchmark; an earlier round measured 9.72 ms on a different Core Ultra 9 285H host. The root hint applies only at the first iterative-deepening pass, where ordering is cheapest, while charging the full inference cost against the move budget. Four real SPRT batches were negative; the mechanism analysis attributes the loss to this cost placement, not to poor policy ordering. A multi-ply model normally requires more output computation or repeated decoding, so it is unlikely to address this budget bottleneck without a separately demonstrated one-pass/low-cost design.

The best smallest experiment is therefore an **offline, no-runtime-change A/B diagnostic**: retain the existing single-move legal cross-entropy baseline and compare it with a short teacher-forced continuation loss (for example, two plies) on the already available compact reference data, using a shared encoder and matched compute where possible. Do not train a large cloud model, add MCTS, change UCI behavior, or infer Elo from the result. The experiment should be pursued only to answer whether continuation supervision improves *first-move* held-out metrics at an acceptable inference budget. If it does not, multi-ply should be dropped for this objective; if it does, it remains a retraining hypothesis, not a deployment candidate.

## Question and decision boundary

This report assesses exactly one question: **should the project replace or augment single-move classification with multi-ply continuation prediction for Maia-style policy and Unarchitectured Metal, given the measured forward budget and the first-iteration failure?**

The decision boundary is deliberately narrow. A better objective must improve first-move behavior on a disjoint holdout while preserving legal-action correctness and fitting the engine's real per-move budget. Lower training loss, improved continuation likelihood alone, a changed top-1 sweep, or a fixed-position result is not enough to authorize a game-facing hint. Any runtime/search/default change still requires the repository's paired-game SPRT gate.

## What the repository currently does

### Single-move classification

The legacy policy trainer (`tools/train_policy.py`) reads 104-byte v2 records containing 12 side-to-move-normalized bitboards, the recorded move, rating, castling rights, en-passant file, and flags. It constructs 780 features (768 piece bits, four castling bits, and eight en-passant one-hot values), then trains a per-rating-bucket MLP with a 4,096-class from-to output and `CrossEntropyLoss` (`tools/train_policy.py:4–13, 48–67, 90–123`). This is ordinary behavior cloning: one observed state, one observed action.

The newer pretraining plan is still next-move prediction. It uses legal-only softmax/cross-entropy on a mixed corpus, conditions on `elo_self` and `elo_oppo`, and uses a trusted-only fine-tuning stage. Its v5 record retains legal actions, previous eight plies, policy kind, and dual Elo (`docs/move-prediction-pretrain-plan.md:10–24, 53–67`). The stated rationale is that repeated positions with different Elo/action labels force the conditioning path to become informative.

This objective already has a useful proof-of-mechanics. The committed 13,076-row NumPy probe reached best validation CE **2.8479** from an initial roughly **3.82**, validation top-1 **0.1687** versus a uniform-legal baseline **0.0879**, and Elo conditioning changed top-1 on **118/200** positions (extremes **115/200**). Mean top-1 probability rose from **0.1561** at Elo 600 to **0.2591** at Elo 3200. These are diagnostic results, not strength evidence, but they show that the current next-move objective can learn both action signal and conditioning signal (`benchmarks/unarchitectured-metal/pretrain-probe-2026-08-28.json`, `docs/move-prediction-pretrain-plan.md:78–92`).

### Unarchitectured Metal policy signal and runtime cost

The exported Unarchitectured student has real policy signal. On 600 provenance-disjoint over-the-board positions labelled by Stockfish 17.1 at 400,000 nodes, full exit 8/256 achieved top-1 **0.2550**, top-3 **0.4567**, mean best-move rank **6.933**, and mean first-move regret **118.7 cp**. Replication on 300 positions gave top-1 **0.2433** and regret **117.2 cp**. The full exit beat a static MVV-LVA/promotion/check ordering baseline on top-1 (**0.255 vs 0.157**) and mean regret (**118.7 vs 260.2 cp**); paired McNemar was **p=1.8e-9** on the primary corpus and **p=3.2e-4** on replication (`docs/unarchitectured-metal-calibration.md:18–30, 49–98`). Thus, single-move policy learning is not an empty or obviously mis-specified objective.

The policy's probability magnitudes also carry information: on the same 600-position analysis, mean top-1 confidence was **0.266** when correct and **0.148** when wrong; confidence normalized by the uniform legal baseline was **6.21x** versus **4.26x**. ECE was **0.0048**, with an in-sample fitted temperature of **0.70** (`docs/policy-prior-calibration.md:17–59`). This supports preserving calibrated legal distributions in the experiment rather than reducing the objective to ranking only. It does not establish that a continuation objective would improve calibration or playing strength.

The cost is material and host-dependent. A controlled two-thread benchmark measured retained-int8 latency of **2.428812 ms** at 2/128, **5.213683 ms** at 4/192, and **12.775529 ms** at 8/256; the one-thread full-forward comparison was **15.454061 ms** retained-int8 versus **21.487748 ms** dequantized (`benchmarks/unarchitectured-metal/runtime-forward-2026-08-22.md:9–35`). A separate real-hardware round used **9.72 ms** as the forward-pass measurement. The latter is not interchangeable with the two-thread result; it is a different host and configuration.

### The first-iteration failure

The root hint applies only when `depth == start_depth`. At that first pass, root scores are still mate sentinels and the normal fallback ordering has no useful search scores. Later iterative-deepening passes overwrite ordering with alpha-beta scores and TT information (`docs/unarchitectured-metal-why-the-hint-costs-elo.md:23–27, 63–70`). The policy therefore improves a cheap first pass, while its preprocessing time is charged to the entire move deadline. Using the 9.72 ms host-specific measurement, the documented budget arithmetic estimates a **9.0%** cost share at 5+0.05 with a 5,000 ms clock, **14.7%** at 2,500 ms, and only **0.4–0.7%** at 60+0.6 (`docs/unarchitectured-metal-why-the-hint-costs-elo.md:72–101`).

Four real SPRT batches all trended negative, while offline ordering-risk analysis found the neural order better than move generation on every reported tail metric: top-1 **0.2683** versus **0.0683**, mean first-move regret **146.3 cp** versus **290.5 cp**, and blunder rate **0.2317** versus **0.4800** (`docs/unarchitectured-metal-why-the-hint-costs-elo.md:34–57`). The repository's conclusion is that the failure is not “the policy predicts bad moves”; it is that the benefit is confined to the cheapest pass and the cost is paid on every move. This is the relevant bottleneck for objective choice.

## Single-move versus multi-ply continuation

| Dimension | Single-move legal classification | Multi-ply continuation prediction |
|---|---|---|
| Target | Recorded human action at the current position | Ordered sequence of future actions, usually teacher-forced |
| Maia alignment | Directly matches the published human-move task | Indirect: future replies include another player's choices and more noise |
| Data requirement | One current state plus legal target | State/history plus every subsequent legal transition and terminal/truncation policy |
| Loss | One legal-action CE per sampled position | Sum/weighted CE across plies, with masking at each step; optional outcome loss |
| Main upside | Direct first-move accuracy and simple calibration | May encode short plans, temporal consistency, and useful history representations |
| Main risk | Myopic/locally ambiguous predictions | Objective dilution, exposure bias at inference, more decoding or output work, and leakage of reply/player identity |
| Runtime implication | One forward pass per decision | One forward pass with multiple heads or repeated autoregressive passes; neither is free |
| Existing evidence | 13k-row probe, 600/300-position policy diagnostics, calibrated confidence | No repository continuation run, metric, checkpoint, or runtime benchmark |

A two-ply loss should not be presumed to be “more information for the same cost.” If the model predicts two actions autoregressively, it needs a representation of the post-move state and a second decoding step. If it predicts both actions in parallel from the initial state, the second target is conditioned on information that may not be available at deployment and can encourage shortcuts. If it uses a multi-head shared encoder, it adds output parameters and training complexity while still requiring a first-move head. The only defensible low-cost version is an auxiliary continuation loss whose extra computation is confined to training and whose deployed inference remains one first-move forward pass; even then, benefit must be measured rather than assumed.

For Maia-style imitation, the direct literature favors the single action target. The original Maia paper states that it repurposes the AlphaZero network to predict **human actions**, trains on human games rather than self-play, and makes predictions without tree search; its supplement specifies a policy cross-entropy loss and equally weighted value MSE [1]. Maia-2 similarly describes policy cross-entropy on one-hot recorded moves, with auxiliary information and game-outcome heads balanced in the optimization [2]. Chessformer/Maia-3 describes supervised prediction of the move actually played, or a playout distribution for engine-distillation data, and reports 5M/23M/79M models with human move-matching accuracy of **55.4%/56.6%/57.1%** on its stated benchmark [3]. These sources demonstrate strong next-action formulations; they do not establish that multi-ply human continuation is superior.

AlphaZero is a different target and should not be used as evidence for replacing Maia's objective. Its published algorithm learns a policy/value network through self-play search, not a human-behavior continuation likelihood; the Science paper describes a general self-play system that defeated specialist programs [4]. The repository already records that an AlphaZero-style route would require MCTS/PUCT, legal masking, replay, batched inference, and a new policy/value ABI, and that the 9.72 ms Unarchitectured timing makes hundreds of serial evaluations incompatible with the current fast-move budget (`docs/reinforcement/06-rl-selfplay.md:8–12, 30–50`). That is outside this Tier 1 item and is not a justification for a multi-ply supervised replacement.

## Smallest testable experiment

### Scope

Run an offline, design-isolated A/B experiment using the existing compact pretraining fixture and the same legal-action encoder. Do **not** alter Rust, UCI defaults, the shipped package, root-hint behavior, or the production trainer ABI. Do **not** use cloud compute, CUDA, MCTS, or a new large corpus.

The baseline arm is the existing next-move legal-only CE model. The candidate arm shares the encoder and first-move policy head but adds a **two-ply teacher-forced auxiliary continuation head**. At training time, the first target is the recorded move at position *t* and the second target is the recorded move at *t+1* from the actual post-move state, with legal masking at both steps. Use a declared loss such as `L = CE_t + 0.5 * CE_t+1`; retain the first-move head and inference path unchanged. If the available compact data cannot provide an unambiguous post-move state and next legal target, stop rather than reconstructing it with undocumented assumptions. This is an auxiliary continuation test, not a claim that a fully autoregressive decoder is ready.

Use a deterministic, game-disjoint split. The minimum diagnostic should use the existing 13,076-row reference only to validate the data path and loss, then a held-out subset large enough to report uncertainty; no result should be promoted from a single random row split. Preserve the same seed, optimizer family, number of updates, parameter-count accounting, and baseline compute budget. Report wall time and parameter count for both arms, since “same number of epochs” is not the same compute.

### Required metrics and gates

The primary endpoint is **first-move legal cross-entropy and top-1 accuracy** on a game-disjoint holdout. Secondary endpoints are top-3, MRR, expected calibration error, top-1 confidence separation between correct and incorrect cases, and continuation CE/top-1 at ply two. Report legal-mask violations (must be zero), target-in-legal rate (must be 100% after data filtering), NaN/Inf counts (zero), and reproducibility across two identical CPU runs.

The candidate is interesting only if it improves the first-move metric beyond a predeclared uncertainty interval, does not degrade calibration materially, and has an explicit inference-cost story. A continuation-only gain is insufficient. A candidate that improves first-move CE but requires a second deployed forward pass should be compared against the measured 2.43/5.21/12.78 ms exit costs and rejected unless a real deployment benchmark shows the added cost fits the budget. The experiment should be stopped if post-move state/legality cannot be represented exactly, if any illegal target is silently masked away, or if the candidate's first-move result is indistinguishable from baseline after accounting for compute.

### What success would and would not authorize

Success would justify a larger, provenance-complete retraining study and possibly an auxiliary continuation loss in the model trainer. It would not authorize a new runtime package, larger GAB, root hint, adaptive behavior, or default flip. Any candidate used by the engine must pass the existing numerical parity, legal/only-move/mate safety, deployment latency, and paired-game SPRT requirements. In particular, it must be tested at an explicitly named exit; the existing evidence shows 2/128 is materially weaker than 8/256 (top-1 **0.185 vs 0.255** on the 600-position corpus), so a full-exit finding cannot be transferred to the cheap exit (`docs/unarchitectured-metal-calibration.md:100–110`).

## Verified, assumed, design-only, and blocked

### Verified from this checkout

* Branch is `manus/rustc-bootstrap-trial`; the worktree already contained an unrelated untracked `docs/reinforcement/06-rl-selfplay.md` before this report. No existing file was overwritten.
* The six preceding reinforcement documents were read, along with the policy trainer, pretraining plan/probe, policy-prior calibration, Unarchitectured calibration/runtime reports, and root-hint failure analysis.
* `python3 -m pytest -q tools/test_policy_prior_calibration.py tools/test_pretrain_move.py` completed **28 passed, 13 subtests passed** in **0.66 s**.
* The repository's stored measurements are: pretraining probe CE **2.8479**, top-1 **0.1687 vs 0.0879** uniform, Elo sweep flips **118/200**; Unarchitectured policy top-1 **0.2550** on 600 and **0.2433** on 300 replication; runtime exits **2.428812/5.213683/12.775529 ms** in the cited two-thread benchmark; historical 9.72 ms forward timing on a different host; and negative real SPRT batches for the root hint.
* Maia 2020, Maia-2, AlphaZero, and Chessformer sources were fetched and read at the URLs listed below. The claims attributed to them are limited to their stated objectives and reported setup/results.

### Design-only or not run

No continuation model, decoder, auxiliary loss, checkpoint, runtime integration, forward benchmark for a new model, self-play, cloud training, cutechess match, or SPRT was run for this report. No PyTorch dependency was installed. The proposed two-ply experiment is a design, not a result.

### Assumptions and blockers

The proposal assumes the compact reference data can reconstruct the exact post-move board, legal action set, and next recorded move without losing castling, en-passant, promotion, side-to-move, or game identity. This must be verified before coding. The existing v2 policy record includes special-rule metadata, but the 13k pretraining probe's exact retained fields and continuation alignment must be audited rather than inferred from prose. The production Unarchitectured training assets/checkpoints and Torch/CUDA environment are unavailable in this sandbox, and no continuation corpus/metrics exist. These blockers prevent a production-scale comparison, not the bounded CPU data-path test.

The 9.72 ms number is host-specific and must not be treated as a universal budget. The 2.43/5.21/12.78 ms numbers are also standalone forward calls, not integrated search NPS or achieved depth. The 600/300-position teacher calibration is Stockfish-best-move agreement, not human move matching or game outcome. The 118/200 conditioning result is a small NumPy probe, not evidence of playing strength.

## Recommendation

**Pursue** the bounded auxiliary two-ply diagnostic only. Keep **single-move legal classification as the primary Maia-style objective** and as the deployment contract. **Defer** any full multi-ply/autoregressive model until the diagnostic proves a first-move benefit at matched compute and exact state/legality. **Drop** the idea that changing the objective alone fixes the first-iteration root-hint failure: the measured failure is primarily that an approximately 9.72 ms (host-specific) preprocessing cost is charged to every move while the policy benefit lands on the cheapest first search pass. A continuation objective will generally add target complexity and may add inference cost; it does not move the benefit to deeper passes or make the forward pass free.

If the two-ply auxiliary arm fails the first-move gate, stop this line and retain the current next-move objective. If it passes, the next work remains retrain-gated and game-facing use remains SPRT-gated. This is the smallest honest experiment that can distinguish an objective bottleneck from the already measured cost-placement bottleneck without spending cloud money or weakening the repository's evidence standards.

## Sources

[1] McIlroy-Young et al., **“Aligning Superhuman AI with Human Behavior: Chess as a Model System,” KDD 2020**, original paper and supplement: <https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf>. The paper describes human-action prediction without tree search; the supplement specifies policy cross-entropy and value MSE.

[2] McIlroy-Young et al., **“Maia-2: A Unified Model for Human-AI Alignment in Chess,” NeurIPS 2024**, HTML paper: <https://arxiv.org/html/2409.20553v1>. Section 3 describes one-hot move cross-entropy, auxiliary multi-hot information loss, and game-outcome value regression.

[3] Tang et al., **“Chessformer: A Unified Architecture for Chess Modeling,” ICLR 2026**, HTML paper: <https://arxiv.org/html/2605.19091v1>. Sections 3–4 describe supervised prediction of played moves or engine playout distributions and report Maia-3 move-matching results.

[4] Silver et al., **“A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play,” Science 2018**, DOI/source page: <https://www.science.org/doi/10.1126/science.aar6404>. This is cited only for the distinction between self-play policy/value learning and human-action imitation.

## Final status

**Recommendation:** pursue bounded auxiliary continuation diagnostic; defer full multi-ply replacement; drop objective-only explanation of the first-iteration failure.  
**Strength/default conclusion:** none.  
**Production change:** none.
