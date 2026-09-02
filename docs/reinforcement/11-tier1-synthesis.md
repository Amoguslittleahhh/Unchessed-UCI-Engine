# 11 — Tier 1 reinforcement synthesis

**Scope.** This document consolidates the five completed Tier 1 reports—self-play viability, move-objective choice, NNUE label noise, search-parameter tuning, and persona real-play measurement—against the standing evidence rules in [the baseline synthesis][1] and its supporting investigations `[01]`–`[05]`. It records the reports' evidence; it does not rerun their tests, create an engine/model candidate, or make a strength claim.

> **Explicit boundary: no Tier 2/3 work or compute spend started.** Across these five investigations, no Tier 2 or Tier 3 implementation, candidate training, self-play campaign, cloud job/rental, paid compute spend, match campaign, or default change was started. The lightweight local checks listed below were Tier 1 verification only. No commit or push was made.

## Consolidated decision

The shared conclusion is **evidence before scale**. The repository has several bounded, offline or default-off diagnostic designs worth retaining, but none authorises a game-facing behavior, a model replacement, a changed UCI default, an Elo claim, or cloud expenditure. The closest near-term research action is a carefully bounded **two-ply auxiliary objective diagnostic** that leaves deployment inference unchanged; it tests a specific data/objective question rather than a strength hypothesis. The other four streams are deferred or conditional until their more basic requirements are met.

Every proposed candidate that could change search, move selection, evaluator behavior, persona behavior, or a default remains subject to the standing requirement for a **fresh, real paired-game SPRT** with immutable binary/model/options/opening/hardware provenance. Unit tests, offline metrics, simulations, a label MAE/Pearson comparison, a fixed-position search screen, and an interrupted match do not cross that boundary.[1]

## Comparison of the five Tier 1 streams

| Stream and report | Evidence-supported finding | Work actually run or inspected | Design-only / deliberately not run | Current decision and next gate |
|---|---|---|---|---|
| **AlphaZero-style RL self-play** [2] | The checked-in `HalfKAv2_hm` NNUE is a scalar alpha-beta evaluator, not a policy/value model. No verified MCTS/PUCT, replay, self-play, policy head, or NNUE batch-throughput measurement exists. | A king-bucket test passed **6/6**; a release-adapter UCI smoke advertised `PolicyFile` but reported no policy net and heuristic priors; two NNUE model files were measured. | No MCTS/PUCT, NNUE throughput benchmark, policy head, self-play game, replay generation, gradient update, full Cargo test, CUDA install, cloud run, or SPRT. Phase A/B limits are proposals only. | **Defer.** Permit only an offline, deterministic interface/throughput gate with no default integration or cloud spend. Drop the approach at current scale if legality, determinism, bounded-throughput, or learned-policy-versus-uniform-baseline gates fail. |
| **Move/piece prediction objective** [3] | Existing single-move legal cross-entropy has measurable policy/conditioning signal. The documented root-hint failure is principally forward-cost placement, not evidence that next-move classification cannot learn. | Focused policy/pretraining tests passed **28 tests plus 13 subtests in 0.66 s**; repository metrics and published objective definitions were reviewed. | No continuation model, auxiliary loss, checkpoint, new forward benchmark, runtime integration, self-play, cloud training, cutechess match, or SPRT. | **Pursue only** an offline, two-ply teacher-forced auxiliary-loss A/B diagnostic with unchanged first-move deployment inference. Keep single-move classification primary; defer full multi-ply/autoregressive replacement. |
| **NNUE label-noise comparison** [4] | A real score-wise MAE/Pearson comparison cannot begin because there is no authoritative NNUE source shard paired with an alternative ordered score sidecar. A model binary is not a score sidecar. | Repository and `/home/ubuntu` inventories, relabel CLI/interface, and focused synthetic/tooling tests were inspected/run; no valid real input pair was found. | No real `compare` run, labels synthesized, search-label generation, training/retraining, match, or SPRT. | **Defer.** First obtain a source shard, matching independent sidecar, full identity/order/state provenance, and teacher configuration. Then run the cheap real comparison; do not infer labels or use shipped NNUE weights as a substitute. |
| **Search bandit/RL tuning** [5] | Prior SPSA was underpowered/poorly controlled rather than a proof against tuning. RL is a poor first method for sparse, delayed, discontinuous game-outcome reward. | Search-option consistency check found **24 options, 0 drifts**; focused tests passed **12**. Prior scripts/logs and tuning literature were examined. | No `SearchStats`, node-limit patch, four-arm fixed-position screen, candidate build, game screen, optimizer comparison, match, or SPRT. | **Defer broad tuning.** First establish exact node limits, default-off telemetry, deterministic safety fixtures, and an incumbent-centered narrow screen. If stable signal exists, consider finite-arm allocation or CLOP/Bayesian/local search over at most 2–3 safe coordinates; keep RL deferred. |
| **PersonaSmooth / EngineDetectV2 real-play protocol** [6] | Opt-in telemetry/parser mechanics exist, but no labelled, joinable real-play corpus establishes flip rate or detector performance. The prior interrupted match measured game outcomes, not the protocol's behavioral endpoints. | Telemetry/parser and detector tests passed **13/13**; parser/manifest option-integrity protections were reviewed. | No compilation, real-play telemetry capture, labelled-opponent study, anonymous/metadata experiment, paired-opening analysis, clustered interval analysis, cutechess match, or fresh SPRT. | **Pursue conditionally** as a preregistered behavioral study, while retaining both defaults as `false`. A behavioral pass must be followed by a new, completed paired Elo SPRT before any default flip. |

## Actually-run evidence versus proposed work

The following distinction prevents an inspection result, synthetic fixture, historical artifact, or literature number from being presented as a newly measured engine result.

| Category | Reported actually-run Tier 1 evidence | What it establishes | What it does **not** establish |
|---|---|---|---|
| **RL/self-play mechanics** | `python3 tools/test_king_buckets.py` passed **6/6**. The existing release adapter completed a UCI smoke and fell back to heuristic priors when no policy net was present. [2] | Feature compatibility test coverage and the absence of a loaded neural policy in that smoke. | MCTS correctness, self-play viability, evaluator throughput, learning, policy quality, or Elo. |
| **Objective/tooling mechanics** | `python3 -m pytest -q tools/test_policy_prior_calibration.py tools/test_pretrain_move.py` passed **28 tests and 13 subtests** in **0.66 s**. [3] | Existing calibration/pretraining tooling behavior. | A continuation objective result, inference latency for a new model, or playing strength. |
| **Label-noise preflight** | Asset inventories and relabel-tool/focused synthetic-tooling checks were performed; usable real source-shard/alternative-sidecar pairs found: **0**. [4] | The real comparison is blocked at input preflight. | Any real label discrepancy, MAE, Pearson correlation, or retraining rationale. |
| **Search parameter consistency** | `python3 tools/check_search_param_consistency.py --repo .` reported **24 options checked, 0 drift(s)**; `pytest` passed **12** focused tests. [5] | UCI/default/clamp consistency for existing search parameters. | That a parameter is tuned, that SPSA worked or failed scientifically, or that any candidate improves Elo. |
| **Persona telemetry/detector mechanics** | `python3 -m unittest tools.test_analyse_adapter_telemetry tools.test_elo_detector -v` passed **13/13**. [6] | Existing parser and detector fixture behavior, including manifest/duplicate/option-integrity cases. | Real-world flip reduction, sensitivity/specificity, clock behavior, or Elo neutrality. |

The following were **designs only**, not experiments whose outcomes can be reported: the self-play Phase A/B pipeline and its throughput threshold; the shared-encoder two-ply auxiliary-loss A/B; the real shard/sidecar comparison pending assets; the four-arm RFP/local-optimizer screen; and the persona 2×2 behavioral matrix with paired openings, pre-registered coverage, and clustered confidence intervals.[2][3][4][5][6]

## Verified numbers and their limits

| Area | Verified number recorded in the reports | Scope and interpretation |
|---|---:|---|
| NNUE evaluator | **5,767,937 parameters**; each inspected model file **23,071,768 bytes** | Size/asset facts for the existing scalar NNUE; they are not a policy/value-model or throughput measurement. [2] |
| AlphaZero reference context | **800** simulations per chess MCTS; **44 million** games; **700k** minibatches; **9 hours** on the reported four-TPU machine | Literature reference, not local performance or a proposed budget. It supports deferring direct replication. [2] |
| Single-move probe | CE **2.8479**; top-1 **0.1687** versus **0.0879** uniform; conditioning flips **118/200** | Stored repository diagnostic on a 13,076-row probe; evidence of learnable next-move/conditioning signal, not strength or a continuation result. [3] |
| Unarchitectured policy diagnostics | Top-1 **0.2550** on 600 positions and **0.2433** on a 300-position replication | Stored teacher-best-move calibration, not human move matching, a new run, or game outcome. [3] |
| Forward timing | Two-thread exits: **2.428812 / 5.213683 / 12.775529 ms**; separate host measurement: **9.72 ms** | Host/configuration-specific standalone forward timings. They are not integrated search throughput and the 9.72 ms Unarchitectured value must not be substituted for NNUE speed. [2][3] |
| Label-noise comparison | Real source records **0**; real alternative-sidecar records **0**; labels synthesized **0** | There is **no real MAE or Pearson number**. This is an unavailable-input result, not a zero-error finding. [4] |
| Prior SPSA setup | **12** games per match at `3+0.03`; **40/200** iteration caps; visible log iterations through **93** | Historical script/log review. A nominal two-sided 200-iteration run would be 4,800 games before confirmation, but the report found no completed campaign. [5] |
| Persona historical match | **2,125–2,095–1,317** at **5,537** games; score **0.503**; approximately **+2.1 Elo** | Historical interrupted combined-options result. Neither SPRT bound crossed; it is not the proposed behavioral study and not a passed strength gate. [6] |
| Persona simulation | **57.3%** simulated reduction in flips; **0/200** synthetic engine flags in each stated declared-human band | Simulation/synthetic-test evidence only, not real-play behavior or detector accuracy. [6] |

## Shared blockers

| Blocker | Affected streams | Required unblocking evidence |
|---|---|---|
| **No real NNUE source data / bound labels / complete state provenance** | Label noise; supervised NNUE follow-on; indirectly the RL comparison baseline | Immutable source-shard manifests/files, a matching alternative sidecar or auditable regeneration, ordered identity binding, full replayable state or exact reconstruction, and declared teacher/configuration. Equal counts or model-file sizes are inadequate. [1][4] |
| **No policy/value self-play stack or NNUE throughput number** | AlphaZero-style self-play | A bounded offline prototype with exact legal masking, deterministic replay, terminal handling, a measured NNUE batch benchmark, and a predeclared under-one-hour host gate before contemplating a policy head or scale. [2] |
| **Continuation-state/alignment uncertainty and unavailable production training assets** | Move-objective experiment | An audit proving exact post-move state, legal actions, next targets, special-rule fields, and game-disjoint identities in the compact diagnostic data. Production comparisons also require the unavailable checkpoints/shards and suitable training environment. [3] |
| **Verification and match floor is incomplete in this sandbox** | Search tuning; persona study; all strength candidates | A current compatible Rust toolchain where required, deterministic fixtures, pinned binaries/evaluator/options/book/hardware/time control, raw PGN/log retention, and a real cutechess/fastchess-capable host. The reports record `/usr/bin/cargo` **1.75.0** and no `cutechess-cli`/`fastchess` on `PATH` for the bandit review. [1][5][6] |
| **No labelled real-play roster, validated clock harness, or generated study manifest** | Persona behavioral measurement | Pre-registered labels and exposure conditions, immutable runner/manifest/telemetry/PGN capture, sufficient clock coverage, paired openings, and cluster-aware analysis. The engine's own detector output cannot supply ground truth. [6] |
| **Missing Unarchitectured oracle/student training assets and Torch** | Context for deployment-oriented objective work; remains blocked from `[00]`–`[05]` | Do not substitute exported student or NNUE artifacts. Oracle and broader Unarchitectured retraining remain governed by their distinct checkpoint, parity, and provenance gates. [1] |

## Recommendations and order of operations

1. **Do not change any default or launch a strength campaign.** Preserve `UnarchitecturedHint=false`, `PersonaSmooth=false`, `EngineDetectV2=false`, and the current search defaults. No report supplies the required real paired-game evidence for a promotion.[1][3][5][6]

2. **Restore the verification floor and implement only default-preserving measurement/correctness work already specified in the baseline synthesis.** Exact `go nodes N` enforcement and default-off `SearchStats` should precede any search-tuning claim; telemetry/parser/provenance changes must prove disabled-path equivalence. This is not permission to tune or to claim Elo.[1][5]

3. **For NNUE labels, stop at honest preflight until real paired assets arrive.** Do not manufacture labels, reinterpret a `.bin` model as a score stream, or start retraining. When assets arrive, first complete the low-cost, real MAE/Pearson comparison with cryptographic binding and a declared teacher; a resulting statistic still does not establish playing strength.[1][4]

4. **Run only the bounded two-ply auxiliary objective diagnostic if exact state and target alignment can be proved.** Keep the first-move head and deployment inference unchanged, use a deterministic game-disjoint split and matched budget, require zero legal-mask violations and 100% target-in-legal rate after filtering, and judge the candidate first on held-out *first-move* metrics. If this gate fails, retain single-move classification and stop the line. A continuation-only gain is insufficient.[3]

5. **Keep AlphaZero-style work at an offline viability gate.** Before policy-head work, require exact legal actions, repeatable trajectories/replay, terminal correctness, and a real NNUE benchmark at batch sizes appropriate to the proposed tree search. Do not use the separate 9.72 ms Unarchitectured timing as NNUE evidence and do not buy compute to compensate for an unmeasured pipeline.[2]

6. **Defer broad bandit/RL search tuning.** Once telemetry and exact node behavior exist, use an incumbent-protected, one-mechanism fixed-position screen and only then a small paired reject-only game screen. Continue only for stable local signal, with at most 2–3 normalized safe coordinates; use finite arms or a noise-aware local optimizer rather than unconstrained 13-parameter RL.[5]

7. **Treat persona work as a behavioral study, not a shortcut to Elo.** If external facilities and labels become available, execute the pre-registered 2×2 design with separate anonymous and metadata-assisted strata, paired openings, coverage accounting, and crossed-cluster inference. Only after its behavioral gates pass may the owner start a separate, completed paired Elo SPRT for exact candidate defaults.[6]

## Promotion boundary

A future successful Tier 1 diagnostic would establish only that the next controlled research step is justified. It would **not** start Tier 2/3 work automatically. Before any game-reachable integration or default change, the project must separately satisfy candidate provenance, legal/safety and numerical-parity checks, deployment latency constraints, reject-only smoke screens where appropriate, and a fresh paired-game SPRT. The historical negative/incomplete evidence remains correctly scoped: four root-hint SPRT batches were negative; the 5,537-game persona result did not cross a bound; the prior 108M NNUE recipe result was not a reason to spend on more of the same labels; and prior SPSA logs are not a completed optimizer result.[1][3][4][5][6]

**Final Tier 1 disposition:** retain one low-cost objective diagnostic and conditional persona protocol; defer self-play, real label-noise inference, and broad search tuning; preserve all defaults; start **no Tier 2/3 work or compute spend** from this synthesis.

## References

[1]: ./00-synthesis.md "Reinforcement investigations — synthesis, implementation order, and gates"
[2]: ./06-rl-selfplay.md "Reinforcement-learning self-play viability"
[3]: ./07-objective.md "Move and piece prediction objective bottleneck"
[4]: ./08-label-noise.md "NNUE label-noise cheap diagnostic"
[5]: ./09-search-bandit.md "Search parameter RL or bandit tuning"
[6]: ./10-persona-protocol.md "Persona real-play measurement protocol for PersonaSmooth and EngineDetectV2"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/11-tier1-synthesis.md`
