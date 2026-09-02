# 09 — Search-parameter bandit / RL tuning

**Investigation ID:** `09-search-bandit`  
**Repository / branch inspected:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/rustc-bootstrap-trial`  
**Scope:** principled bandit, reinforcement-learning, and black-box optimizer methods for UCI search-parameter tuning, specifically as an alternative to the failed blind SPSA attempt. This is exactly one Tier 1 item. No Tier 2/3 work, dependency installation, cloud run, commit, or push was performed.

## Decision

> **Recommendation: DEFER a bandit/RL campaign; do not drop the research question.** First build the smallest measurement floor and run a narrow, local-candidate diagnostic. Do not launch a 13-parameter optimizer, change a default, or call any result an Elo improvement until a candidate has passed the repository’s paired-game SPRT gate.

A bandit is a reasonable **outer-loop allocator** for a small, finite set of already-safe candidates. It is not a substitute for search-parameter calibration: an arm must be a complete engine configuration, and every arm comparison consumes real games. RL is a poor first tool here. The reward is sparse and delayed (game outcomes), the search tree changes discontinuously when pruning thresholds cross activation boundaries, and the opponent/book/hardware/time-control distribution is part of the environment. The most defensible optimizer, if the diagnostic supports continuing, is a **bounded, warm-started, noise-aware local black-box method**—SPSA with corrected scale/validation, CLOP, or Bayesian/dueling optimization over at most two or three parameters—not policy-gradient RL.

## What is in the engine

The existing `SearchParams` contains 13 numeric search controls plus the default-off `ProbcutSeeFilter`: RFP margin; null-move base/divisor; LMR depth and move thresholds; aspiration delta/depth; ProbCut margin/reduction/depth; and futility margin/depth. They are advertised and clamped in UCI. The code already implements iterative deepening, PVS, TT, null-move pruning, reverse/per-move futility, ProbCut, extensions, LMR, killers/history, SEE ordering, and aspiration windows. Therefore the problem is not absence of a search algorithm; it is estimating whether small changes to interacting, tree-changing thresholds improve playing strength.

The audit explicitly says the 13 search parameters are **hand-picked defaults** and that the planned SPSA tuning run was never executed. The current hard-coded RFP depth ceiling (`depth <= 6`) is adjacent to the tunable RFP margin. The audit’s proposed real-hardware work order starts from the current defaults, uses advertised bounds, tests one parameter at a time, and requires a real SPRT. Its standing convention is `Threads=1`, `Hash=256`, `tc=5+0.05`, `elo0=0`, `elo1=5`, `alpha=beta=0.05`, with a 30,000-game cap for inconclusive candidates.

The reinforcement synthesis adds two important prerequisites: exact `go nodes N` handling and default-off `SearchStats` before tuning. Search telemetry should count TT probes/hits/cutoffs, RFP/NMP/ProbCut/futility actions, LMR reductions/re-searches, quiescence skips, and aspiration failures. Without these counters, a candidate can appear promising while merely changing node allocation or accidentally disabling a tactical safeguard.

## The failed blind SPSA attempt

The checked-in scripts are not evidence of a successful search-parameter calibration. `scripts/exhibition/spsa_conthist_temp.py` hard-codes a `3+0.03` time control, 12 games per iteration, 40 iterations, and the standard SPSA exponents `alpha=0.602`, `gamma=0.101`. The older `wsl-workspace/scripts/spsa_passedpawn.py` uses 12 games per iteration, `3+0.03`, and 200 iterations. Each SPSA iteration evaluates a plus and minus configuration; the nominal budget is therefore **24 games/iteration**, or **4,800 games for 200 iterations** before any confirmation matches. The available logs show only 94 lines in `spsa_pp2_run.log` (startup plus iterations through 93), not a completed 200-iteration campaign.

The visible run behavior is diagnostic of an underpowered/noisy setup, not proof that SPSA cannot work. Early iterations repeatedly report scores such as `0.333`, `0.375`, `0.500`, and `0.625` from only **12 games per match**, while the parameter vector remains effectively at `[50.0, 50.0]`; by iterations 73–93 it moves only around `[49.9, 50.1]`. The script points to absolute paths outside this checkout for cutechess, book, engine, and output. Those paths are not available in this sandbox. The run therefore cannot be reproduced here, and its logs do not establish a completed match, an SPRT decision, or a stable optimizer trajectory.

The likely failure modes are concrete:

1. **Too few games per gradient sample.** Twelve games per side of a two-sided perturbation produce a very high-variance win-rate difference. A 0.500 score at this sample size is not a measured zero gradient.
2. **Blind starting point and perturbation scale.** The scripts start at `[50, 50]` with large early perturbations (the log begins around `c_k=25`). SPSA assumes a locally informative finite difference; a broad perturbation can cross qualitatively different pruning regimes, while a small one is swallowed by match noise.
3. **No baseline control or validation holdout.** There is no documented identical-build control, repeated evaluation of the incumbent, or independent confirmation of a proposed update. A noisy plus/minus winner can become the next baseline and compound drift.
4. **Coupled integer thresholds.** RFP/NMP/LMR/ProbCut/futility parameters interact. Updating a simultaneous vector without activation telemetry cannot distinguish useful search from altered tactical risk.
5. **Objective mismatch and cost.** Fast `3+0.03` games reduce wall time but amplify opening, draw, and time-management variance. The project’s actual promotion convention is a paired game SPRT, not an optimizer’s interim score.

Thus the honest conclusion is **“blind SPSA protocol failed to produce evidence,” not “SPSA is disproven.”**

## Literature assessment

### SPSA and its limits

Spall’s implementation paper describes SPSA as a two-measurement simultaneous perturbation gradient estimator, useful when the objective is noisy and the number of parameters is large. That efficiency comes from accepting a noisy gradient and depends on choosing perturbation scales and gain sequences appropriate to the local signal: [Spall, *Implementation of the Simultaneous Perturbation Algorithm*](https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF). It does not guarantee recovery from an arbitrary starting point, discontinuous objective, or insufficient replication.

Stockfish’s authoritative Fishtest documentation confirms that chess tuning uses normalized parameter scales, two-sided win/loss signals, parallel out-of-order updates, and SPSA. Crucially, it also describes pentanomial match modeling and GSPRT for efficient sequential testing: [Stockfish Fishtest Mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html). This supports borrowing the **discipline and normalization**, not assuming that a small local script will inherit Stockfish’s data volume and infrastructure.

### CLOP and Bayesian optimization

CLOP is explicitly designed for noisy black-box parameter tuning. Its author describes local regression with confidence-based rejection of samples confidently inferior to the mean: [Coulom, *CLOP: Confident Local Optimization for Noisy Black-Box Parameter Tuning*](http://remi.coulom.free.fr/CLOP/). This is attractive when the candidate space is low-dimensional and relatively smooth, but search pruning has integer thresholds, invalid/safety regions, and potentially non-smooth Elo response. CLOP should therefore be considered only after restricting the space and defining safe bounds.

The chess-specific Bayesian comparison by Ivec and Vojnović reports a method intended for objectives observable only by comparing parameter sets, and compares it with SPSA. In their Stockfish experiment, tuning 29 deliberately weakened parameters for 100,000 iterations at `5+0.05`, SPSA ended about **64 Elo below** the original while their Bayesian variant ended about **50 Elo below**; after restoring one parameter stuck in a local optimum, the figures were about **−22** and **−15 Elo**, respectively. These are useful evidence that optimizer choice and local optima matter, but they are not a transferable guarantee for this engine: [Ivec & Vojnović, *Bayesian statistics approach to chess engines optimization*](https://arxiv.org/abs/2205.15602) (paper PDF: [arXiv PDF](https://arxiv.org/pdf/2205.15602)).

The broader chess-programming survey lists SPSA, CLOP, evolutionary methods, and Bayesian optimization as black-box approaches, while noting that black-box tuning handles search/evaluation interaction at substantial time cost: [ChessProgramming.org, Automated Tuning](https://chessprogramming.org/Automated_Tuning). The survey’s RL section concerns TD/TD-leaf evaluation learning, not a demonstrated replacement for noisy game-based pruning-constant calibration. That distinction matters: TD-style RL can learn a value function from many trajectories, whereas this task has a small discrete control vector and an expensive binary outcome objective.

### What the literature does and does not establish

| Claim | Status | Interpretation for this repository |
|---|---|---|
| SPSA can estimate a noisy multivariate black-box gradient with two evaluations per iteration. | **Verified from authoritative method sources.** | It is a candidate method, not a license to use 12-game gradients. |
| Normalized coordinates and per-axis perturbation scales are important in chess tuning. | **Verified in Stockfish documentation.** | Use ranges tied to observed activation/safety behavior, not raw heterogeneous UCI units. |
| CLOP/Bayesian methods can improve over SPSA on some noisy chess-engine tasks. | **Reported by authoritative/project sources, but not independently reproduced here.** | Worth a narrow A/B diagnostic; no expectation of automatic superiority. |
| RL will improve these UCI search parameters. | **Unsupported for this engine/task.** | Defer RL; there is no state/reward dataset or controlled online environment. |
| Any optimizer result is an Elo result. | **False under project policy.** | Only a completed paired-game SPRT can support strength/default conclusions. |

## Cost, starting-point, and safe-candidate constraints

A parameter evaluation is not a cheap scalar function call. It is a paired match with opening-book and color effects, draw outcomes, engine startup, and time-control variance. The Fishtest documentation notes that pentanomial analysis can save testing resources, but it does not make games free. The repository’s own convention uses `elo0=0`, `elo1=5`, and 5% risks; a true 5-Elo decision can require thousands of games, and an inconclusive candidate can consume the documented 30,000-game cap. A 13-dimensional optimizer that proposes dozens of simultaneous candidates can spend more games on selection noise than on confirmatory evidence.

Starting-point sensitivity is especially severe here. The current defaults are the only deployed, internally consistent point; they are not known to be globally optimal, but they are the safest incumbent. Start at that point, or at a candidate that differs in exactly one well-understood coordinate. Do not start from zero, imported Stockfish constants, or an optimizer-generated vector outside the advertised range. The audit records a failed king-safety experience as the reason to start from validated real numbers.

Safe candidate design should be:

* **One mechanism at a time:** first RFP margin or RFP depth ceiling, then one NMP/LMR/aspiration/ProbCut/futility coordinate; do not bundle unrelated pruning rules.
* **Bounded and integer-valid:** use existing UCI clamps; reject values that disable a guard or enter mate-range behavior unless explicitly designed and tested.
* **Default-preserving:** adding telemetry or exposing `RFPMaxDepth` with default 6 must produce identical fixed-depth score/PV/node totals against the legacy path.
* **Tactically screened:** include mate, zugzwang/null-sensitive, narrow-window, quiet-futility, and LMR boundary positions. A fixed suite is reject-only, not strength evidence.
* **Incumbent-protected:** never replace the incumbent solely because one noisy batch wins. Require a repeated/confirmatory comparison and retain all raw games, hashes, options, book, and logs.
* **Evaluation-stable:** hold evaluator, thread count, hash, book, binary, and time control fixed. Search tuning against HCE and then claiming NNUE strength is invalid; the default shipped evaluator is NNUE.

## Smallest useful diagnostic

The smallest diagnostic should answer whether there is a measurable, repeatable local signal before any bandit/RL campaign:

1. **Instrumentation only:** add default-off `SearchStats` (or use an equivalent fixed-position harness) and parameterize `RFPMaxDepth` with legacy default **6**. This is design-only in this report; no source change was made.
2. **Candidate set:** incumbent plus four safe, one-coordinate candidates around the incumbent: `RFPMargin` at current value ± one modest step, and `RFPMaxDepth` at 5 and 7. If RFP activation is too rare in the chosen corpus, replace the depth pair with one nearby `LMRMinDepth` pair, but do not add dimensions.
3. **Fixed-position screen:** 50–100 positions spanning start, opening tabiya, hanging material, forced mate, zugzwang/null-sensitive, and quiet late-move cases. Run fresh-TT fixed-depth searches under HCE and shipped NNUE. Record score/PV/node totals and the feature counters. This costs CPU minutes, not games, and can reject obvious safety/tree explosions; it cannot rank Elo.
4. **Incumbent repeatability probe:** on the same deterministic harness, repeat incumbent and each candidate twice with identical settings. Require stable counters and no illegal/mate-regression behavior. If the candidate’s purported effect is not larger than run-to-run variation, stop.
5. **Only if the screen is clean:** run a small paired-opening reject-only match, such as 200 games per candidate versus incumbent at the project’s fixed `Threads=1`, `Hash=256`, `tc=5+0.05` configuration. This is a screening experiment, not an SPRT and not evidence of a positive Elo result. Use the result only to select at most one candidate for the real SPRT.
6. **Optimizer comparison, if still justified:** compare warm-started coordinate/local selection against SPSA on the same two-coordinate candidate set, with identical game budget and a held-out confirmation batch. A bandit may allocate extra games to the currently plausible finite arms; it must not adaptively fish across many arms without recording the multiple-testing procedure.

A pass means only that a narrow candidate has a reproducible local signal and is safe to test. A fail means defer optimizer research and retain defaults. The smallest diagnostic must not install Torch, create self-play infrastructure, or alter live defaults.

## Verified, assumed, blocked

### Verified in this review

* The repository is on `manus/rustc-bootstrap-trial`; the worktree already had unrelated untracked reports (`06-rl-selfplay.md`, `07-objective.md`, and `10-persona-protocol.md`) before this report. They were not modified.
* `SearchParams` has the 13 numeric controls and default-off `ProbcutSeeFilter` described above; the shipped code has the search machinery and UCI plumbing.
* `python3 tools/check_search_param_consistency.py --repo .` completed with **24 options checked, 0 drift(s)**.
* `python3 -m pytest -q tools/test_search_param_consistency.py` completed **12 passed**.
* The prior SPSA scripts specify 12 games per match, `3+0.03`, and 40/200 maximum iterations; `spsa_pp2_run.log` contains 94 lines and visible iterations through 93, with the parameter vector nearly unchanged around `[50, 50]`.
* The documented repository calibration protocol uses real paired games and SPRT; no completed search-parameter SPRT was found or run here.
* `/usr/bin/cargo` is version **1.75.0** in this sandbox. `cutechess-cli` and `fastchess` are not on `PATH`. No match, training, cloud job, or expensive dependency installation was attempted.

### Design-only in this report

The four-arm RFP diagnostic, telemetry schema/use, 200-game screen, and optimizer comparison are proposed designs. No candidate engine was built, no fixed-position counter sweep was run, and no game screen was run. “Bandit as finite-arm allocator” and “warm-started two-coordinate optimizer” are recommendations, not results.

### Assumptions and blockers

The report assumes that a current Rust toolchain and the reviewer’s cutechess/book environment can be provisioned later, as the synthesis and audit state. It does not assume that the earlier SPSA logs are statistically representative or that the reported external Stockfish Bayesian experiment transfers to Unchessed. The immediate blockers for a real campaign are the unavailable cutechess/fastchess and book on this host, the old Cargo lockfile incompatibility, and the absence of a completed baseline-control/paired-game budget. These blockers prevent an honest Elo or optimizer comparison here.

## Final recommendation

**Defer** a principled bandit/RL tuner until default-off telemetry, exact node-budget behavior, deterministic pruning-boundary fixtures, a 1,000-game identical-build harness sanity check, and a narrow incumbent-centered screen exist. Then, if the screen shows a stable signal, pursue **finite-arm bandit allocation or CLOP/Bayesian local search** over two or three safe coordinates, with normalized perturbations and a held-out confirmation. Keep RL out of the first campaign. Drop only the idea of an unconstrained 13-parameter RL/bandit run: its cost, multiple-testing exposure, starting-point sensitivity, and sparse noisy reward make it an unsound next step. Any accepted default still requires a fresh paired-game SPRT with immutable binary/options/book/hardware provenance.

## References

1. [Spall, “Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization,” Johns Hopkins APL](https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF).
2. [Stockfish, “Statistical Methods and Algorithms in Fishtest”](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html).
3. [Rémi Coulom, “CLOP: Confident Local Optimization for Noisy Black-Box Parameter Tuning”](http://remi.coulom.free.fr/CLOP/).
4. [Ivec and Vojnović, “Bayesian statistics approach to chess engines optimization,” arXiv:2205.15602](https://arxiv.org/abs/2205.15602).
5. [ChessProgramming.org, “Automated Tuning”](https://chessprogramming.org/Automated_Tuning).
6. Repository-local evidence: [`docs/reinforcement/00-synthesis.md`](00-synthesis.md), [`docs/reinforcement/01-search.md`](01-search.md), [`docs/parameter-calibration-audit.md`](../parameter-calibration-audit.md), [`unchessed-core/src/search.rs`](../../unchessed-core/src/search.rs), and the prior SPSA scripts/logs named above.

**Report status:** complete; research/design only. No strength claim is made.
