# Persona real-play measurement protocol: PersonaSmooth and EngineDetectV2

**Tier 1 item:** Persona real-play measurement protocol  
**Status:** Design-only; no real-play telemetry study was executed.  
**Repository/line:** `/home/ubuntu/Unchessed-UCI-Engine`, branch `manus/rustc-bootstrap-trial` (the requested research-facilities line).  
**Decision question:** Do the opt-in `PersonaSmooth` and `EngineDetectV2` paths improve observable persona stability and opponent classification in real games, without creating unacceptable false positives or changing the separate playing-strength conclusion?

## Executive recommendation

**Pursue, conditionally, as a bounded observational real-play study; defer any default flip until its preregistered behavioural gates pass.** The study is worthwhile because the existing interrupted engine match measured Elo, not flip rate or detection accuracy. It should not be presented as a strength experiment, and it should not be used to replace a fresh, properly completed Elo SPRT if a default change is proposed. Drop the study if the required labelled/anonymous coverage cannot be recruited or if telemetry provenance is incomplete; do not manufacture labels from telemetry after seeing outcomes.

The existing code makes this a feasible, low-cost study: adapter telemetry is opt-in, the parser requires a runner-owned manifest, and the analyser already reports persona transitions, detector confusion counts, and basic coverage. The current defaults remain `false`; the proposal changes neither production code nor defaults.

## What was read and what was run

I read `docs/reinforcement/00-synthesis.md` through `05-oracle.md`, plus the persona gate/result notes, the high-level detector integration note, the SPRT scripts, `tools/analyse_adapter_telemetry.py`, `tools/elo_detector.py`, their tests, and the relevant `unchessed-core/src/adapt.rs` telemetry and state-machine code. I ran:

```text
python3 -m unittest tools.test_analyse_adapter_telemetry tools.test_elo_detector -v
```

This completed **13/13 tests successfully**. The tests include parser rejection of missing labels, duplicate records, unknown schema fields, and mid-game option changes; valid per-game aggregation; declared-human bands; Maia-versus-Stockfish classification; opening premove protection; and the anonymous perfect-low-loss trigger. `rustc` was present at `/usr/bin/rustc`; `cutechess-cli` was not found. I did not compile, launch an engine match, collect telemetry, run cutechess, run an SPRT, install dependencies, use cloud resources, commit, or push.

The repository already records these relevant real numbers, which are **not results of this proposed study**: the prior combined option match stopped at 5,537/10,000 games after interruption, with **2,125–2,095–1,317**, score fraction **0.503**, approximately **+2.1 Elo**, and neither SPRT bound crossed. The simulation note reports a **57.3% reduction** in mode flips, while the detector test simulation reports **0/200 engine flags** in each declared-human band (1,200, 1,600, 2,000, 2,200, and 2,500). These are historical/simulation evidence only; they do not establish real-play behavioural performance.

## Implementation contract and measurement units

Telemetry lines are UCI `info string [UnchessedTelemetry] v=1` records. The current parser joins by `(run, game)`, rejects duplicate event indexes, requires the manifest label, and rejects an option state that changes within a game. The manifest can carry `Adaptive`, `UCI_LimitStrength`, `PersonaSmooth`, `EngineDetectV2`, and `OwnBook`. Observation records expose `source`, `cp_loss`, `difficulty_weight_milli`, `legal_count`, `had_choice`, `clock_available`, `opp_time_used_ms`, `estimate_elo`, `confidence_cp`, `weight_milli`, `suspicion_milli`, `low_loss_streak`, `samples`, `is_computer`, `declared_elo`, `suspect`, `suspect_reason`, and `action_full`. Decision records expose raw/EMA evaluation, modes, candidate, dwell/emergency status, selected move, and the same option state.

The analysis unit is the **game**, with repeated observations and decisions nested inside it. A game is not an independent move. A player/session/opponent identity is a higher-level cluster and an opening identity is a crossed/repeated cluster. Raw JSONL telemetry, manifest, game PGN, engine binary hash, repository commit, UCI options, time-control configuration, runner version, and clock-source status must be retained immutably. Telemetry is diagnostic evidence, not an oracle and not a substitute for game outcomes.

## Preregistered design

### Arms and labelled matrix

Use a 2×2 factorial matrix, randomised at the game level within each opponent/session and opening block:

| Arm | `PersonaSmooth` | `EngineDetectV2` | Purpose |
|---|---:|---:|---|
| A: baseline | false | false | Current default path |
| B: smoothing only | true | false | Isolate EMA/dwell effect |
| C: detection only | false | true | Isolate detector effect |
| D: combined | true | true | Candidate shipping behaviour |

`Adaptive=true` is mandatory in every arm because otherwise the relevant persona/detector paths are bypassed. Keep `Threads=1`, fixed hash, fixed binary, fixed NNUE/eval file, `OwnBook=false`, and identical search limits across arms. Do not mix binary commits or unrelated options.

Recruit or generate pre-labelled opponent strata, with labels fixed before randomisation:

| Label stratum | Target | Required games per arm | Total games |
|---|---|---:|---:|
| Declared human, 1,200–1,799 | human-like, declared rating | 30 | 120 |
| Declared human, 1,800–2,399 | human-like, declared rating | 30 | 120 |
| Declared human, ≥2,400 | titled/strong human where available | 30 | 120 |
| Human-like engine (e.g. Maia-labelled) | known engine identity and rating | 30 | 120 |
| Conventional engine (e.g. Stockfish/Rubi) | known engine identity | 30 | 120 |
| Anonymous human/unknown | no identity/rating metadata exposed | 30 | 120 |
| **Minimum** | six strata | **30** | **720** |

The target is 30 games per cell, not 30 observations. If independent opponents are available, use at least 10 opponents per stratum and three games per opponent per arm; otherwise use at least 6 opponents and report the reduced effective sample size. A labelled game’s ground truth is the pre-registered roster (`human`, `human_like_engine`, or `conventional_engine`), not the engine’s own `is_computer` field. For anonymous cases, preserve the hidden truth in a sealed key held by the study operator and do not reveal it to the engine or analysis coder until the primary lock.

### Metadata-assisted versus anonymous cases

Each labelled stratum is run in two exposure conditions where feasible: **metadata-assisted** (the engine receives the normal GUI/`UCI_Opponent` kind, name, and declared Elo) and **anonymous** (same opponent and opening schedule, but omit identity/rating metadata). The minimum 720-game matrix therefore expands to 1,440 games if every stratum has both exposure conditions. If resources only permit one condition, prioritise anonymous cases for the primary detector claim and declare metadata-assisted results secondary.

Never compare metadata-assisted and anonymous games as if they were the same treatment: metadata changes the information available to the detector. Stratify randomisation and analysis by exposure condition. For a human-like engine, record both the true identity and the metadata shown; Maia must not be collapsed into the conventional-engine class.

### Paired openings and randomisation

Create a frozen opening set of **60 opening positions**, each with a unique ID, ECO/PGN provenance, side-to-move, and maximum start ply. Use 15 openings per broad family (open, semi-open, closed, flank), or an equivalent balanced set; do not select openings after seeing telemetry. For each opening ID, play a two-game colour-swapped pair: the same opponent and arm combination faces the same starting position with colours reversed. Randomise arm order, opponent order, and pair order with a predeclared seed. A second replicate of each pair is allowed only if the stopping rule has not been reached and must be marked as a replicate, not treated as independent evidence.

Use the same opening schedule across arms and balance colour. If a platform cannot provide identical opening positions, match by opening ID and report the deviation. Opening ply is fixed (for example, 16) and `OwnBook=false`; no adaptive opening selection is allowed. The primary behavioural comparison is within matched opening/opponent blocks, not an unpaired aggregate.

### Clock protocol

Use one declared fast-real-play protocol, for example **5+0.05** (five minutes plus 50 ms per move), and do not mix time controls in the primary analysis. Run on fixed hardware with one engine process per worker, `Threads=1`, fixed hash, and a stable CPU governor where available. Synchronise clocks through the match harness, record engine-reported `opp_time_used_ms`, and retain PGN clock annotations. A clock observation is eligible only when `clock_available=true` and the opponent time value is monotonic and non-missing; premoves/opening observations must remain present but be flagged by move phase, not silently deleted.

Predeclare phase bins (opening: plies 1–16; early middlegame: 17–40; late: >40) and a minimum timing threshold for a “rapid” reply based on the harness resolution (for example ≤100 ms, with the exact threshold fixed before data collection). Do not infer human/engine truth from time alone. If clock availability is below 90% of eligible observations in a cell, that cell is missing the timing endpoint rather than a negative result.

## Outcomes and analysis

### Primary behavioural endpoints

1. **Persona flip rate:** for game *g*, `F_g / max(D_g−1, 1)`, where `F_g` is the number of adjacent `mode_after` changes and `D_g` is the number of persona decisions. The primary contrast is D versus A, pooled only through a cluster-aware model. Report absolute difference, ratio, and 95% CI. A flip is a transition in the recorded mode, not a change in raw evaluation or candidate.
2. **Detector performance:** using sealed truth labels, calculate TP, TN, FP, FN and report sensitivity/recall, specificity, false-positive rate, precision, and balanced accuracy. Analyze both the final game-level `suspect` decision and the observation-level confusion counts. Do not call the `suspect` label “accuracy” in anonymous cases until truth is unblinded.
3. **Detection latency:** among true suspects, first qualifying suspect observation in ply and within-game observation count; among false positives, similarly report onset phase and signal (`labelled_computer`, ceiling, or clock). This is secondary but important for the documented opening discounts and streak gate.

Secondary endpoints are coverage (`observations`, `skipped`, `clock_available`, `had_choice`), option-state integrity, decision count, skipped-observation reasons, selected mode, emergency/dwell events, and per-phase flip/detection results. Report distributions, not only means. The analyser’s existing pooled counts are a starting summary; the preregistered analysis must preserve per-game rows and avoid treating pooled moves as independent.

### Clustered confidence intervals and contrasts

Use two-sided 95% confidence intervals, with the analysis plan frozen before unblinding. Primary inference uses a mixed-effects or GEE model with arm, exposure condition, truth stratum, colour, opening family, and phase as fixed effects; random intercepts (or cluster-robust standard errors) for opponent/session and opening ID; and the paired opening block as the matching unit. For binary detector outcomes, use a logistic mixed model or cluster-robust binomial intervals. For flip counts/opportunities, use a binomial model with game-level numerator/denominator and cluster-robust variance. If convergence fails, use a predeclared bias-corrected cluster bootstrap resampling opponent/session clusters and retaining all games within each sampled cluster; do not switch to ordinary iid intervals post hoc.

There are crossed clusters (opponent and opening), so resample or robustly account for both; a one-way game-level Wilson interval is descriptive only. Report the number of clusters, cluster-size range, ICC or variance component where estimable, and sensitivity intervals under ICC values 0, 0.02, 0.05, and 0.10. Cluster methods are appropriate because observations within a player/opponent or opening block share conditions and are not independent [1].

### Coverage and exclusions

The denominator for persona coverage is every valid persona decision emitted; the denominator for detector coverage is every eligible opponent observation, with clock coverage separately defined. Count malformed, duplicated, missing-manifest, changed-option, and missing-clock records. A game is excluded from the primary endpoint only for a predeclared protocol failure (wrong binary/options, incomplete PGN, or parser integrity error); a resign, timeout, disconnect, or early termination is an outcome/coverage record, not a convenient deletion. Never exclude a game because its detector result is surprising. A cell with fewer than 24 completed games or less than 90% valid telemetry is under-covered and cannot pass a success criterion.

## Sample-size reasoning

The proposed minimum has 30 games per arm×stratum×exposure cell and 10 independent opponents where possible. For paired proportions, 120 paired games per arm comparison is enough to estimate a large behavioural change but not a subtle one. As a planning illustration, if the baseline flip rate is 0.20 and the candidate is expected to reduce it to 0.10, an iid approximation requires roughly 199 games per arm for 80% power at two-sided α=0.05; the exact requirement depends on within-pair correlation and the paired discordance rate. Inflate the effective requirement by the cluster design effect `DE = 1 + (m−1)ICC`, where *m* is games per opponent cluster. With m=3 and ICC values 0.05/0.10, DE is 1.10/1.20, so 220–240 games per arm is a conservative benchmark for that large effect. The proposed 720-game minimum is therefore a precision/coverage design across strata, not a claim of 80% power for every cell.

For detection, a cell with 30 true negatives can only estimate a zero false-positive rate as an upper one-sided 95% bound of about 9.5% (the rule-of-three approximation); it cannot establish a 1% false-positive guarantee. A 90% sensitivity estimate from 30 true suspects has a wide interval. Accordingly, detection results are gated by predeclared practical thresholds and intervals, and the report must state when the study is underpowered. If the project needs a 2% false-positive upper bound after observing zero false positives, approximately 150 independent negatives are needed before clustering inflation; recruit more clusters rather than merely more games against one opponent.

These calculations are planning values, not observed results. Final sample size may increase for attrition, but stopping early for a favourable point estimate is prohibited. A sequential stop is allowed only for an explicitly predeclared safety failure (for example, severe false-positive harm), with all stopped data retained and labelled exploratory.

## Preregistered success criteria

The combined arm D may be recommended for a separate default-review gate only if all of the following are met in the primary analysis:

* at least 24 complete games in every required cell, ≥90% valid telemetry, ≥90% clock availability for timing endpoints, and no unresolved provenance or option-integrity failures;
* versus baseline A, the absolute flip-rate reduction is **≥25% relative** (and ≥5 percentage points where the baseline permits), with the two-sided 95% CI excluding zero in the prespecified primary contrast; the effect is directionally consistent in at least four of six truth strata and in both exposure conditions;
* for declared humans, false-positive rate is **≤5%**, with the upper 95% cluster-aware CI **≤10%**; no human stratum has a point estimate >10% without an adjudicated protocol explanation;
* for conventional engines, sensitivity is **≥90%** (cluster-aware lower 95% CI ≥75%); for human-like engines, the detector does not classify more than 10% as conventional-engine suspect unless the predeclared ground truth says they should be suspect;
* no arm produces a safety-relevant increase in emergency FULL actions, missing telemetry, or clock-dependent misfires, and metadata-assisted/anonymous conclusions are reported separately;
* the result is replicated directionally in the paired colour/opening analysis, not only in pooled moves.

Failure of a statistical significance condition is not evidence of harm; it is a **defer** outcome when precision/coverage is inadequate. A false-positive gate failure is a **do not ship** outcome until the cause is fixed and a new preregistered study is run. None of these behavioural criteria authorises an Elo claim.

## Separation from Elo SPRT

The existing `sprt_persona_smooth_detect.sh` compares the same binary with both options on versus both off at `5+0.05`, `Adaptive=true`, fixed openings, and SPRT bounds `elo0=0`, `elo1=5`, α=β=0.05. That is a playing-strength gate. This protocol instead estimates flips, labels, timing coverage, and detector errors in real play. It must not reuse an Elo result as evidence of behavioural success, and telemetry outcomes must not be added to an Elo likelihood as if they were game wins/losses.

If the behavioural gate passes and the owner wants to change defaults, run a **new completed real paired Elo SPRT** for the exact candidate defaults and binary; the interrupted 5,537-game result is not a formal bound crossing. Stockfish’s Fishtest documentation describes pentanomial match modelling and generalized SPRT for engine evaluation [2]. Those methods support the separation here: Elo evidence concerns match outcomes, while this protocol concerns clustered behavioural endpoints.

## Verified, assumed, and blocked

| Classification | Statement |
|---|---|
| Verified here | Branch is `manus/rustc-bootstrap-trial`; requested documents/code were read; telemetry and detector tests pass **13/13**; parser enforces manifest labels and stable options; `rustc` exists; `cutechess-cli` is unavailable. |
| Verified from repository records | Defaults are false; telemetry is opt-in; historical combined run is 2,125–2,095–1,317 at 5,537 games and ~+2.1 Elo without crossing an SPRT bound; simulation numbers stated above. |
| Assumed for design | Six strata, 60 openings, 30 games/cell, 10 independent opponents/stratum, 5+0.05 protocol, clock threshold and 90% coverage rule. These are proposed preregistration values, not measured facts. |
| Not run | No real-play matrix, telemetry capture, human recruitment, anonymisation experiment, paired-opening analysis, clustered CI, compiler build, cutechess match, or fresh SPRT. |
| Blockers | No `cutechess-cli` in this sandbox; no recruited/labelled real opponents or sealed roster; no validated clock harness and no generated study manifest. A real study needs those external facilities and an immutable runner. |

## References

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4521133/ "Rutterford et al., Methods for sample size determination in cluster randomized trials, International Journal of Epidemiology (2015)"  
[2]: https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html "Stockfish, Statistical Methods and Algorithms in Fishtest"  

The repository sources used for implementation claims are `tools/analyse_adapter_telemetry.py`, `tools/elo_detector.py`, `tools/test_analyse_adapter_telemetry.py`, `tools/test_elo_detector.py`, `unchessed-core/src/adapt.rs`, `docs/persona-engine-detect-uci-gate.md`, `docs/persona-smooth-detect-sprt-result.md`, and `scripts/sprt-history/sprt_persona_smooth_detect.sh`.
