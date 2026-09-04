# Live Rapid Benchmark and Persona Stability Review

**Date:** 2026-09-04
**Branch:** `manus/research-facilities`
**Target:** Unchessed Heavy Optimisation after confidence-aware Elo/persona coupling commit `6b0d84d`

## Executive summary

A live eight-game short-rapid benchmark was completed at **3 minutes plus 2 seconds increment** from the standard starting position. The updated Unchessed build scored **0/4 against Stockfish 16** and **4/4 against Maia-3 5M configured for Elo 1500**. These are useful smoke-test observations, not a statistically reliable Elo estimate: each pairing contains only four games, all games use one opening start position, and the engine implementations have very different objectives. The four-game results have wide Wilson 95% intervals: **0–48.99%** for Unchessed against Stockfish and **51.01–100%** against Maia-3.

The match harness completed all games legally and recorded checkmate terminations. The Stockfish games averaged 123.5 plies and 550.71 seconds; Maia-3 games averaged 57.5 plies and 211.15 seconds. The Maia comparison is especially not an objective-strength test because Maia-3 is a human-move predictor rather than a conventional maximising search engine.

## Benchmark configuration

| Item | Configuration |
|---|---|
| Unchessed | `unchessed-adapter` release build from the audited optimisation copy |
| Stockfish | Stockfish 16, Debian package, one engine process with 1 thread and 32 MiB hash |
| Maia | Official Maia-3, 5M checkpoint, CPU, no AMP, Elo 1500, UCI history enabled |
| Time control | 180,000 ms initial time, 2,000 ms increment per side |
| Games | 4 Unchessed–Stockfish, 4 Unchessed–Maia-3, alternating colours |
| Opening | Standard starting position, no opening book for Unchessed |
| Hardware | Linux x86-64 VM, six visible CPUs |
| Termination | All eight games ended by checkmate; no illegal moves or time forfeits |

## Results

| Opponent | Games | Unchessed wins | Draws | Unchessed losses | Score | Wilson 95% interval | Mean plies |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stockfish 16 | 4 | 0 | 0 | 4 | 0.0% | 0.0–48.99% | 123.5 |
| Maia-3 5M, Elo 1500 | 4 | 4 | 0 | 0 | 100.0% | 51.01–100.0% | 57.5 |
| **Combined** | **8** | **4** | **0** | **4** | **50.0%** | Not pooled across unlike opponents | **90.5** |

The Stockfish result should not be interpreted as evidence that the updated Elo calibration weakened the engine by a measured Elo amount. A four-game sample against Stockfish is far too small for that conclusion, and the opponent has a major objective-search advantage. Similarly, the Maia result demonstrates that Unchessed can decisively outplay this particular 1500-Elo human-policy configuration under this setup; it does not establish a general rating advantage over Maia or over human players.

## Persona telemetry finding

The first benchmark artifact reports zero persona decisions. This was traced to a **benchmark-harness parser defect**, not to the engine: the runner's regular expression required Elo and confidence fields on `persona_decision` lines, while those fields are emitted on opponent-observation events. A direct UCI smoke test confirmed that enabling `AdapterTelemetry=true` produces persona telemetry, including `mode_after=MATCH`.

The harness was corrected to parse `event=persona_decision` by its mode field independently. The completed eight-game artifact predates that parser correction, so it must not be used to claim measured persona transition counts. The corrected harness is committed alongside the report for the next benchmark run. This is an important reproducibility limitation and is intentionally disclosed rather than backfilling telemetry from unrecorded output.

## Persona stability code review

The review covered the complete independent audit, the earlier cross-engine integration plan, the persona implementation, and the Elo coupling change history.

### State-machine stability

`PersonaState` uses an exponential moving average with `ALPHA = 0.35`, a two-observation dwell requirement for non-emergency mode changes, and a two-ply cooldown after a deliberate transition. This prevents one-ply evaluation noise from repeatedly switching between MATCH, CLINCH, PUNISH, and DEFEND. The first observation seeds the EMA without counting as a vote.

Emergency transitions are explicit and limited to three classes: suspected engine opponents force FULL play, a severe raw or smoothed evaluation collapse enters DEFEND, and a fresh opponent blunder while ahead enters PUNISH. Emergency transitions clear dwell and cooldown state, which is appropriate for safety but means they intentionally bypass smoothing.

The uncertainty band widens the CLINCH entry deadband when evidence is sparse. This is a sound use of uncertainty because CLINCH is the most sensitive mode to a narrow evaluation interval around zero. However, this mechanism affects CLINCH entry only; it does not suppress a verified blunder emergency, which is deliberately treated as higher-confidence evidence.

### Elo calibration and coupling

The updated code no longer uses the Elo point estimate alone to infer a weak opponent. The weak-opponent PUNISH branch now requires the target plus its confidence band to remain substantially below the effective engine ceiling. This prevents a fresh 1500 prior with a wide confidence interval from entering PUNISH after an ordinary advantage.

The MATCH target adds one quarter of the confidence band to `estimate + 60`. As evidence accumulates, the confidence interval narrows and this premium decreases. The result is a cautious early policy that converges toward the established estimate after more observations.

The detector remains deliberately conservative for anonymous humans: sustained high-quality play needs substantial weighted evidence and a low-loss streak, while declared humans are exempt from the V2 ceiling detector. Known computers can still force FULL when their declared or inferred strength passes the V2 threshold. This policy protects humanisation from false engine accusations but makes engine detection dependent on correct UCI metadata for some opponents.

### Numerical and protocol safeguards

The independent audit fixed non-finite policy priors by rejecting NaN, infinity, and non-positive values and clamping valid weights. It also changed malformed `searchmoves` handling to fail closed instead of searching unrestricted moves. The review found no regression in the full validation suite: the latest relevant run passed **129 tests, 0 failures, 6 ignored**, plus release build and UCI smoke checks.

The audit also retained correct stop-token reset and worker joining, historical-ply opponent observation accounting, stale Aegis hint identity protection, and legal alpha-beta authority over external or neural hints.

## Reproducibility artifacts

The raw eight-game result file contains every move sequence, result, termination reason, ply count, and elapsed time. The corrected runner and analyzer are included in the optimisation copy. The analyzer uses Wilson intervals because four-game normal approximations are misleading at 0% and 100% scores.

## Recommended next experiment

The next run should use the corrected telemetry parser and at least 50 colour-balanced games per pairing, with a fixed opening suite rather than a single start position. For objective strength, use Stockfish at a deliberately matched node or time budget and pre-register an SPRT; for Maia, report human-move agreement, blunder rate, and mode-transition behavior rather than treating the result as an engine Elo match. Persona stability should be evaluated with transition counts, dwell violations, emergency-transition rates, confidence trajectories, and latency percentiles.
