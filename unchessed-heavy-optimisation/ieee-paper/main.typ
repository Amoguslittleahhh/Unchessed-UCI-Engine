#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Confidence-Aware Elo Calibration and Persona Stability in an Adaptive UCI Chess Engine],
  abstract: [
    Adaptive chess engines must balance objective tactical strength with controlled humanisation. This paper presents an engineering audit and live evaluation of an isolated Unchessed UCI engine optimisation branch. The work preserves the five protected personas—Full, Match, Clinch, Punish, and Defend—while hardening their state transitions, opponent-Elo inference, policy-prior handling, UCI restrictions, and neural-root interfaces. The central new change is confidence-aware Elo/persona coupling: weak-opponent Punish activation uses an upper confidence bound rather than a point estimate, while Match target strength includes a decaying uncertainty premium. A two-ply transition cooldown and exponential evaluation smoothing reduce non-emergency persona flapping without suppressing tactical emergencies. A live eight-game 3+2 benchmark scored 0/4 against Stockfish 16 and 4/4 against Maia-3 5M at Elo 1500. These results are explicitly treated as small-sample stress observations rather than a reliable Elo claim. All eight games completed legally by checkmate. The paper also documents a benchmark telemetry parser defect discovered during review, the corrective action, reproducibility artifacts, and a roadmap for larger colour-balanced SPRTs and human-policy evaluation.
  ],
  authors: (
    (
      name: "Manus AI",
      department: [Independent Engine Systems Research],
      organization: [Unchessed Heavy Optimisation],
      location: [Research branch: manus/research-facilities],
      email: ""
    ),
  ),
  index-terms: ("UCI chess engine", "persona stability", "Elo uncertainty", "human-like chess", "Stockfish", "Maia-3", "neural root priors"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Modern chess engines optimize a narrow objective: maximize the quality of the selected move under a search and evaluation budget. A human-facing adaptive engine has an additional policy objective. It may need to play at a target strength, preserve recognizable behavior, react differently to a blunder than to a positional advantage, and remain safe when the opponent model is uncertain. These requirements create a control problem around the search rather than a replacement for search.

Unchessed addresses this problem with five protected personas: Full, Match, Clinch, Punish, and Defend. The optimization branch studied here retains alpha-beta search as the authority for legality, tactical completion, and final move validity. Human-policy and neural evidence can influence ordering or candidate sampling, but cannot silently change persona semantics or bypass safety gates. This separation is consistent with the different roles of Stockfish-style search and NNUE evaluation, Lc0 policy/value inference, Maia human-move prediction, and AlphaZero-style self-play policy learning [@stockfish; @nnue; @lc0; @lc0tech; @maia; @alphazero].

The present work makes four contributions:

+ It audits and strengthens the persona state machine with evaluation smoothing, dwell, cooldown, emergency overrides, and uncertainty-aware CLINCH handling.
+ It changes opponent-Elo coupling so sparse evidence cannot prematurely classify an unknown opponent as weak.
+ It documents fail-closed boundaries for UCI restrictions, neural root hints, and malformed policy weights.
+ It reports a live 3+2 comparison against Stockfish 16 and official Maia-3 5M, including the important limitation that the first match artifact did not retain persona counts because of a harness parser defect.

The paper is an engineering report rather than a claim of Stockfish parity or a reconstruction of proprietary AlphaZero infrastructure. The public AlphaZero paper establishes a policy/value self-play method with Monte Carlo tree search, legal action masking, and visit-count targets, but not all production code, weights, or distributed training details [@alphazero].

= Protected persona contract

The persona layer consumes completed search lines and opponent evidence. It does not replace the underlying search. The intended contract is summarized in @tab:personas.

#figure(
  placement: top,
  table(
    columns: (1.25fr, 3.95fr),
    align: (left, left),
    inset: 4pt,
    stroke: (x, y) => if y == 0 { (bottom: 0.5pt) },
    table.header[Persona][Protected behavior],
    [Full], [Use the strongest completed alpha-beta result; do not humanise against a verified engine opponent.],
    [Match], [Play near the target Elo using bounded human-plausibility priors while retaining tactical safety.],
    [Clinch], [Prefer controlled, low-risk conversion when the game is late and evaluation is narrow.],
    [Punish], [Exploit verified blunders and large, well-supported skill gaps; never decline a found mate.],
    [Defend], [Maximize resistance after a severe evaluation collapse and do not repel safe draws.],
  ),
  caption: [Persona invariants preserved by the optimization branch.],
) <tab:personas>

The state controller has two classes of transition. Ordinary changes require a stable candidate proposal and dwell; emergency changes respond immediately to high-confidence tactical or opponent evidence. This distinction is important: smoothing is useful against evaluation noise, but a severe collapse or verified blunder must not wait for a multi-ply filter.

= Elo model and confidence-aware coupling

== Evidence model

The live opponent model begins with a broad prior near 1500 Elo. A move's centipawn loss is mapped to an Elo sample by a logarithmic curve:

$ E_"sample" = min(max(2950 - 850 ln(1 + L / 20), 400), 3200), $

where $L$ is the measured centipawn loss. Observations are weighted by position difficulty. Book moves, forced positions, and near-trivial choices carry less evidence than positions with multiple meaningful alternatives. The running mean is updated with a bounded evidence weight and a small decay so the model can track fatigue or changing play quality.

Clock evidence is deliberately weak and gated. Near-instant strong moves count only when the position offered real choice and the opening discount has expired. This prevents memorized opening play or premoves from being treated as engine evidence. A sustained ceiling pattern requires weighted evidence, a sufficient sample count, and a low-loss streak. Declared human opponents are not automatically upgraded to Full merely because they play a strong game.

The model exposes an uncertainty band:

$ C = 600 / sqrt(w) dot max(0.6, min(2.0, sqrt(V) / 400)), $

where $w$ is the effective evidence weight and $V$ is the exponentially weighted variance of Elo samples. The band is used as a policy confidence measure, not as a statistically calibrated rating interval.

== Failure mode in the original coupling

The previous weak-opponent trigger relied too heavily on the point estimate. At the beginning of a game, the point estimate is near the prior but the confidence band is wide. A normal +250 cp advantage could therefore be interpreted as proof that the opponent was weak, causing premature Punish behavior and excessive humanisation.

== New coupling rule

The revised weak-opponent trigger computes:

$ U_"opponent" = T + C,$

where $T$ is the Match target Elo and $C$ is the confidence band. The skill-gap branch can enter Punish only if $U_"opponent" + 500$ remains below the effective engine ceiling and the evaluation lead exceeds +250 cp. The independent fresh-blunder emergency remains unchanged.

The Match target is now:

$ T = min(T_"cap", max(500, E + 60 + C / 4)). $

The uncertainty premium is deliberately modest. It biases early Match play toward stronger, safer decisions against an unknown opponent and decays as evidence accumulates. This is not a claim that the confidence band is a formal Bayesian posterior; it is a conservative control heuristic designed to reduce false persona transitions.

== Transition filtering

The persona evaluation uses an exponential moving average with coefficient 0.35 on the newest completed search score. The first score seeds the filter and does not count as a vote. A normal candidate must receive two agreeing updates. After a deliberate transition, a two-update cooldown prevents an immediate non-emergency reversal. During cooldown, the conflicting mode is retained as diagnostic candidate state but cannot replace the active mode.

The uncertainty band also widens the CLINCH entry deadband when evidence is sparse. Emergencies bypass dwell and cooldown: suspected engine opponents can force Full, a raw or smoothed evaluation below -220 cp can force Defend, and a fresh opponent blunder while ahead can force Punish. Opt-in telemetry reports the active mode, candidate, dwell, cooldown, and emergency reason without feeding telemetry back into selection.

= Safety and interface audit

== UCI lifecycle and restrictions

The independent audit checked command lifecycle, stop handling, worker joining, option changes, `ucinewgame`, ponder cancellation, position replacement, and quit. Stop tokens are reset before new searches, workers are joined before state-changing commands, and stale ponder results are discarded. Evaluator changes clear the transposition table.

A specific fail-open bug was fixed in `searchmoves`. If a UCI request contained only illegal or stale restricted moves, the old implementation could search the unrestricted legal move set. The corrected implementation intersects the requested strings with generated legal moves and returns an empty result when the restriction is entirely invalid. The UCI adapter then emits `bestmove 0000` rather than inventing an unrestricted move.

== Policy-prior numerical firewall

Human-policy providers return relative weights rather than authoritative scores. Each weight is sanitized before sampling: non-finite and non-positive values become a finite default, and valid values are clamped to the interval $[0.1, 100.0]$. Missing entries receive the same safe default. This prevents NaN propagation, infinite totals, candidate disappearance, and pathological concentration.

== Neural root-prior firewall

The internal root-hint path begins with Unchessed's own legal move generator. Every root record is created from a legal move. A hint can affect ordering only if its move matches a legal root move and its policy score is finite. The hint vector never becomes the legal move list, and completed alpha-beta scores remain authoritative.

A future external Lc0 provider must bind position history, legal-action fingerprint, model and schema versions, en-passant state, halfmove state, request token, and deadline. Lc0 policy, value, and visit quantities must remain provider-specific evidence; they must not be averaged directly with Stockfish centipawns or Maia WDL predictions. The child process must complete the UCI handshake, drain both output streams, stop and drain before reuse, and validate every returned move against the local legal generator [@lc0tech; @uci].

These controls are semantic firewalls, not cryptographic security mechanisms. They protect against stale responses, malformed values, illegal moves, and timing races, but do not prove that a neural model is strategically correct.

= Implementation and validation

The isolated branch is a copy of the repository's main branch under `unchessed-heavy-optimisation`; the main branch was not edited. The implementation is Rust-based and was validated with the current stable toolchain available in the sandbox. The full workspace validation passed 129 tests, with zero failures and six ignored tests. Release compilation, UCI smoke checks, deep perft, and the earlier portable build checks also passed.

The implementation was reviewed through the audit history in commits `704381f` and `6b0d84d`. The former fixed fail-open search restrictions and malformed policy weights. The latter introduced confidence-aware Elo coupling and regression tests for premature Punish prevention and shrinking uncertainty premium. The relevant tests verify that a fresh uncertain prior does not trigger weak-opponent Punish, while sustained low-quality evidence can still trigger it. The optimization copy now includes the canonical `unchessed-nnue.bin` asset (SHA-256 `38845a16d73a6fe0bd4ac95c86c017c65c97bc82c7ce2f6dce2f1b3fbe8577b5`), and benchmark scripts require an explicit `EvalFile` path.

= Main-branch paired comparison

An externally supplied informal paired-game result compared the `main` adapter with the isolated optimization adapter. The reported configuration used six openings, alternating colors, 400 ms per move, one search thread, 64 MiB Hash, `OwnBook=false`, `Adaptive=false`, and the same explicit NNUE file on both engines. The report stated that neither side produced an illegal move or crashed.

The supplied result was `main 7 -- heavy-optimization 5, 6 draws`. These counts sum to 18 games, although the report also described the experiment as 12 games. The arithmetic inconsistency is preserved rather than silently corrected. Interpreting the win, loss, and draw counts literally gives the following descriptive table.

#figure(
  placement: top,
  table(
    columns: (1.6fr, 0.6fr, 0.6fr, 0.6fr, 0.9fr, 0.9fr),
    align: (left, right, right, right, right, right),
    inset: 3pt,
    stroke: (x, y) => if y == 0 { (bottom: 0.5pt) },
    table.header[Adapter][Wins][Draws][Losses][Game points][Score],
    [Main branch], [7], [6], [5], [10/18], [55.6%],
    [Heavy optimization], [5], [6], [7], [8/18], [44.4%],
  ),
  caption: [Externally supplied main-versus-optimization result. The literal counts imply 18 games, not 12.],
) <tab:main-compare>

This observation indicates operational stability under the reported fast control, but it does not establish a strength improvement. The heavy-optimization adapter scored 44.4% under the literal 18-game interpretation, a two-game-point deficit. The sample is too small and the control too fast to support an Elo estimate or a promotion decision. The result also disabled Adaptive behavior, so it does not test persona transitions, Elo detection, or low-time persona gates.

Because the result was supplied externally and no machine-readable PGN, exact opening list, engine commit identifiers, or SPRT log accompanied it, this paper treats it as an attributed observation rather than an independently reproducible experiment. A valid follow-up must resolve the game-count discrepancy, publish all PGNs and hashes, and use a predeclared paired-game SPRT.

= Live rapid benchmark

== Experimental setup

The live comparison used the release build of the updated Unchessed adapter, Stockfish 16 from the Debian package, and the official Maia-3 5M checkpoint configured for Elo 1500. The archived match was run before the evaluator-provenance correction and did not send `EvalFile`; it is therefore an HCE-only result, not a shipped-NNUE strength result. Maia-3 is a human-move prediction engine rather than an objective maximising engine; its result is therefore a human-policy stress observation, not a conventional objective-strength estimate [@maia3]. A fair NNUE match must rerun the same harness with `--eval-file /path/to/unchessed-nnue.bin`.

Eight HCE games were played from the standard starting position: four Unchessed--Stockfish games and four Unchessed--Maia-3 games, with colours alternated. Unchessed opening-book shortcuts were disabled. Each side received 180,000 ms plus 2,000 ms increment. The test ran on a Linux x86-64 virtual machine with six visible CPUs. All games ended by checkmate; no illegal moves, crashes, or time forfeits occurred. The result is retained as a clearly labelled HCE baseline; it is not used to claim performance for the NNUE configuration.

== Results

#figure(
  placement: top,
  table(
    columns: (1.5fr, 0.55fr, 0.75fr, 0.65fr, 0.65fr, 0.85fr, 1.35fr),
    align: (left, right, right, right, right, right, right),
    inset: 3pt,
    stroke: (x, y) => if y == 0 { (bottom: 0.5pt) },
    table.header[Opponent][Games][W][D][L][Score][Wilson 95%],
    [Stockfish 16], [4], [0], [0], [4], [0.0%], [0.0--48.99%],
    [Maia-3 5M, Elo 1500], [4], [4], [0], [0], [100.0%], [51.01--100.0%],
    [Combined], [8], [4], [0], [4], [50.0%], [not pooled],
  ),
  caption: [Eight-game 3+2 live benchmark. Wilson intervals reflect the four-game pairing sample size.],
) <tab:rapid>

The Stockfish pairing scored 0/4 for Unchessed. This is a useful stress result but not a reliable Elo estimate: four games, one starting position, and a strong objective-search opponent cannot support a parity claim. The Maia pairing scored 4/4 for Unchessed, with all games ending in checkmate. This indicates that the tested Unchessed configuration decisively outplayed the specific 1500-Elo Maia-3 policy configuration under this setup; it does not establish general superiority over Maia or human players.

The mean Stockfish game length was 123.5 plies and 550.71 seconds. The mean Maia game length was 57.5 plies and 211.15 seconds. The large timing difference reflects the different engine workloads and game lengths rather than a direct speed ranking.

== Telemetry reproducibility finding

The first completed match artifact reported zero persona decisions. During review, this was traced to a benchmark harness parsing defect. The engine emitted `event=persona_decision` lines, but the parser incorrectly required Elo and confidence fields on those lines; those fields are emitted on separate opponent-observation events. A direct UCI smoke test confirmed that `AdapterTelemetry=true` produces persona telemetry including `mode_after=MATCH`.

The parser was corrected and committed, but the eight completed games were not rerun after the correction. Consequently, the paper reports no fabricated persona transition count. The raw artifact remains valid for results, move legality, termination, and elapsed-time analysis, but not for measured persona-transition frequency. This limitation is a positive audit result: the missing instrumentation was identified and disclosed rather than silently inferred.

= Discussion

The confidence-aware coupling addresses a concrete control failure: sparse opponent evidence should not be treated as a confident weak-player classification. The change is intentionally asymmetric. Uncertainty raises the early Match target slightly, reducing the risk of underplaying a strong unknown opponent, while the weak-opponent Punish trigger requires the upper confidence bound to remain low. This makes false-positive humanisation less likely without preventing sustained weak-play adaptation.

Smoothing and cooldown address a different failure mode: noisy search evaluations can cause policy flapping even when Elo estimation is correct. The two mechanisms are complementary. Confidence affects whether a mode is proposed; dwell and cooldown affect whether a proposal is committed. Emergency evidence bypasses both when a tactical or detector signal is considered sufficiently strong.

The live benchmark cannot establish that these changes improve playing Elo. It did establish that the release artifact was runnable against two external UCI engines for eight games and that all games completed legally. The Stockfish result demonstrates that substantial objective-strength work remains before any Stockfish-parity claim could be credible. The Maia result demonstrates useful headroom for the tested human-policy target but should be followed by move-agreement and blunder-calibration studies, not merely more engine matches.

= Limitations and future work

The Elo sample curve and confidence band are engineering heuristics, not calibrated rating posteriors. Calibration should use rating-stratified human games with held-out players, position-difficulty buckets, clock controls, and proper scoring rules. The current benchmark uses one opening position and only four games per opponent. A stronger objective evaluation should use a fixed opening suite, colour balancing, matched threads and hash, fixed hardware, and a preregistered SPRT with an explicit null and alternative.

The Maia comparison should report move-match likelihood, legal top-k agreement, blunder frequency, and mode-specific behavioral statistics by target rating and time control. A future benchmark should rerun with the corrected telemetry parser and report transition counts, confidence trajectories, emergency rates, dwell violations, latency percentiles, and low-time fallback rates.

External Lc0 integration remains a proposed provider boundary rather than an implemented dependency. AlphaZero-inspired policy priors remain subordinate root-ordering evidence. Any future search change must first pass legality, perft, fixed-node equivalence, stale-result, and low-time gates before a strength experiment. Neural models should be pinned by binary, network hash, schema, backend, and runtime.

= Conclusion

This work strengthens an adaptive UCI engine without weakening the protected persona contract. The key change is to couple persona behavior to uncertainty-aware opponent Elo rather than to a point estimate alone. A wide early confidence band now suppresses premature weak-opponent Punish behavior, while a decaying Match uncertainty premium preserves caution against unexpectedly strong opponents. EMA smoothing, dwell, cooldown, emergency overrides, numerical sanitization, and fail-closed protocol handling provide complementary stability layers.

The live 3+2 evaluation completed eight legal games, producing a 0/4 result against Stockfish 16 and a 4/4 result against Maia-3 5M at Elo 1500. These observations are reproducible but not sufficient for an Elo claim. The most defensible conclusion is narrower: the optimized branch is operational, its safety and persona controls are test-backed, and its uncertainty-aware coupling is ready for a larger pre-registered calibration and strength study.
