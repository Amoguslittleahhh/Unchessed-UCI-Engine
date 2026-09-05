#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [A Safety-Gated CLINCH Adapter for Persona-Conditioned Human-Like Chess Engines],
  abstract: [
    Human-like chess engines require more than a scalar playing-strength limit. They must produce moves that are natural, preserve engaging positions, and make mistakes that resemble plausible human errors without sacrificing tactical integrity. This paper presents a dedicated CLINCH adapter for the Unchessed UCI chess engine. The adapter combines a search-derived centipawn-loss budget, shallow reply-concentration probing, policy-net naturalness, and an engagement heuristic. We also refine the broader persona system with phase-aware natural-blunder scheduling and bounded engagement shaping in MATCH mode. The design preserves the existing FULL, PUNISH, DEFEND, and MATCH semantics by making the policy a constrained selector rather than an evaluator. The implementation is validated by the Rust workspace test suite, including 131 passing tests and tactical-search regressions. We describe the algorithm, safety invariants, implementation details, limitations, and a proposed experimental protocol for future Elo and human-likeness evaluation.
  ],
  authors: ((
    name: "Manus AI",
    department: [Unchessed AI Research],
    organization: [Independent Engineering Research],
    location: [Sandboxed Research Environment],
    email: ""
  ),),
  index-terms: ("chess engine", "human-like play", "persona adaptation", "policy network", "CLINCH", "tactical safety"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction <sec:intro>

Modern chess engines optimize move quality, but a human-facing engine has a different product objective. It may need to match an opponent's level, punish a mistake, defend an inferior position, or create a tense and educational game. These goals cannot be reduced to selecting the move with the highest evaluation. A humanized engine must satisfy three simultaneous requirements: moves should be *natural*, positions should remain *engaging*, and errors should be *natural blunders* rather than arbitrary random degradation.

Unchessed already separates these goals through a persona state machine. Its FULL mode preserves raw engine strength; MATCH samples human-plausible alternatives; PUNISH responds to an opponent's mistake; DEFEND maximizes resistance; and CLINCH seeks a difficult but winnable game in a drawish position. The previous CLINCH implementation was an inline heuristic over a small MultiPV set. It measured the gap between an opponent's best and second-best replies, but did not treat policy naturalness or post-move engagement as first-class signals.

This paper introduces a dedicated CLINCH adapter with an explicit objective and safety contract. It is designed as a layer between search and final move selection. The adapter never creates a move, never changes the evaluator, and never permits a candidate outside a strict search-approved loss window. Instead, it ranks legal candidates according to tactical acceptability, opponent reply concentration, human policy preference, and engagement.

The contribution is deliberately conservative. A fixed-position parity result is not treated as an Elo result, and the implementation does not claim a strength gain without a paired-game experiment. The paper therefore distinguishes engineering correctness from behavioral and playing-strength validation.

= Background and Related Work <sec:background>

== Policy networks and human-like play

Leela Chess Zero describes a policy as a probability distribution over moves that guides search, while a value estimate evaluates the resulting position @lc0. Maia instead trains neural networks to predict human moves rather than optimal engine moves @maia. Maia-2 extends this direction with skill-aware attention, integrating player skill with encoded board features and evaluating both move-prediction accuracy and coherence across skill levels @maia2.

These ideas motivate a strict separation in Unchessed. The policy net should express *naturalness*, not tactical truth. Search remains responsible for tactical correctness, while the persona system determines how much naturalness is allowed to influence the final decision.

== NNUE evaluation and low-latency inference

Stockfish's NNUE documentation emphasizes sparse features, incremental updates, simple low-precision layers, and quantization @stockfish. These principles are relevant because a humanization layer must not impose an uncontrolled search-time cost. The current Unchessed design therefore leaves NNUE evaluation and search unchanged and applies persona logic only after the candidate lines have been generated.

== Dataset quality and evaluation leakage

The quality of the training distribution is as important as model architecture. Research on NNUE datasets reports that unstable or tactically noisy positions can harm convergence and deployment behavior @dataset. The Unchessed training branch consequently combines quiet-position filtering with a duplicate-safe validation split. That split assigns identical board positions to the same partition so that checkpoint selection does not benefit from exact-position leakage.

= Existing Persona Architecture <sec:existing>

The persona system maintains an opponent model, a smoothed evaluation state, and a mode with hysteresis. The opponent model estimates playing strength from observed centipawn loss and timing signals. Persona transitions use confidence-aware thresholds, emergency transitions for engine suspicion or severe evaluation collapse, and dwell periods to avoid flapping.

The final selector receives search lines, the current mode, the opponent model, a move-prior provider, and a shallow probe callback. FULL and DEFEND select the best line directly. PUNISH prefers forcing moves among near-best lines. MATCH builds a wider candidate pool at lower target Elo and samples according to evaluation loss and policy priors. Before this work, CLINCH evaluated only the first three lines and added a reply-gap bonus.

#figure(
  placement: top,
  table(
    columns: (auto, auto, 1fr),
    inset: 5pt,
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) },
    table.header[Mode][Primary objective][Policy role],
    [FULL], [Raw engine strength], [Bypassed],
    [MATCH], [Human-like level matching], [Strong naturalness prior],
    [PUNISH], [Convert opponent mistakes], [Tie-breaker among forcing moves],
    [DEFEND], [Maximum resistance], [Small guide influence],
    [CLINCH], [Win a drawish game engagingly], [Naturalness among probe-qualified traps],
  ),
  caption: [Persona modes and the proposed division of responsibility.],
) <fig:modes>

The design principle is that a policy prior is a *selector aid*. It is never allowed to remove all engine-approved alternatives, introduce a move that search did not examine, or suppress a forced mate.

= CLINCH Adapter <sec:clinch>

== Objective

Let $m$ be a candidate move, $L(m)$ the centipawn loss relative to the best search line, $G(m)$ the reply-concentration gap after the move, $P(m)$ the human-policy prior, and $E(m)$ an engagement score. The CLINCH adapter maximizes

$ S(m) = -L(m) + alpha G(m) + beta ln(P(m) + epsilon) + gamma E(m), $

subject to

$ 0 <= L(m) <= L_max $.

The implementation uses $L_max = 40$ centipawns, $alpha = 0.60$, $beta = 10$, $gamma = 1$, and $epsilon$ is represented by a bounded positive prior floor. The coefficients are tuning anchors rather than claims of optimality.

== Candidate safety filter

The adapter considers at most the top five search lines and discards every candidate whose loss exceeds the CLINCH budget. This preserves a meaningful tactical margin while giving CLINCH more room than a pure best-move selector. If no line survives, the best move is returned with a fallback reason.

For every surviving candidate, the adapter performs the existing shallow probe after the candidate move. If the probe returns at least two replies, the gap between the first and second reply scores estimates how narrowly the opponent must respond. A large gap is useful because it identifies positions where the opponent has a difficult practical choice.

== Naturalness and engagement

The policy prior is evaluated only on the filtered candidate set. A move with a high human-policy probability receives a naturalness bonus, but that bonus cannot overcome the loss ceiling. Engagement receives two small bonuses. The first applies when the probed position has a moderate number of legal replies, representing meaningful choice rather than a sterile forced line. The second applies when both queens remain on the board, preserving tactical tension in positions where CLINCH is intended to create a challenging game.

This realizes the three golden rules as separate terms:

#enum(
  [*Natural moves:* supplied by the trained policy prior and bounded by search loss.],
  [*Engaging positions:* supplied by reply-choice and queen-tension signals.],
  [*Natural blunders:* handled in MATCH through a separate calibrated error regime, not by corrupting CLINCH's safety boundary.],
)

The third rule is important. CLINCH should create pressure without intentionally blundering. Natural errors belong primarily to MATCH, where they can be controlled by opponent level, game phase, and a loss band.

== Algorithm

```text
best <- first search line
safe <- top five lines with loss <= 40 cp
if safe is empty: return best
priors <- policy(position, safe, target_elo)
for candidate in safe:
    after <- make(candidate)
    replies <- shallow_probe(after)
    gap <- score(reply_1) - score(reply_2), or zero
    engagement <- moderate_reply_choice + queen_tension
    score <- -loss + 0.60*gap + 10*log(prior) + engagement
return candidate with maximum score
```

The adapter is deterministic given the search lines, policy output, and probe results. Randomness is not required for CLINCH because the goal is to select a strategically difficult line, not to simulate an error distribution.

= Persona Algorithm Improvements <sec:persona>

== Phase-aware natural blunders

The previous MATCH blunder probability depended mainly on target Elo. This produced a uniform error schedule across the game. The revised function preserves the existing Elo relationship but multiplies it by a phase factor: 0.45 in the opening, 1.0 in the middlegame, and 0.75 in the late game. The schedule reflects the intended behavioral distinction between rehearsed opening play, complex middlegame decisions, and endgame precision.

The resulting probability is

$ p_b(r, t) = 0.35 max(0, (2200-r)/1700) f(t), $

where $r$ is target Elo and $f(t)$ is the phase factor. At target Elo 2200 or above, the deliberate blunder regime is disabled. The normal MATCH selector still applies its evaluation-loss budget and tactical mate filter.

== Bounded engagement in MATCH

MATCH now adds a small engagement multiplier to normal candidate weights. A candidate receives a modest bonus when it leaves the opponent a moderate number of legal replies or preserves both queens. Since the multiplier is at most 1.10, it cannot compensate for a large evaluation loss. This keeps engagement subordinate to both search and naturalness.

== Persona separation

The improved algorithm avoids flattening all modes into one policy sampler. FULL remains an exact best-line mode. PUNISH remains forcing and conversion-oriented. DEFEND remains resistant and does not use the natural-blunder regime. CLINCH is now a dedicated adapter with its own objective. MATCH is the only mode where realistic blunder sampling is expected.

= Implementation <sec:implementation>

The implementation adds a public `ClinchAdapter` to `unchessed-core/src/adapt.rs`. Its parameters are explicit, cloneable, and independently testable. The existing `MovePrior` trait remains unchanged, so both the heuristic prior and the Maia-style policy net can be used without an artifact-format migration.

The current policy integration uses the existing `prior.priors(position, moves, target_elo)` interface. This makes the new adapter immediately compatible with legacy policy artifacts. A future dual-Elo runtime should extend the context to include opponent Elo and persona kind, matching the dual-Elo pretraining direction and the skill-aware conditioning described by Maia-2.

The NNUE runtime is not modified by this work. This is intentional: evaluation and humanization have separate responsibilities. The search score establishes the safety boundary, while the policy and persona layers operate over already generated candidates.

= Validation <sec:validation>

== Software correctness

The modified branch was compiled with Rust 1.98.1. The complete Rust workspace test suite passed 131 tests, with six pre-existing ignored tests and zero failures. The passing tests include move generation, search, NNUE behavior, policy-related UCI handling, tactical mate protection, stale-hint rejection, and the new humanization regressions.

The new tests verify that middlegame blunder probability exceeds opening and endgame probability, target Elo 2200 disables deliberate blunders, and engagement scoring remains bounded. The existing tactical tests continue to verify mate selection, hanging-queen capture, and fail-closed behavior.

== Tooling

Cute Chess 1.5.1 was installed successfully and reports its Qt and Ubuntu runtime versions. A direct UCI node-limited search returned `readyok` and a legal move. A short process-level Cute Chess match did not complete within the sandbox timeout; therefore, this paper makes no Elo claim.

#figure(
  placement: top,
  table(
    columns: (1fr, auto, 1fr),
    inset: 5pt,
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) },
    table.header[Validation item][Result][Interpretation],
    [Rust workspace tests], [131 passed], [No regression detected in the available suite],
    [Ignored Rust tests], [6], [Existing ignored tests; not counted as failures],
    [Release build], [Passed], [All workspace binaries linked successfully],
    [Direct UCI test], [Passed], [`readyok` and legal node-limited move],
    [Cute Chess version], [1.5.1], [Harness installed and launches],
    [Elo or human-likeness result], [Not measured], [Requires trained artifacts and paired games],
  ),
  caption: [Validation status for the implementation round.],
) <fig:validation>

== Threats to validity

The coefficients were not tuned by a statistically powered match campaign. The sandbox did not contain a trained dual-Elo deployment artifact or PyTorch, so policy quality and human-likeness could not be measured end-to-end. The engagement heuristic is intentionally simple and should not be interpreted as a learned model of human interest. Reply-gap concentration is a practical proxy for difficulty, not a complete measure of educational or aesthetic value.

A further limitation is that the current CLINCH adapter examines the existing MultiPV lines rather than generating a separate policy-guided candidate pool. This is safe and inexpensive, but a future experiment should compare it with a wider legal candidate pool filtered by shallow search.

= Experimental Protocol <sec:protocol>

A credible evaluation should use fixed binaries, fixed evaluator files, fixed books, fixed hash and thread settings, and immutable model hashes. The primary ablation should compare four configurations: no policy, policy in MATCH only, the dedicated CLINCH adapter, and a policy applied indiscriminately to every persona. Each configuration should be evaluated both with Adaptive off and with Adaptive on.

The behavioral metrics should include policy top-1 accuracy, candidate naturalness, average centipawn loss, blunder frequency by game phase, queen-retention frequency, legal-reply entropy, probe reply gap, mode-transition rate, and tactical-failure count. A paired-game SPRT should be used for playing-strength claims. Human-likeness claims additionally require a held-out human-move prediction set and a blinded human evaluation.

The proposed acceptance rule is asymmetric. Any tactical failure is a hard rejection. A candidate may be promoted only if it preserves the tactical suite, does not reduce raw strength beyond the pre-registered tolerance, and improves at least one humanization metric without materially worsening the others.

= Conclusion <sec:conclusion>

This work turns CLINCH from an inline heuristic into a dedicated, safety-gated adapter. The design operationalizes natural moves, engaging positions, and natural blunders without conflating them. Search remains the authority for tactical correctness; the policy net supplies naturalness; probe analysis supplies practical pressure; and the persona state machine decides which behavioral regime is appropriate.

The implementation is deliberately compatible with the existing Unchessed architecture and is validated at the software-correctness level. The next step is not to enable the adapter unconditionally, but to run the ablation and SPRT protocol with a trained policy artifact and a fixed NNUE. Only then can the project determine whether the adapter improves human-facing play in practice.

