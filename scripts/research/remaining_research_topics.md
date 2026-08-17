# Unchessed AI — remaining research backlog (post-IEEE-doc)

The 44-page research compendium (`unchessed_ieee.pdf`, also as a .docx) already
answers 19 of the ~104 brainstormed topics: NNUE v4 ablation, king-bucket
count, accumulator width, dual-net setups, quantization, incremental
accumulator updates, WDL exponent, alternative feature sets, output head
design, Syzygy label correction, curriculum/quiet-position filtering,
adversarial position mining, label noise/bootstrapping, SEE, correction +
continuation history, NNUE-eval-volatility time allocation, the pre-SPRT
smoke test, time-pressure policy net input, and blunder-rate calibration.

Everything below is what's **left** — 84 items across NNUE internals, data
pipeline, search, movegen/core, opening book, persona/adapter, policy net,
the (still unbuilt) Reviewer tool, testing discipline, calibration, and
longer-horizon ideas — plus 11 reinforcement items that cross-check gaps the
IEEE doc admits but never resolves. None of the 84 were part of the
19-question research prompt, so none have literature-backed answers yet.
Each entry gives: why it matters, a concrete way to test/answer it, and a
rough effort/impact read.

## NNUE / eval architecture, beyond the 9 answered questions

### 1. Eval scale/calibration check post-v4
- **Why it matters:** search constants (contempt=25, LMR reduction tables,
  futility margins) were originally tuned around v1's eval. If v4's raw
  output scale drifted even slightly from v1's after retraining, every
  cp-denominated constant in the search is now silently miscalibrated
  without any test having caught it — SPRT validates playing strength, not
  whether the *scale* the search assumes is still correct.
- **How to test:** run both v1 and v4 on a fixed 1000-position test set at
  equal search depth, compare the distribution (mean, variance) of raw
  eval outputs. If the distributions differ meaningfully, recompute the
  contempt/futility constants' effective cp meaning under v4 and check
  whether they should be rescaled.
- **Effort/impact:** small (a scripted eval-distribution comparison); worth
  doing once, cheap insurance against a class of bug that wouldn't show up
  in SPRT unless it were large.

### 2. Multi-perspective (STM + non-STM) simultaneous training
- **Why it matters:** the current pipeline likely trains on one
  perspective's label per position; Stockfish's modern recipe trains both
  the side-to-move and non-side-to-move accumulators against the same
  target simultaneously, which can improve sample efficiency without
  changing the network architecture at all.
- **How to test:** modify the training loop to compute loss against both
  perspectives per position (doubling effective supervision per sample
  without new data), retrain a v4-equivalent network, SPRT against the
  original v4.
- **Effort/impact:** small-to-moderate training-pipeline change, no
  inference-side changes; low risk since it doesn't touch the deployed
  architecture, only how it's trained.

### 3. Piece-value regularization
- **Why it matters:** small NNUE nets trained purely on WDL loss can drift
  from sane material intuitions in undertrained regions of position space
  (e.g. valuing a rook oddly in rare structures), because nothing in the
  loss explicitly anchors piece values to material-counting sanity.
- **How to test:** add a regularization term penalizing deviation from
  expected material-based eval on a synthetic "material-only" position set
  (KvK+material-imbalance positions with no positional complexity), retrain,
  compare SPRT strength and check for reduced blunder rate specifically in
  materially lopsided positions.
- **Effort/impact:** moderate (new loss term + synthetic test set
  construction); speculative payoff, best treated as a Tier 3 experiment.

### 4. Endgame-phase-specific sub-net or phase-interpolated output weighting
- **Why it matters:** distinct from the dual small+big net idea (already
  covered in the IEEE doc's Q4) — this is about *within* a single network,
  interpolating or switching output weights by game phase (similar to
  Stockfish's output buckets, but the doc's Q4 discussion of output buckets
  was in the context of dual-net comparison, not analyzed as its own
  question for Unchessed).
- **How to test:** add phase-conditioned output weights (keyed by material
  count, cheap to compute) to the existing 256-wide accumulator's output
  head, retrain, SPRT against v4. This is architecturally much cheaper than
  a dual-net setup since it doesn't require two full weight sets.
- **Effort/impact:** moderate; this is likely underrated relative to the
  dual-net idea the doc deprioritized, since output bucketing gets most of
  the phase-awareness benefit at a fraction of the cost.

### 5. Full 8-way symmetry exploitation beyond horizontal mirroring
- **Why it matters:** HalfKAv2_hm only exploits horizontal (left-right)
  symmetry; chess has no vertical or diagonal symmetry due to pawn
  direction and castling, so this is a genuine open question rather than
  an obvious win — worth explicitly ruling in or out rather than assuming
  it's free extra compression.
- **How to test:** analyze whether any additional symmetry actually holds
  once pawns and castling rights are accounted for (likely: none does,
  given pawns move in a fixed direction) — this is mostly a documentation/
  analysis task confirming horizontal mirroring is the ceiling, not an
  experiment.
- **Effort/impact:** trivial (a few hours of analysis); low expected payoff
  since the answer is probably "no further symmetry exists," but worth
  writing down explicitly so it stops recurring as an open question.

### 6. Label smoothing / soft-target training
- **Why it matters:** the current WDL loss trains against hard-computed
  target probabilities derived from search eval; label smoothing (blending
  a small uniform prior into the target distribution) is a standard
  regularizer in classification-style training that could reduce
  overfitting to noisy HCE-derived labels, complementing rather than
  replacing the WDL exponent tuning already planned (Q7).
- **How to test:** add a smoothing parameter to the WDL target computation,
  sweep a couple of small values (e.g. 0.01, 0.05) alongside the existing
  exponent grid search, since both are single-line training-loop changes.
- **Effort/impact:** trivial to implement, cheap to test alongside Q7's
  planned sweep; low individual expected impact but near-zero marginal
  cost to check.

## Training data / labeling, beyond the 4 answered questions

### 7. Self-play opening diversity / temperature sampling audit
- **Why it matters:** if self-play games used for NNUE labeling all start
  from similar openings (e.g. the engine's own book, played
  deterministically), the resulting 108M-position dataset could
  systematically under-represent large swaths of opening theory,
  compounding whatever stylistic blind spots the engine already has.
- **How to test:** compute the opening-move distribution (first 8-10 plies)
  across the actual training shards and compare entropy against a
  reference distribution (e.g. Lichess master games' opening frequency).
  Low entropy relative to the reference indicates a diversity problem worth
  fixing before the next training run.
- **Effort/impact:** cheap (a scripted analysis of existing shard data);
  directly informs whether curriculum learning (already in the doc's Q11)
  or the bootstrapping transition (Q13) needs an accompanying diversity fix.

### 8. Position dedup across shards
- **Why it matters:** if self-play games share early-game trajectories
  (plausible, since the same engine plays both sides against itself),
  near-duplicate positions could inflate the apparent 108M-position count
  without adding real training signal, silently reducing effective dataset
  size below what the training-time-budget math assumes.
- **How to test:** hash positions (Zobrist key already available) across a
  sample of shards, measure duplicate/near-duplicate rate. If duplication
  is high (e.g. >10%), dedup before the next training run or the
  bootstrapping transition (Q13).
- **Effort/impact:** cheap (a hashing pass over existing data); worth
  running once before investing further compute in larger datasets.

### 9. Human-game-derived positions blended into NNUE training
- **Why it matters:** the policy net already uses 19.9M human Lichess
  positions; the NNUE eval net currently doesn't touch this data at all.
  Blending in human-game positions (labeled the same way as self-play
  positions) could diversify the NNUE's training distribution away from
  the engine's own self-play blind spots, mirroring #7's diversity concern
  from a different angle.
- **How to test:** label a subset of the existing Lichess policy-net corpus
  with the same HCE/NNUE labeling pipeline used for self-play data, mix a
  modest fraction (e.g. 10-20%) into the next NNUE training run, SPRT
  against a self-play-only baseline of the same total size.
- **Effort/impact:** moderate (requires running the labeling pipeline on a
  new data source, not just new self-play games); a genuinely novel data
  source rather than a rehash of existing self-play, so worth prioritizing
  above pure self-play scaling.

### 10. Position sampling from actual UCI games played
- **Why it matters:** once the engine has enough logged real games (vs
  humans or other engines via UCI), those positions represent genuinely
  different distributions than self-play — real opponents make real
  mistakes and take real lines self-play won't reach — but there's
  currently no pipeline to capture and reuse them for training.
- **How to test:** N/A yet — this is a "wait for enough data" item. Once
  a meaningful volume of logged UCI games exists, treat it the same as #9:
  label and blend a modest fraction into training, SPRT-compare.
- **Effort/impact:** low effort to build (log storage + labeling reuse),
  but blocked on having actual usage volume; revisit once the engine sees
  real play.

### 11. Draw-heavy dataset rebalancing
- **Why it matters:** self-play between similarly-strong engine copies
  tends to produce more draws than games against varied human opposition;
  if the training set's win/draw/loss ratio doesn't reflect the target
  distribution the WDL loss assumes, the network may be systematically
  miscalibrated on decisive-result probability estimation.
- **How to test:** compute the actual W/D/L ratio across training shards,
  compare against Lichess's rating-band-appropriate ratios (available from
  their public data). If skewed, either resample or add a rebalancing
  weight to the loss.
- **Effort/impact:** cheap analysis, moderate fix if needed (resampling
  requires either discarding data or reweighting the loss); worth checking
  before the bootstrapping transition (Q13) to avoid baking in a skew that
  then propagates across generations.

### 12. Held-out cross-validation set separate from training shards
- **Why it matters:** the doc's Appendix F evaluation metrics (training/
  validation loss curves) assume a validation split exists, but it's not
  clear from the current pipeline description whether validation data is
  truly held out or just a slice of the same self-play distribution — a
  held-out set drawn from a *different* source (e.g. #9's human positions)
  would catch overfitting the current setup might miss.
- **How to test:** if not already separate, carve out a fixed validation
  set from a distinct data source (human positions, or an early frozen
  self-play batch) and never retrain on it across future NNUE generations;
  track validation loss on this fixed set across every bootstrapping
  generation (ties directly to reinforcement item #87 below).
- **Effort/impact:** cheap if done now, more valuable the earlier it's
  established (later retrofitting can't recover a "clean" validation set
  if all existing data has already been trained on at some point).

## Search, beyond SEE / correction-continuation history / time allocation

### 13. Aspiration windows
- **Why it matters:** a standard, usually-free-Elo technique (search
  around the previous iteration's score with a narrow window, widening on
  fail-high/fail-low) that reduces node counts without changing search
  logic; the doc's search-feature list (PVS, quiescence, null-move, LMR,
  killers/history, TT) doesn't mention it, suggesting it may be missing.
- **How to test:** check if already implemented; if not, implement and
  SPRT at standard bounds ([0,5] Elo, 400-800 games).
- **Effort/impact:** small implementation effort, well-established
  technique with predictable modest gains (typically single-digit to
  low-double-digit Elo) — a good Tier 1 candidate alongside SEE.

### 14. Razoring
- **Why it matters:** a shallow-depth pruning technique (drop to
  quiescence search early if the static eval is far below alpha) that
  complements the existing futility-pruning-shaped hole in the search
  feature list; standard in most competitive engines.
- **How to test:** implement, SPRT against baseline; test at the same time
  as futility pruning (#15) since they interact (both are static-eval-based
  early pruning decisions at different depths).
- **Effort/impact:** small effort, modest but reliable expected gain.

### 15. Futility pruning
- **Why it matters:** listed on the doc's own future-improvements
  discussion implicitly (it's a standard companion to null-move and LMR)
  but not confirmed present in Unchessed's current search feature list —
  worth explicitly auditing.
- **How to test:** audit codebase for existing implementation; if absent,
  implement with standard margins, SPRT, then re-tune margins specifically
  for NNUE eval (the doc's Appendix J notes NNUE's higher per-node cost
  changes the cost/benefit of pruning heuristics vs. HCE).
- **Effort/impact:** small effort; the doc's own appendix (Section J) flags
  that pruning heuristics have a "double benefit" under NNUE eval
  specifically (reduce both node count and eval cost), making this
  higher-value now than it would've been under the old HCE.

### 16. ProbCut
- **Why it matters:** a shallow-search-based pruning technique that uses a
  reduced-depth search to predict whether a full-depth search would fail
  high/low; more sophisticated than razoring/futility, typically added
  after those are in place and tuned.
- **How to test:** implement after razoring/futility (#14/#15) land and are
  validated, since ProbCut's calibration depends on having stable shallow-
  search eval behavior to predict from; SPRT independently.
- **Effort/impact:** moderate effort, sequenced after the simpler pruning
  techniques; don't front-load this before #14/#15.

### 17. Singular extensions
- **Why it matters:** extends search depth for moves that appear uniquely
  best at a node (avoiding pruning away the one critical line in tactical
  positions); a well-established technique but adds nodes rather than
  cutting them, so it needs careful tuning to not offset the gains from
  pruning-heavy techniques (#13-16).
- **How to test:** implement, SPRT; specifically test interaction with the
  pruning techniques above since the doc's Appendix J explicitly notes
  extensions have a "double cost" under NNUE (more nodes AND each node is
  more expensive to evaluate than under HCE).
- **Effort/impact:** moderate effort; sequence this after the pruning
  techniques are locked in, since its net value depends on how aggressively
  the tree is already being cut elsewhere.

### 18. Multi-cut pruning
- **Why it matters:** prunes subtrees unlikely to contain the best move
  based on multiple reduced-depth searches quickly reaching a cutoff;
  complements but is distinct from ProbCut and singular extensions.
- **How to test:** implement after the core pruning suite (#13-16) is
  stable; SPRT independently, then combined with ProbCut (#16) to check
  for redundant pruning logic (both target similar situations).
- **Effort/impact:** moderate effort; lower priority than the more
  foundational pruning techniques since its marginal contribution once
  ProbCut exists is less certain.

### 19. Internal iterative deepening/reduction
- **Why it matters:** when no TT move is available for move ordering at a
  node, IID does a shallow search first to seed a good ordering guess;
  useful specifically at nodes where the TT lookup misses, which becomes
  more common as Hash pressure increases in longer games.
- **How to test:** implement, SPRT with a focus on longer time controls
  (where TT pressure and misses are more relevant) not just the standard
  blitz SPRT bounds.
- **Effort/impact:** small effort; expected impact modest and most visible
  at longer TCs, so test conditions matter more here than for most other
  search items.

### 20. Late move pruning (distinct from LMR)
- **Why it matters:** LMR *reduces* search depth for late move-ordered
  moves; late move pruning *skips* them entirely past a certain move count
  at shallow remaining depth. Different technique, often implemented
  alongside LMR but not the same thing — worth confirming Unchessed has
  the pruning variant, not just the reduction variant.
- **How to test:** audit for existing implementation; if absent, implement
  with standard move-count thresholds scaled by depth, SPRT.
- **Effort/impact:** small effort, modest expected gain, standard technique.

### 21. Lazy SMP multi-threading
- **Why it matters:** the single biggest missing standard feature relative
  to competitive engines — multi-threaded search via shared TT with
  independent search threads (no explicit work distribution) is how nearly
  every modern engine scales with available cores; currently absent per
  the README's roadmap.
- **How to test:** implement, verify no race conditions in TT/accumulator
  access (ties to reinforcement item #64 in the testing section), measure
  Elo gain at various thread counts via SPRT at fixed total node budget vs
  fixed wall-clock time (both matter for different use cases).
- **Effort/impact:** large effort (genuine multi-threading correctness work
  in Rust, though Rust's ownership model helps catch data races at compile
  time per the doc's Appendix H observation), high expected impact —
  probably the single largest engine-strength item left on the entire
  backlog for anyone with multi-core hardware.

## Movegen / core engine

### 22. Deeper perft suite
- **Why it matters:** current perft verification is startpos + Kiwipete
  only; the doc's own Appendix G explicitly recommends "additional perft
  test positions that specifically exercise [en passant, castling, multiple
  promotion types]" since these move types aren't well-exercised by the two
  standard positions.
- **How to test:** add the standard supplementary perft test positions
  (e.g. positions 3-6 from the well-known chess programming perft test
  suite, which specifically stress en passant, castling both directions,
  and promotions) to the existing test suite.
- **Effort/impact:** trivial (well-known reference positions and expected
  node counts already exist publicly); should be done regardless of
  priority elsewhere since it's nearly free and directly de-risks #91
  (the fuzz+perft reinforcement item).

### 23. Chess960/Fischer Random support
- **Why it matters:** a meaningfully large scope addition (castling rules
  alone become significantly more complex with variable starting
  positions), not clearly justified by current project goals, but worth an
  explicit scoping decision rather than leaving it as an ambient unanswered
  question.
- **How to test:** N/A — this is a scope decision, not an experiment.
  Worth writing a short cost/benefit note (how much of the movegen/castling
  code would need to change) before deciding whether it's in scope at all.
- **Effort/impact:** large effort if pursued; low urgency — should probably
  stay explicitly out of scope unless there's a specific reason (e.g. a
  target platform that requires it) to take it on.

### 24. Magic bitboard vs current movegen performance comparison
- **Why it matters:** if the current movegen doesn't already use magic
  bitboards for sliding piece attacks, this is a well-known, large
  performance win (10x+ over naive ray-based attack generation in some
  implementations); worth confirming what's actually implemented.
- **How to test:** audit the current movegen implementation; if not magic
  bitboards, benchmark nodes/sec before and after switching on a fixed
  search depth/position set.
- **Effort/impact:** if already implemented, zero further action needed
  (just confirm); if not, moderate implementation effort for a
  well-understood, high-confidence performance win.

### 25. SIMD-accelerated movegen inner loops
- **Why it matters:** the doc covers SIMD extensively for NNUE inference
  but never discusses movegen itself; if movegen is a measurable fraction
  of total per-node cost (likely small relative to NNUE eval, but worth
  confirming rather than assuming), SIMD could help there too.
- **How to test:** profile the search to see what fraction of per-node time
  is movegen vs eval vs other search overhead; only pursue SIMD movegen
  work if movegen is a non-trivial fraction (if NNUE eval dominates per the
  doc's own ~50 Mnps claim, this is likely low-value).
- **Effort/impact:** profiling is cheap; the SIMD work itself is only
  worth doing if profiling shows it matters — likely low priority given
  NNUE eval probably dominates node cost.

### 26. Zobrist hashing collision rate audit
- **Why it matters:** at the current TT size (default 128MB per the
  README), collision rate directly affects search quality; the doc's
  Appendix G discusses Zobrist hashing in the context of the TT but doesn't
  quantify Unchessed's actual collision rate at realistic hash sizes.
- **How to test:** instrument the TT to log collision events (same key,
  different position) during a batch of test games at default and larger
  Hash settings, compute observed collision rate, compare against the
  theoretical birthday-bound expectation for the hash width used.
- **Effort/impact:** small effort (logging + analysis); mostly a sanity
  check unless something is actually wrong, in which case it's a
  correctness bug worth fixing immediately.

### 27. Draw detection edge cases
- **Why it matters:** the doc's own move-generation appendix (G) flags that
  50-move-counter and repetition detection interactions with the TT and
  null-move pruning are exactly the kind of edge case that "may not be
  exercised by the standard start position and Kiwipete position perft
  tests" — a class of bug (incorrect draw claims/misses) that wouldn't show
  up in perft at all since perft doesn't test draw logic.
- **How to test:** build a targeted test set of positions specifically
  designed to stress repetition-across-null-move and 50-move-counter
  interaction with TT hits (e.g. positions with a forced repetition
  available only in a search line that goes through a stored TT entry),
  verify draw detection triggers correctly.
- **Effort/impact:** moderate effort to construct good test positions; high
  value since a draw-detection bug directly costs or gains games in a way
  that's hard to diagnose from SPRT results alone (it just looks like
  noise).

### 28. Static null-move / reverse futility pruning under check extensions
- **Why it matters:** the doc explicitly documents that null-move pruning
  "is disabled in positions where it is known to be unreliable (zugzwang
  positions, positions in check, endgame positions with few pieces)" —
  worth confirming Unchessed's implementation correctly disables it in all
  these cases, not just the "in check" case which is the easiest to get
  right.
- **How to test:** construct a small test set of known zugzwang positions
  and sparse-endgame positions, verify the engine's null-move logic
  actually disables pruning in each case (not just doesn't crash — actually
  produces correct evaluations where null-move would otherwise mislead).
- **Effort/impact:** small effort; correctness-focused rather than
  Elo-focused, but zugzwang mishandling is a classic source of silent
  endgame weakness.

## Opening book

### 29. Book-learning from own game results
- **Why it matters:** the current book is static (hand-curated lines with
  fixed popularity weights); reweighting lines by the engine's own observed
  win rate in them would let the book improve from experience the same way
  the NNUE improves from bootstrapped training data.
- **How to test:** log game outcomes keyed by which book line was played,
  after enough games (hundreds+ per line) compute win-rate deltas from the
  static weights, experiment with reweighting the top few lines by a small
  amount, SPRT the reweighted book against the static one.
- **Effort/impact:** moderate effort (needs outcome logging infra first);
  needs substantial game volume to get statistically meaningful per-line
  win rates, so this is a slow-burn item, not a quick win.

### 30. Wider troll-tier coverage
- **Why it matters:** currently a small hand-picked set (Bongcloud,
  Scholar's mate attempts, Stafford, Fried Liver per the README); more
  variety would make the troll persona feature less repetitive for repeat
  opponents, a pure user-experience improvement rather than a strength one.
- **How to test:** N/A as an experiment — this is content curation. Survey
  known "meme opening" lines from chess community sources, verify each is
  objectively bad-but-playable (not a trap that backfires), add to the
  troll tier with the same bail-out-guard-eligible tagging as existing
  lines.
- **Effort/impact:** small effort, pure content work; low priority but
  cheap.

### 31. Book move selection weighted by persona
- **Why it matters:** currently the troll tier is binary-gated by Elo
  estimate and persona (per the README's MATCH-mode-only troll gating), but
  CLINCH mode (drawish-endgame trap-seeking) could similarly benefit from
  preferring *sharp* mainline book choices over quiet ones, which the
  current book selection doesn't distinguish.
- **How to test:** tag book lines by sharpness (already partially implied
  by ECO categorization — open games tend sharper than closed), bias
  weighted random selection toward sharp lines specifically when the
  persona is CLINCH, SPRT/playtest for whether this measurably increases
  CLINCH's win-conversion rate in drawish positions reached from these
  openings.
- **Effort/impact:** moderate effort (needs a sharpness tagging pass over
  the ~45 book lines, small set so tractable); speculative payoff, worth
  a small experiment rather than large investment.

### 32. Automatic book generation from a masters PGN corpus
- **Why it matters:** the current ~45 lines are hand-curated; a
  popularity-weighted book built from filtering a large masters PGN
  database (parallel to how the policy net uses Lichess data) would scale
  coverage far beyond what's feasible by hand, and could feed directly into
  #29's win-rate-learning idea as a much larger starting candidate pool.
- **How to test:** source a masters-game PGN corpus, build a frequency-
  weighted opening tree filtered to a reasonable depth (the doc's Appendix
  C references 2-4 move deep books as sufficient for a small project's
  testing), compare book coverage/depth against the current embedded book.
- **Effort/impact:** moderate effort (data sourcing + tree-building script,
  no engine code changes); high leverage since it's a one-time investment
  that dramatically increases book quality and directly enables #29 and #31.

### 33. Book depth tuning per persona
- **Why it matters:** `BookDepth` is currently a single global setting
  (default 16 per the README); DEFEND mode (playing worse, wants to
  extend book longer to delay complexity) and CLINCH mode (wants sharp
  positions sooner) plausibly benefit from different depths, but this is
  currently unexplored.
- **How to test:** once persona-aware book weighting (#31) exists,
  experiment with per-persona `BookDepth` overrides, playtest/SPRT against
  the current single-depth baseline specifically in games that reach each
  persona.
- **Effort/impact:** small implementation effort (a persona-keyed lookup
  instead of a single constant), moderate testing effort (needs enough
  games in each persona to be meaningful); sequence after #31.

### 34. Polyglot/embedded book merge and priority logic robustness
- **Why it matters:** the README describes the external Polyglot book as
  taking priority with fallback to the embedded book for positions not
  covered — worth confirming this fallback logic is bug-free at the
  boundary (positions covered by neither, or covered by both with
  conflicting move preferences).
- **How to test:** construct test cases at the boundary (a position in the
  embedded book but not the external one, vice versa, and a position in
  both with different top moves), verify the engine's selection matches
  the documented priority rule in each case.
- **Effort/impact:** small effort, correctness-focused; a plausible source
  of subtle inconsistent book behavior if untested.

### 35. ECO-name coverage completeness audit
- **Why it matters:** pure documentation/UX quality — verifying the ~45
  embedded lines' ECO codes and names are accurate and that logging
  (`info string` output referencing opening names) doesn't silently
  mislabel a line.
- **How to test:** cross-reference each embedded line's ECO code/name
  against a reference ECO database.
- **Effort/impact:** trivial, low priority, but cheap to do in the same
  pass as #30 (troll-tier expansion) since both involve reviewing the same
  book-line data structure.

## Adapter / persona system

### 36. Faster opponent-Elo convergence via priors
- **Why it matters:** the doc's Appendix M gives the exact convergence math
  (posterior std. dev. ~70 Elo after 8 moves, ~57 after 12, using a
  normal(1500, 500) prior) — a tighter or better-informed prior (e.g.
  seeded from time-per-move variance in the first few moves, or from
  platform/format context) could shrink this uncertainty window faster
  without changing the underlying Bayesian model.
- **How to test:** using the doc's own formula (posterior variance =
  250000/(1+N×6.25)), compute what a narrower starting prior (e.g. std dev
  300 instead of 500, if justified by some prior signal) would do to
  convergence speed, then test candidate priors (e.g. weighted by declared
  UCI_Opponent rating even for unrecognized names, or by game-format
  metadata) against the current flat prior in logged games.
- **Effort/impact:** small implementation effort (swap the prior
  parameters), needs a reasonable amount of game data to validate
  convergence-speed improvement empirically rather than just by the
  formula.

### 37. Policy-net-based engine-tell detection
- **Why it matters:** the current engine-tell detector (per the README)
  uses move-timing plus quality; a KL-divergence signal between the human
  policy net's predicted move distribution and the opponent's actual moves
  could catch a different failure mode — an engine-assisted human playing
  slowly (defeating the timing signal) but selecting moves no real human at
  the claimed rating would pick.
- **How to test:** compute KL divergence between the relevant rating
  bucket's policy net output and the opponent's actual move choices across
  a game, track this alongside the existing timing-based suspicion score,
  see if it flags cases the timing signal misses (using known engine-vs-
  engine test games as ground truth for "should be flagged").
  This item directly relates to reinforcement item #93 for the specific
  time-pressure interaction case.
- **Effort/impact:** moderate effort (policy net inference is already
  built, just needs the divergence computation and threshold tuning); high
  value as an anti-cheat improvement with no established prior art in the
  chess-engine-persona space per the doc's own note that Unchessed's
  persona system is unique.

### 38. Hysteresis threshold tuning via SPRT sweep
- **Why it matters:** the doc's Appendix M gives example thresholds
  (MATCH→PUNISH at >100cp loss, PUNISH→MATCH at Elo estimate dropping 200
  below the current threshold) but these read as illustrative defaults, not
  tuned values — there's no evidence any SPRT-style testing has actually
  optimized them.
- **How to test:** treat each persona transition's enter/exit threshold as
  a tunable parameter, run SPRT comparisons of a few candidate values
  against the current defaults (e.g. ±25% on each threshold), measuring
  either win rate against a fixed opponent-strength ladder or a
  human-plausibility proxy metric if available.
- **Effort/impact:** moderate effort (needs a testing harness that can
  simulate/replay opponent-strength scenarios reliably, which may not
  exist yet); high potential value since these thresholds directly shape
  the core user-facing behavior of the adaptive engine.

### 39. Contempt value sweep
- **Why it matters:** `Contempt=25` is a single hardcoded default (per the
  README's UCI options table) with no documented tuning process behind it.
- **How to test:** SPRT-sweep a small range of contempt values (e.g. 10,
  25, 40, 60) specifically in CLINCH-mode-triggering scenarios (drawish
  endgames vs weaker opposition), since that's where contempt is "wired
  into the search itself" per the README.
- **Effort/impact:** small effort (single-parameter sweep, existing SPRT
  infra); direct, well-scoped experiment.

### 40. Troll bail-out eval-gate threshold sweep
- **Why it matters:** the bail-out guard that exits troll lines when the
  position "has become tactically dangerous" uses an eval threshold that,
  like the persona hysteresis thresholds (#38), appears to be a default
  rather than a tuned value.
- **How to test:** sweep the bail-out threshold across a few values, measure
  both false-positive rate (bailing out of troll lines that were actually
  fine) and false-negative rate (continuing troll lines that were actually
  refuted) using logged troll-tier games as the test corpus.
- **Effort/impact:** small effort, needs a decent sample of troll-tier games
  to measure against; low urgency but cheap once enough troll-tier game
  data exists.

### 41. CLINCH "narrow-safe-path" metric validation
- **Why it matters:** the README describes CLINCH as using a "narrow-safe-
  path metric" to select venomous lines, but there's no stated validation
  that this metric actually correlates with real winning chances in
  practice versus just looking sharp on paper.
- **How to test:** for a sample of CLINCH-mode games, compare the
  narrow-safe-path metric's assessment of "venomous" positions against
  actual game outcomes and post-hoc engine analysis of whether the chosen
  lines were objectively good practical tries.
- **Effort/impact:** moderate effort (needs game sampling + analysis
  tooling); validates a core claim about the persona system's most
  distinctive behavior rather than just assuming the metric works as
  designed.

### 42. DEFEND persona resignation/save-rate tuning
- **Why it matters:** DEFEND mode "digs in when worse" per the README, but
  there's no stated policy on when (if ever) the engine should resign a
  clearly lost position versus playing on, and no measurement of how often
  DEFEND actually saves games it should save (or fights on too long/short
  in ones it shouldn't).
- **How to test:** log DEFEND-mode games' final evaluations at the point
  DEFEND activates vs. eventual outcome, compute the actual save rate
  (games recovered from a "worse" evaluation) as a baseline, then
  experiment with adjusting DEFEND's threshold/behavior to see if save rate
  improves without regressing overall Elo.
- **Effort/impact:** moderate effort; needs a reasonable sample of DEFEND-
  triggering games to get a meaningful baseline first.

### 43. Sandbagger detection refinement
- **Why it matters:** the README describes erratic play (brilliancies mixed
  with blunders) as widening the model's uncertainty rather than narrowing
  it — a reasonable design, but the specific widening rate is presumably a
  hardcoded constant with the same "default, not tuned" issue as #38-40.
- **How to test:** using the doc's Appendix M Bayesian framework, treat the
  uncertainty-widening rate for erratic play as a tunable parameter, test
  candidate values against synthetic "sandbagger" game scripts (alternating
  strong/weak moves at known patterns) to see which widening rate correctly
  avoids over- or under-reacting.
- **Effort/impact:** moderate effort (needs synthetic test game
  construction since real sandbagger examples may be rare); a real gap
  since a poorly-tuned widening rate could make the adaptive engine either
  too gullible or too paranoid against legitimate erratic human play.

### 44. Persona transition logging/analysis tooling
- **Why it matters:** the README confirms every persona transition is
  logged (`persona MATCH -> PUNISH (eval 990 cp, opponent ~1011)`), but
  there's no mention of tooling to aggregate these logs across many games
  into structured stats — without that, none of #38-43's tuning experiments
  have good instrumentation to measure against.
- **How to test:** N/A as an experiment — this is infrastructure that
  unblocks several other items. Build a script that parses `info string`
  persona-transition logs across a batch of games into a structured table
  (transition type, trigger eval/Elo values, game outcome).
- **Effort/impact:** small effort, high leverage — this is a prerequisite
  for #38, #39, #40, #42, and #43 all being properly measurable rather than
  guessed at.

### 45. `UCI_Opponent` seeding table completeness
- **Why it matters:** the README states known engines (Stockfish, Leela,
  Komodo, RubiChess) seed the Elo estimate at their real strength — worth
  auditing exactly which engine names/strings are recognized versus falling
  through to the neutral default, since GUI-reported opponent names aren't
  perfectly standardized.
- **How to test:** enumerate the actual recognized-name list in the code,
  compare against the set of UCI engine identifier strings commonly sent by
  popular GUIs (En Croissant, Arena, CuteChess) for well-known engines,
  identify gaps.
- **Effort/impact:** small effort (a lookup-table audit and expansion);
  directly improves correctness of the "hard-locked off against strong
  engines" trolling safeguard the README describes.

### 46. Multi-persona blending near threshold boundaries
- **Why it matters:** persona transitions are currently discrete
  (hysteresis-gated switches), which the README frames as intentional
  ("commits to a plan instead of flapping") — but this means behavior can
  change abruptly right at a threshold crossing, which might read as less
  natural than a gradual blend, at some cost to the "commitment" benefit
  hysteresis provides.
- **How to test:** this is genuinely a design tradeoff, not a clear win —
  worth a focused playtest comparing discrete switching (current) against
  a small blending window (e.g. interpolating move selection weight between
  two personas for a few plies around a transition) for perceived
  naturalness, without assuming blending is strictly better.
- **Effort/impact:** moderate effort, uncertain payoff, genuinely
  exploratory; lower priority than the tuning items (#38-43) since those
  have clearer success criteria.

## Policy net, beyond time-pressure and blunder-rate calibration

### 47. Finer/more rating buckets
- **Why it matters:** currently 4 buckets (<1300, 1300-1599, 1600-1899,
  1900+) blended linearly between the two nearest; the accuracy table shows
  meaningfully different behavior even within a single bucket's range
  (e.g. a 1310 and a 1590 player both fall in the same "1300-1599" bucket
  despite likely playing quite differently), which finer buckets could
  address.
- **How to test:** train an additional experimental bucket scheme (e.g. 6-8
  buckets instead of 4) on the same 19.9M-position dataset, compare top-1
  accuracy per bucket against the current 4-bucket scheme at matched rating
  points (interpolated for the current scheme).
- **Effort/impact:** moderate effort (retraining with a new bucket scheme,
  reusing existing data); directly informed by #49's architecture choice
  since a larger network might be needed to support more buckets without
  each one being undertrained.

### 48. Non-linear blending between adjacent buckets
- **Why it matters:** the README states the engine "blends the two buckets
  nearest its current target rating" without specifying the blend function
  — if it's linear interpolation, that's an assumption about how move
  preference actually varies with rating that hasn't been validated against
  real data.
- **How to test:** using the existing bucket networks, compare linear
  blending's move predictions at intermediate ratings (e.g. target 1450,
  blending the <1300 and 1300-1599 buckets) against those buckets' own
  independently-observed accuracy curves to see if a non-linear blend
  (e.g. weighted by each bucket's local accuracy) would predict real
  intermediate-rating play better.
- **Effort/impact:** small effort (no retraining needed, just changing the
  blend function at inference time and comparing against held-out
  intermediate-rating human game data); directly testable without new
  training runs, so a cheap experiment.

### 49. Conv/resnet policy architecture
- **Why it matters:** the doc discusses Chessformer/transformer approaches
  at length in the context of NNUE evaluation and lc0-style search, but
  never applies that discussion specifically to Unchessed's own Maia-style
  policy net, which currently uses (per context) a simpler architecture
  than the convolutional or transformer options surveyed.
- **How to test:** train a small conv or resnet-style policy net on the
  same 19.9M-position dataset, compare top-1 accuracy against the current
  architecture at matched parameter budgets, to see whether architecture
  or data volume is the current bottleneck (a useful diagnostic regardless
  of which one wins).
- **Effort/impact:** moderate-to-large effort (new architecture,
  retraining); high potential value since Maia's own published history
  (per the doc's Appendix K survey) shows meaningful accuracy gains from
  architecture improvements over the years (Maia 1 → Maia 2 → Chessformer),
  so there's real precedent for this mattering.

### 50. En-passant accuracy improvement at the <1300 bucket
- **Why it matters:** 18.8% en-passant accuracy at <1300 vs 67.4% at 1900+
  is the single largest per-feature accuracy gap in the entire policy net
  — the doc itself notes this is *real* human behavior (weaker players
  genuinely miss/decline en passant), which means the low number might
  already be roughly correct rather than a network deficiency, but this
  hasn't been separately verified.
- **How to test:** compute the actual en-passant-availability decline rate
  in real <1300 Lichess games (ground truth) and compare directly against
  the network's 18.8% prediction accuracy — if the ground-truth decline
  rate is also very low, the network may already be near-optimal and this
  isn't a bug to fix; if there's a meaningful gap, it's a genuine
  undertraining issue (en passant positions being rare in the training
  data, oversampling already applied per the README but possibly
  insufficient).
- **Effort/impact:** small effort (a data analysis task before any model
  changes); this should be done *before* #47 or #49 are used to try to fix
  this specific number, since the "problem" might not actually be a
  problem.

### 51. Larger/more diverse Lichess training sample
- **Why it matters:** 19.9M positions from 781K games is the current
  dataset size; Maia's own comparably-scoped published work and the doc's
  Otter/1e4.ai citations use datasets one to three orders of magnitude
  larger (Otter: 6.1 billion positions from 117 million games) — worth
  understanding whether Unchessed's current scale is actually a bottleneck.
- **How to test:** plot top-1 accuracy vs. training set size using
  subsampled fractions of the existing dataset (e.g. 25%/50%/75%/100%) to
  estimate whether the current accuracy curve has already plateaued at
  19.9M positions or is still climbing — this tells you whether more data
  would help before investing in acquiring it.
- **Effort/impact:** cheap (reuses existing data and training pipeline,
  just at different sample sizes); a standard and worthwhile diagnostic
  before deciding whether to invest in a much larger data-gathering effort.

### 52. Policy net calibration against game-speed splits
- **Why it matters:** the current training data is "non-bullet rated
  games" per the README — bullet/blitz/classical players plausibly exhibit
  different move-selection patterns at the same nominal rating (time
  pressure changes decision quality, per the same logic underlying Q18's
  time-pressure feature), so a single non-bullet-trained net may not
  generalize well if the adaptive engine is used in fast time controls.
- **How to test:** evaluate the current policy net's accuracy separately
  against held-out blitz and rapid/classical game samples at matched
  ratings, check for a meaningful accuracy gap between the two.
- **Effort/impact:** small effort (analysis only, no retraining needed for
  the initial diagnostic); if a gap is found, this directly informs
  whether Q18's time-pressure feature needs game-speed-specific training
  data, not just clock-fraction-remaining as a scalar.

### 53. Joint policy+value net
- **Why it matters:** the current setup trains a policy net (move
  prediction) and uses NNUE separately for evaluation; a joint model
  predicting both human move preference and game outcome (as AlphaZero-
  style networks and Maia's own architecture discussions touch on) could
  let the two objectives inform each other during training, though this is
  a bigger architectural departure than most other policy-net items here.
- **How to test:** this is a larger research question best scoped as "is it
  worth trying" rather than jumping straight to implementation — start with
  a literature check on whether joint training has shown benefits in the
  Maia lineage specifically (the doc's Appendix K survey doesn't mention
  this being tried), then a small-scale prototype if it looks promising.
- **Effort/impact:** large effort, speculative payoff, genuinely Tier 3
  material; don't prioritize ahead of the cheaper, better-scoped items
  above.

### 54. Policy net move legality/consistency checks under check-evasion
- **Why it matters:** positions with forced or heavily constrained legal
  moves (in check, few escapes) are a different regime than open positions
  with many reasonable choices — worth confirming the policy net's
  predictions remain sensible (i.e. concentrated on actually-legal moves,
  not wasting probability mass on illegal ones) in this narrower-choice
  regime specifically.
- **How to test:** evaluate policy net output distributions specifically on
  a filtered set of in-check positions from the validation data, check what
  fraction of predicted probability mass falls on legal vs illegal moves,
  and whether accuracy in this subset differs meaningfully from the overall
  numbers.
- **Effort/impact:** small effort (analysis only using existing validation
  infrastructure); a correctness/robustness check more than a strength
  improvement.

## Reviewer / analysis tool (entirely unaddressed by the research doc)

### 55. Move classification scheme
- **Why it matters:** the Reviewer tool doesn't exist yet per the README's
  roadmap; before building it, the classification scheme
  (brilliant/good/inaccuracy/mistake/blunder) needs a defined eval-delta-
  to-label mapping, and the doc's own research (Q19 discussion) notes that
  Lichess-style approaches use win-probability deltas, not raw centipawns.
- **How to test:** N/A as an experiment yet — this is a design decision.
  Draft the win-probability-based thresholds (reusing the same WDL-to-
  probability conversion already used in NNUE training, per the doc's
  Appendix I scaling discussion) rather than inventing a separate raw-cp
  scheme, so the Reviewer's notion of "how bad was this move" is consistent
  with the engine's own internal win-probability model.
- **Effort/impact:** moderate design effort, no engine changes yet; this is
  the foundational decision #56-61 all build on.

### 56. Accuracy % formula
- **Why it matters:** directly follows from #55 — once move classification
  thresholds exist, the game-level accuracy % needs its own formula, and
  the doc's own research confirms Lichess's approach (win-probability-based,
  not cp-loss-based) is the standard reference point worth adapting rather
  than inventing from scratch.
- **How to test:** implement the win-probability-based accuracy formula
  once #55's thresholds exist, validate against a small set of manually-
  reviewed games (games where a human expert's intuitive sense of "how
  accurately was this played" can be sanity-checked against the formula's
  output).
- **Effort/impact:** small effort once #55 is settled; needs a manual
  validation pass to catch a formula that produces technically-correct but
  intuitively-wrong numbers.

### 57. Critical-moment detection
- **Why it matters:** not every position is equally informative for
  classification — flagging genuinely critical moments (where the
  eval swing between the best move and a plausible alternative is large)
  makes the Reviewer's output more useful than uniformly classifying every
  move with equal weight.
- **How to test:** define critical moments as positions where the top two
  candidate moves' eval gap exceeds a threshold, prototype this against a
  handful of real games, check whether flagged moments match human
  intuition about where the game was actually decided.
- **Effort/impact:** small-to-moderate effort, builds directly on #55/#56's
  eval infrastructure; a UX-quality feature more than a correctness one.

### 58. PGN batch review CLI ergonomics/output format design
- **Why it matters:** pure tooling/UX design for the Reviewer's actual
  interface — needs to be usable (readable output format, sensible batch
  processing of multiple games) rather than just technically functional.
- **How to test:** N/A as an experiment — prototype the CLI early (even
  before #55-57 are fully finalized, using placeholder classification
  logic) to get ergonomics feedback before the underlying analysis logic
  is fully built, since interface design often reveals requirements the
  analysis logic needs to support.
- **Effort/impact:** small-to-moderate effort; worth doing in parallel with
  #55-57 rather than strictly after, to avoid designing the analysis logic
  in a vacuum from how it'll actually be consumed.

### 59. Opening-phase-aware leniency in classification
- **Why it matters:** early-game inaccuracies (playing an objectively
  slightly-worse but well-known opening move) shouldn't be classified the
  same as a middlegame blunder of equal eval magnitude — human review tools
  typically apply phase-aware leniency, and this needs an explicit design
  decision for the Reviewer to feel fair rather than needlessly harsh in
  book-adjacent positions.
- **How to test:** once #55's base classification exists, add an opening-
  phase discount (e.g. reduced severity for the first N moves or while
  within known book lines, reusing the engine's own opening book data as
  the "known theory" reference), validate against manually-reviewed games
  for whether it changes classifications in ways that match human
  intuition.
- **Effort/impact:** small effort once #55 exists; directly improves
  perceived Reviewer quality, a common source of user complaints in other
  chess analysis tools when absent.

### 60. Best-alternative-move display
- **Why it matters:** classification alone ("this was a mistake") is less
  useful without showing what the better move was — a basic feature
  expected of any move-quality review tool, not present in early
  classification-only prototyping.
- **How to test:** N/A as an experiment — straightforward feature addition
  once #55's analysis pass computes candidate moves anyway (the best
  alternative is already computed as part of determining classification
  severity, just needs to be surfaced in output).
- **Effort/impact:** small effort, high user-facing value, nearly free once
  #55-56 exist since the underlying computation already happens.

### 61. Game-level summary stats
- **Why it matters:** aggregating per-move classifications into a
  game-level view (accuracy trend, phase breakdown) is what makes the
  Reviewer useful for understanding a whole game rather than move-by-move
  only — standard in comparable tools (Lichess/chess.com analysis).
- **How to test:** N/A as an experiment — implement once #55-56 exist,
  validate the phase-breakdown boundaries (opening/middlegame/endgame
  split) against a reasonable heuristic (e.g. material count or move
  number thresholds already used elsewhere in the codebase for consistency).
- **Effort/impact:** small-to-moderate effort, depends on #55/#56/#59;
  natural capstone feature for the Reviewer's first version.

## Testing / regression discipline, beyond the pre-SPRT smoke test

### 62. Fuzzing the UCI protocol parser
- **Why it matters:** malformed or unexpected UCI input (from a
  misbehaving GUI, a malicious input source, or just an edge case in the
  protocol) could crash the engine or cause undefined behavior; the doc
  doesn't address UCI parser robustness at all, focusing entirely on
  NNUE/search correctness.
- **How to test:** build a fuzzing harness (this overlaps substantially
  with reinforcement item #91's combined fuzz+perft+feature-consistency
  harness — implement as one unified tool rather than a separate one) that
  feeds malformed UCI command sequences and verifies no crashes/panics,
  only graceful rejection of invalid input.
- **Effort/impact:** moderate effort if built standalone, smaller marginal
  effort if folded into #91's combined harness (recommended); a basic
  robustness hygiene item any UCI-exposed binary should have.

### 63. Stress-testing under extreme TT pressure specifically
- **Why it matters:** the doc's Appendix S stress test covers 24-hour
  continuous play for memory leak detection generally, but doesn't
  specifically target TT-pressure edge cases like `Hash=1` (minimal TT,
  forcing constant eviction) or very long single games (where the 50-move
  counter, repetition detection, and TT collision rate all interact under
  sustained pressure differently than in a typical game).
- **How to test:** run dedicated test games at `Hash=1` and at very long
  time controls specifically, watching for TT-eviction-related evaluation
  inconsistencies or performance cliffs distinct from the general memory-
  leak concern the doc's stress test targets.
- **Effort/impact:** small effort (reuses existing stress-test
  infrastructure with different parameters); targets a specific edge case
  the doc's general test doesn't cover.

### 64. Multi-threaded search race condition audit
- **Why it matters:** directly gates #21 (Lazy SMP) — once multi-threading
  is implemented, shared TT and any other shared mutable state need
  rigorous race-condition testing; Rust's ownership model catches many
  classes of data race at compile time (per the doc's Appendix H), but
  logical races (e.g. TT entries being overwritten mid-read in ways that
  produce a technically-safe but logically-corrupt read) can still occur.
- **How to test:** once #21 lands, run thread-sanitizer-equivalent tooling
  if available for Rust, plus differential testing (same position searched
  single-threaded vs multi-threaded, checking for evaluation consistency
  beyond just "no crash").
- **Effort/impact:** moderate effort, mandatory companion to #21 rather
  than optional — don't ship Lazy SMP without this.

### 65. Continuous perft regression as a wired-up CI check
- **Why it matters:** the doc's Appendix S checklist lists perft regression
  as a testing phase, but doesn't confirm it's actually automated in CI
  versus run manually — an automated, always-on check is what actually
  prevents regressions from landing, versus a documented-but-not-enforced
  test that can silently stop being run.
- **How to test:** N/A as an experiment — verify whether perft (and
  ideally the broader test suite) runs automatically on every change; if
  not, wire it into CI.
- **Effort/impact:** small effort if CI infrastructure already exists for
  the project; high value since it converts a "we should remember to
  test this" item into a "the pipeline blocks it automatically" guarantee,
  directly closing the same class of gap that let v3's regression through
  in the first place (a process gap, not just a testing-technique gap).

## Calibration / deployment

### 66. Empirical `UCI_Elo` calibration
- **Why it matters:** the biggest open question in the whole deployment
  category — there's no confirmation anywhere in the README or the IEEE
  doc that setting `UCI_Elo=1200` actually produces play that measures at
  ~1200 in real games, as opposed to just selecting the nearest policy net
  bucket and hoping the mapping is accurate.
- **How to test:** run a rating-ladder test — play the engine at several
  `UCI_Elo` settings against known-strength opponents (reference engines
  set to specific Elo levels, or rated bots on a platform like Lichess),
  measure the engine's actual performance rating at each setting, compare
  against the target.
- **Effort/impact:** moderate effort (needs access to calibrated opponents
  and enough games per setting for a meaningful sample); high value since
  this is a core advertised feature of the adaptive engine that's currently
  unverified.

### 67. `UCI_LimitStrength` behavior consistency across personas
- **Why it matters:** the README states `UCI_LimitStrength` works
  "independently" of the adaptive system, constraining maximum strength
  regardless of persona — worth confirming this holds correctly in all
  four personas, not just the common MATCH-mode case (e.g. does a
  strength-limited engine in PUNISH mode, which "snaps to full strength,"
  actually stay capped as intended?).
- **How to test:** test each persona explicitly with `UCI_LimitStrength`
  enabled, verify none exceed the configured cap regardless of what the
  persona system would otherwise do.
- **Effort/impact:** small effort, correctness-focused; a plausible gap
  since PUNISH's explicit design intent ("snaps to full strength") could
  conflict with a strength cap if not implemented carefully.

### 68. Contempt/draw-aversion interaction with `UCI_LimitStrength`
- **Why it matters:** CLINCH mode's contempt-driven draw-aversion is
  wired into the search itself per the README — worth confirming this
  still produces sensible behavior (fighting for wins appropriately, not
  irrationally) when the engine is simultaneously strength-limited, since
  a weakened engine might not have the tactical accuracy to justify the
  same contempt-driven risk-taking a full-strength engine would.
- **How to test:** test CLINCH-mode games specifically with
  `UCI_LimitStrength` enabled at various levels, check for a higher rate
  of contempt-driven sacrifices that don't actually work out (since the
  weaker search might miscalculate the resulting complications).
- **Effort/impact:** small-to-moderate effort; a specific interaction case
  worth checking rather than assuming CLINCH's design generalizes cleanly
  to limited-strength play.

### 69. Book usage rate calibration per target Elo
- **Why it matters:** real players of different strengths leave opening
  theory at different points (weaker players deviate from book earlier,
  often unintentionally); if low-rated personas in MATCH mode follow the
  embedded book too accurately/too long, that's itself a human-plausibility
  gap parallel to the policy net's move-selection calibration work (#94).
- **How to test:** compare the engine's actual book-exit-move distribution
  at various target Elo settings against real human book-exit statistics
  from Lichess games at matched ratings.
- **Effort/impact:** small-to-moderate effort (data analysis, then possibly
  a book-depth or move-selection adjustment); ties conceptually to #33
  (persona-specific book depth) and #94 (blunder-rate calibration) — all
  three are instances of the same underlying question: does the engine's
  behavior at a target rating actually match real humans at that rating,
  beyond just move-choice accuracy.

### 70. Engine identification string / UCI option discoverability audit
- **Why it matters:** pure GUI-compatibility hygiene — confirming the
  engine's UCI identification (name, author, option descriptions) is
  correctly discoverable by common GUIs (En Croissant, Arena, CuteChess),
  since the README documents En Croissant integration specifically but
  doesn't confirm broader GUI compatibility.
- **How to test:** test UCI option discovery and identification in at
  least one GUI beyond En Croissant, verify all documented options
  (Hash, MultiPV, Adaptive, UCI_LimitStrength, etc.) appear correctly with
  sensible descriptions.
- **Effort/impact:** small effort, low-but-real value for broader
  usability beyond the one GUI currently confirmed to work well.

## Longer-horizon / exploratory

### 71. Tablebase integration for actual endgame play
- **Why it matters:** distinct from Q10 (using tablebases to correct
  training *labels*) — this is about probing Syzygy tablebases live during
  search for perfect endgame play, which the doc's Appendix Q mentions only
  as a brief future-direction note without analysis.
- **How to test:** implement Syzygy probing for positions with ≤7 pieces
  reached during search, SPRT specifically in test games that reach
  tablebase-covered endgames, measure the improvement in endgame
  conversion rate.
- **Effort/impact:** moderate-to-large effort (tablebase file management,
  probing code, search integration); high value specifically for endgame
  precision, a well-established technique in top engines with clear
  expected benefit.

### 72. Pondering support
- **Why it matters:** thinking during the opponent's time is a standard
  competitive feature listed on the README's roadmap but not yet
  implemented; particularly relevant given the adaptive engine's
  time-management sophistication already described.
- **How to test:** implement standard UCI ponder support (`go ponder`,
  `ponderhit`), verify correctness against the UCI protocol spec, measure
  effective time gained in games against ponder-aware opponents.
- **Effort/impact:** moderate effort, standard well-understood feature;
  primarily a competitive-play feature rather than one that affects the
  adaptive/persona system.

### 73. MultiPV-weighted move selection inside the adapter
- **Why it matters:** MultiPV output already exists per the README's
  search feature list; using it inside the adapter itself — weighting
  among the top-N moves by policy net probability rather than always
  taking the single best move — could make MATCH mode's move selection
  richer than the current (presumably best-move-filtered-by-policy-net)
  approach, without needing any new search infrastructure.
- **How to test:** prototype weighting the top-3-5 MultiPV moves by policy
  net probability instead of just filtering to the single best move,
  compare resulting game style/human-plausibility against the current
  approach in playtest games.
- **Effort/impact:** small-to-moderate effort (mostly adapter logic, reuses
  existing MultiPV and policy net infrastructure); a relatively cheap way
  to potentially improve MATCH mode's naturalness.

### 74. Puzzle-rush/tactics-trainer mode
- **Why it matters:** reuses the Reviewer's classification logic (once
  built, #55-61) to identify tactical positions and present them as
  training puzzles — a natural extension of existing infrastructure into a
  new user-facing feature.
- **How to test:** N/A as an experiment — feature-scoping task, blocked on
  the Reviewer existing first.
- **Effort/impact:** moderate effort, blocked dependency on #55-61; a good
  "phase 2" feature once the Reviewer ships.

### 75. Game-phase-aware NNUE accumulator refresh strategy
- **Why it matters:** the doc's Appendix Q notes phase transitions
  (particularly castling, which changes multiple features at once) have a
  different cost profile for incremental updates than typical moves —
  worth exploring whether a smarter refresh strategy (e.g. proactively
  full-refreshing at known expensive transition points rather than always
  incrementally updating) could reduce worst-case per-move cost.
- **How to test:** profile the cost of incremental updates specifically at
  castling moves and other multi-feature-change transitions, compare
  against a full-refresh baseline at those specific points, decide whether
  a hybrid strategy is worth the added complexity.
- **Effort/impact:** small profiling effort, moderate implementation effort
  if a hybrid strategy proves worthwhile; a micro-optimization, lower
  priority than the larger search/eval items above.

### 76. Cross-engine style transfer study
- **Why it matters:** genuinely exploratory — would training a policy net
  on a specific human's games (rather than a rating-bucket aggregate)
  produce a recognizable "plays like this specific person" persona, versus
  just a generically-humanlike one? No established prior art for this in
  the doc's survey.
- **How to test:** train a small experimental policy net on a single
  prolific player's public game history (assuming sufficient volume and
  appropriate data licensing), qualitatively compare its move preferences
  against that player's known style/openings.
- **Effort/impact:** moderate effort, purely exploratory/novelty value; a
  fun Tier 3 experiment with no clear product requirement behind it
  currently.

### 77. Anti-fingerprinting audit
- **Why it matters:** the flip side of #37's engine-tell detection — does
  Unchessed's own play have detectable statistical tells (e.g. suspiciously
  consistent timing, characteristic eval patterns) that a sophisticated
  opponent could use to identify it's an engine, undermining the
  human-plausibility goal of the whole persona system?
- **How to test:** apply the same kind of analysis Unchessed's own
  engine-tell detector uses (move-timing consistency, quality-vs-position-
  difficulty correlation) to Unchessed's own MATCH-mode games, see if it
  would flag itself.
- **Effort/impact:** small-to-moderate effort (reuses existing detection
  logic against the engine's own output); a clever, cheap self-check with
  real value for the persona system's core credibility.

### 78. Explainability layer
- **Why it matters:** currently `info string` output narrates persona
  switches and book/troll choices per the README, but doesn't explain
  search-level reasoning (why this move over alternatives) in natural
  language — a nice-to-have feature for users wanting to understand the
  engine's thinking beyond raw eval numbers.
- **How to test:** N/A as an experiment — feature design task. Prototype
  simple templated explanations first (e.g. "chose Nf3 over Bc4: 15cp
  better and avoids early queen exposure") using data the search already
  computes, before considering anything more sophisticated.
- **Effort/impact:** moderate effort for a basic version, potentially large
  if pursued to a more sophisticated natural-language level; a genuine
  differentiator but not core to engine strength.

### 79. Long-term self-play league
- **Why it matters:** currently strength progress is tracked via pairwise
  SPRT (new version vs. immediately-previous version); a round-robin league
  across multiple historical NNUE/policy versions would reveal whether
  progress is actually monotonic over the project's history or whether
  some "improvements" quietly regressed against older versions in ways
  pairwise testing wouldn't catch (transitivity isn't guaranteed in chess
  engine strength comparisons).
- **How to test:** periodically run a round-robin tournament including the
  current version and several historical checkpoints (v1, v3, v4, and
  future versions), track relative Elo across the whole set rather than
  just the most recent pairwise comparison.
- **Effort/impact:** moderate ongoing effort (needs to keep historical
  binaries/weights available and periodically re-run); good project
  hygiene for long-term tracking, complements rather than replaces SPRT.

### 80. Neural-network-based move ordering
- **Why it matters:** the doc's Appendix Q mentions this as a future
  direction (a small policy network predicting beta-cutoff probability per
  candidate move) but explicitly leaves it unanalyzed as a formal question
  — worth scoping properly rather than leaving as a passing mention.
- **How to test:** the doc's own appendix identifies the key challenge
  correctly (inference speed — must evaluate all legal moves at every node
  without slowing the search) — a first experiment should measure whether
  a very small network (16-32 hidden units, per the doc's own suggestion)
  can run fast enough to be net-positive before investing in training data
  and integration.
- **Effort/impact:** moderate-to-large effort, genuinely speculative;
  the doc's own framing (comparing it favorably to static heuristics like
  MVV-LVA/history) suggests real potential, but the speed constraint makes
  this higher-risk than most other search items.

### 81. Neural-network-based pruning
- **Why it matters:** same appendix, same "mentioned but not analyzed"
  status as #80 — a small network predicting whether a subtree is worth
  searching at all, positioned by the doc as more sophisticated than static
  pruning heuristics (futility, null-move) because it can consider full
  position complexity rather than a few hand-crafted features.
- **How to test:** same core challenge as #80 (speed vs. accuracy at every
  candidate move); worth prototyping only after #80 establishes whether a
  small-enough network is fast enough to be viable at all in this engine's
  search loop, since the same infrastructure question underlies both.
- **Effort/impact:** large effort, speculative, should be sequenced after
  #80's speed feasibility question is answered rather than pursued in
  parallel.

### 82. Endgame-specific hybrid evaluation
- **Why it matters:** the doc's Appendix Q notes NNUE struggles with
  specific endgame concepts (opposition, triangulation, zugzwang,
  fortress) that occur in a small fraction of training positions, and
  suggests a hybrid (NNUE for general positions + a specialized rule-based
  or neural endgame module) without analyzing it further — this overlaps
  meaningfully with #71 (tablebase integration), which solves the ≤7-piece
  case; this item is specifically about the 8+-piece endgame gap
  tablebases don't cover.
- **How to test:** first establish whether the gap is real for Unchessed
  specifically — construct a test set of known-tricky endgame concepts
  (opposition/zugzwang positions beyond tablebase range) and measure NNUE
  v4's actual accuracy on them before deciding a specialized module is
  needed.
- **Effort/impact:** the diagnostic step is cheap; building an actual
  hybrid module is large effort and should only be pursued if the
  diagnostic confirms a real, sizable gap.

### 83. Multi-agent/opponent-style-conditioned training
- **Why it matters:** the doc's Appendix Q raises this as a speculative
  direction — adapting the *evaluation function itself* to the detected
  opponent style (weighting tactical vs strategic features differently
  against different opponent types), which would be a meaningfully deeper
  integration between the persona system and the eval than the current
  design (where the persona system operates at move-selection/search level,
  not eval level, per the README's own description).
- **How to test:** this is a large, genuinely research-level question, not
  something to prototype casually — worth a scoping note on whether the
  expected benefit (more sophisticated adaptation) justifies the
  architectural complexity (opponent-conditioned training data generation,
  a more complex training pipeline) before any implementation is attempted.
- **Effort/impact:** very large effort, highly speculative; correctly
  placed in Tier 3/exploratory territory, not worth pursuing without a much
  stronger signal that the current move-selection-level persona system is
  hitting a real ceiling.

### 84. Transformer-based evaluation feasibility for alpha-beta search
- **Why it matters:** the doc notes transformers are much slower than NNUE
  for evaluation and that incremental updates (NNUE's core speed advantage)
  don't apply to attention-based architectures — but doesn't explore a
  hybrid where a transformer is used only for *training-time label
  generation* (replacing or supplementing the HCE/NNUE-search-based
  labeling in Q13's bootstrapping plan) while NNUE remains the actual
  inference-time evaluator.
- **How to test:** this only makes sense to explore after Q13's
  NNUE-based bootstrapping (already Tier 1 priority in the doc) is
  established as a baseline — then compare whether a transformer-labeled
  dataset (using one of the existing chess transformer models as an oracle
  labeler, at high compute cost but only for label generation, not
  inference) produces a measurably better-trained NNUE than NNUE-labeled
  bootstrapping alone.
- **Effort/impact:** large effort (requires running a substantial
  transformer model for label generation, likely off-project compute);
  genuinely Tier 3, speculative, but conceptually distinct from "use a
  transformer for eval" (which the doc correctly rules out) — this is
  "use a transformer as a one-time labeling oracle," a meaningfully
  different and cheaper proposition.

## Reinforcement topics — deepening/stress-testing existing items rather than new surface area

These don't open new categories; each one sharpens or cross-checks an item
already listed above (or in the IEEE doc), because the original phrasing was
a one-liner. Each entry below follows the same shape: what's being
reinforced, why the original item was incomplete on its own, a concrete way
to actually test it, and a rough effort/impact read — so these can be picked
up directly rather than needing another round of scoping.

### 85. Ablation interaction re-check at 512-wide
- **Reinforces:** #1 (v4 ablation) + #3 (accumulator width), i.e. Q1/Q3 in
  the IEEE doc.
- **Gap in the original:** the doc's own Q1 pitfalls section admits
  "ablation results may not generalize across network sizes... a feature
  that is beneficial at 256-wide accumulators might become neutral or
  harmful at 512." But Q1 (ablation) is slotted into Tier 1 and Q3 (width)
  into Tier 3 — meaning the roadmap's own ordering guarantees the ablation
  gets treated as permanently settled before the thing that could
  invalidate it is even tried.
- **How to test:** if/when a 512-wide network is ever trained, re-run the
  cheapest slice of the Q1 ablation (v1 + mirroring-only vs v1 + own-king-
  only) at 512-wide and diff against the 256-wide ablation's ranking of
  which change mattered most. You don't need the full 4-config sequential
  ablation again — just enough games to check whether the *ordering* of
  contributions holds, not the exact Elo numbers.
- **Effort/impact:** cheap if bundled into Q3's own testing (same network
  training run, one extra SPRT comparison); high value because it prevents
  quietly shipping a false belief about which architectural change matters.

### 86. Re-tune WDL exponent after QAT, not before
- **Reinforces:** #7 (Q7, WDL exponent grid search).
- **Gap in the original:** Q7's own pitfalls section says the optimal
  exponent "may interact with the evaluation scaling parameter" and that
  scaling has to be recalibrated whenever the output representation
  changes. Quantization (Q5) changes exactly that — int8/int16 rounding
  shifts the effective output distribution — but the doc's Tier 1/Tier 2
  ordering puts the exponent sweep (Tier 1) *before* QAT's accuracy is
  fully validated, meaning the sweep's conclusion could be stale by the
  time QAT ships.
- **How to test:** treat the exponent sweep as two passes, not one — an
  initial cheap pass on the current float32 pipeline (as already planned),
  then a second smaller sweep (just 2.0/2.5/3.0, not the full 1.5-3.5
  range) after QAT lands, to confirm the earlier winner still holds under
  the quantized training loop.
- **Effort/impact:** the second pass is ~3 short training runs, small
  relative to the first sweep; skipping it risks leaving free Elo on the
  table if QAT shifts the optimum.

### 87. Bootstrapping validation against a frozen holdout, not just Syzygy
- **Reinforces:** Q13 (HCE→NNUE label bootstrapping) + #16/Q10 (tablebase
  label correction).
- **Gap in the original:** the doc names "external validation against
  tablebases and reference engines" as the model-collapse mitigation for
  Q13, the highest-risk item in the whole roadmap. But tablebases only
  cover 7-piece endgames — they say nothing about whether the bootstrapped
  v5/v6/v7 generations are quietly degrading in the opening or middlegame,
  which is most of the game.
- **How to test:** freeze a few thousand HCE-labeled positions from the
  *original* v1 dataset (before any bootstrapping) as a permanent,
  never-retrained holdout (ties directly to #12 above). After each
  bootstrapping generation, compute the Pearson correlation between the
  new network's evaluations and this frozen holdout's labels (the doc's
  Appendix F already defines this metric for a different purpose — reuse
  it here). A steadily falling correlation across generations is the
  actual collapse signal endgame tablebases can't provide.
- **Effort/impact:** near-zero marginal cost (the holdout set and
  correlation metric both already exist elsewhere in the doc's own
  proposals) but closes the single biggest blind spot in the
  highest-risk item on the roadmap.

### 88. SEE + correction history combined regression test
- **Reinforces:** Q14 (SEE) + Q15 (correction/continuation history).
- **Gap in the original:** the doc's own "Search Heuristic Interaction
  Effects" appendix (Section J) explicitly describes how SEE changes what
  data correction history sees — SEE causes earlier beta cutoffs, which
  means fewer moves get searched, which means the history heuristic
  updates less often. The doc flags this as "generally a net positive" but
  never actually measures it; it's reasoning from first principles, not a
  result.
- **How to test:** the doc's Tier 1 already plans both SEE and (Tier 2)
  correction history as independent SPRT tests with individually-estimated
  gains (15-30 Elo each). Add one more SPRT leg: SEE+correction-history
  together vs. SEE alone, to check whether the combined gain is
  roughly additive (~30-60 Elo) or whether the interaction actually eats
  into one or the other's contribution.
- **Effort/impact:** one extra SPRT run (~400-800 games) reusing infra
  already built for the two individual tests; prevents a false "both
  landed, gains must have stacked" assumption if they don't.

### 89. Persona-specific correction history tables, tested not just designed
- **Reinforces:** #43 (persona/sandbagger interactions) + Q15's own
  appendix note.
- **Gap in the original:** Section J of the doc's appendix explicitly
  raises the question of whether MATCH-mode corrections (learned while the
  engine is intentionally playing below its ceiling) contaminate PUNISH or
  CLINCH mode's correction history, then answers its own question with "the
  simpler approach (shared correction history) is likely sufficient" —
  which is a guess dressed as a conclusion, not a tested finding.
- **How to test:** once correction history (Q15) is implemented, run a
  direct SPRT comparison — shared correction table across all four
  personas vs. per-persona tables cleared on transition — specifically in
  games that spend meaningful time in more than one persona (i.e. games
  against opponents whose strength triggers at least one MATCH→PUNISH or
  MATCH→CLINCH transition, not games that stay in one persona throughout).
- **Effort/impact:** small implementation delta (a correction-history table
  keyed by persona instead of global), one SPRT run; resolves a design
  question the source doc explicitly left open.

### 90. King-bucket ablation cross-check against the Q1 ablation
- **Reinforces:** Q2 (king-bucket count sensitivity).
- **Gap in the original:** Q2's "Established vs. Open" section notes that a
  16-bucket/512-wide network might land at similar strength to the current
  32-bucket/256-wide one "since both have roughly the same total parameter
  count but distribute capacity differently" — but this is speculation, and
  it directly overlaps with what the Q1 ablation is trying to isolate
  (how much of v4's gain came from the bucket/mirroring scheme itself).
- **How to test:** if a king-bucket sweep (Tier 3, deferred) is ever run,
  don't treat it as a standalone experiment — score it against the same
  v1/v4 SPRT baselines used in the Q1 ablation, so the two studies produce
  directly comparable numbers instead of two disconnected data points that
  can't be reconciled later.
- **Effort/impact:** free if planned in advance (just requires reusing the
  same baseline networks and SPRT bounds); expensive to retrofit if the two
  studies are run years apart with different infra.

### 91. Fuzz + perft interaction check
- **Reinforces:** #22 (deeper perft suite) + #62 (UCI protocol fuzzing).
- **Gap in the original:** the doc's Appendix G is unusually direct about
  this exact gap: "[perft] does not verify that the specific moves are
  correct... Agreement with these values is a necessary but not sufficient
  condition for move generator correctness," and separately warns that a
  feature-encoder bug downstream of movegen "may not be exercised by perft
  ... because the NNUE evaluation error may be small (the position is
  'close' to a legal position), making the bug difficult to detect."
  Listing perft-expansion and UCI-fuzzing as two separate backlog items (as
  the previous list did) misses that the doc is describing one combined
  failure mode.
- **How to test:** build a single fuzz harness that mutates FEN strings and
  move sequences, then checks three things per mutation, not one: (a)
  movegen legality (existing perft-style check), (b) feature-vector
  consistency against the nnue-rs reference implementation (the doc's own
  suggested debugging strategy in Appendix P), and (c) UCI parser
  robustness on malformed variants of the same input.
- **Effort/impact:** moderate (one new test harness instead of three
  separate ones) but directly targets the exact "passes perft, evaluates
  wrong" failure mode the doc calls out as the hardest bug class to catch.

### 92. Smoke-test false-negative audit for small-magnitude regressions
- **Reinforces:** Q17 (pre-SPRT smoke test), already adopted, but the doc
  admits a real gap in its own design.
- **Gap in the original:** the doc states the 100-game/1+0.1s/40%-threshold
  smoke test reliably catches regressions "larger than approximately 50
  Elo" — that number is asserted, not measured. There's no evidence for
  where the actual detection floor is, and a much smaller but still-real
  regression (say -20 Elo) could pass the smoke test and only get caught
  later by the full SPRT, or not at all if nobody runs a full SPRT on
  every change.
- **How to test:** replay the smoke test's exact parameters (100 games,
  1+0.1s) against the actual v3 vs. v1 matchup already on record (known
  true difference: -70.3 Elo) and, separately, against smaller synthetic
  regressions if any exist in the project's SPRT history — this gives an
  empirically measured detection floor instead of the doc's back-of-
  envelope estimate, and tells you whether the 40% threshold needs
  tightening.
- **Effort/impact:** essentially free — it's a re-analysis of games and
  logs that already exist (`sprt_nnue_v3.log`, `sprt_nnue_v3.pgn`) rather
  than new engine work.

### 93. Time-pressure policy net vs. engine-tell detector conflict, tested not just flagged
- **Reinforces:** #37 (engine-tell detection) + Q18 (time-pressure policy
  input).
- **Gap in the original:** Q18's own pitfalls section states the risk
  plainly — "if the policy net learns to 'blunder under time pressure,'
  the engine-tell detector might misinterpret these deliberate blunders as
  engine-like behavior... potentially triggering an unwanted switch to
  full-strength play" — and then says "this interaction should be
  explicitly handled in the persona logic" without specifying how, or
  proposing a test.
- **How to test:** once Q18 is implemented, run a matrix of test games
  across MATCH mode at several target Elo bands, specifically in
  time-pressure endgames (under 30 seconds remaining), and log whether the
  engine-tell suspicion score spikes on the policy net's own deliberate
  late-game blunders. If it does, the fix is likely to exempt
  policy-net-selected moves from suspicion-score contribution entirely,
  but that should be confirmed necessary before adding the exemption.
- **Effort/impact:** small — mostly logging and analysis of existing
  self-play infrastructure once Q18's feature exists; prevents a
  regression where MATCH mode randomly snaps to full strength near time
  pressure for no visible reason.

### 94. Blunder-rate calibration loss vs. move-prediction accuracy tradeoff, quantified
- **Reinforces:** #50 (en-passant accuracy gap) + Q19 (blunder-rate
  calibration).
- **Gap in the original:** Q19's pitfalls section says calibration "may
  conflict with move-prediction accuracy" and "this trade-off must be
  managed carefully" — again, a named risk with no attempt to size it. It's
  entirely possible the current policy net is already close to real human
  blunder rates and calibration would be solving a problem that doesn't
  exist, or that the gap is large and this is genuinely the highest-value
  item in the whole policy-net backlog.
- **How to test:** before touching the loss function, run the current
  (uncalibrated) policy net's move selections through a large batch of
  self-play or logged games, compute the actual blunder rate per rating
  bucket (moves losing >50cp per the doc's own blunder threshold from
  Q19), and compare against published Lichess blunder-rate-by-rating
  statistics. This is pure measurement, no training required, and directly
  answers whether Q19's proposed loss-function change is worth building at
  all.
- **Effort/impact:** cheap (a scripted analysis pass over existing game
  logs); this is the single best "measure twice, cut once" step available
  in the entire policy-net section of the backlog.

### 95. Reviewer accuracy formula validated against the policy net's own blunder data
- **Reinforces:** #55-56 (Reviewer move classification/accuracy formula) +
  #94 above.
- **Gap in the original:** the Reviewer tool (unaddressed by the IEEE doc
  entirely) and the policy net's blunder-rate calibration (Q19) are
  conceptually the same measurement — "how bad was this move, in a way
  that maps to a rating band" — built twice, independently, with no
  guarantee they'd agree with each other.
- **How to test:** once both exist, run the Reviewer's move classification
  over a set of games and check whether its inaccuracy/mistake/blunder
  rates per phase roughly match the move-quality distribution the policy
  net was calibrated to reproduce (item 94) at the corresponding rating
  bucket. Disagreement between the two would mean at least one of them is
  using a threshold or scale that doesn't reflect real human play.
- **Effort/impact:** deferred by nature (both dependencies must exist
  first) but cheap once they do — it's a cross-validation pass, not new
  engine work — and catches an inconsistency that would otherwise surface
  as "the Reviewer says I played well but the adaptive engine still thinks
  I'm sandbagging" style user-facing confusion.
