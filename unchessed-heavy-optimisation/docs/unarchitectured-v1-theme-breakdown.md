# What the aggregate policy accuracy was hiding

Applies the theme-balancing idea from *Grounded Chess Reasoning in Language
Models via Master Distillation* ([arXiv:2603.20510](https://arxiv.org/abs/2603.20510))
to this project's calibration corpus.

That paper credits much of its result to **theme-balanced sampling for
comprehensive tactical coverage** rather than model scale. The generalisable
claim survives the move from LLMs to a 4.2M-param policy net: a single
aggregate accuracy number hides systematic blindness to particular motifs,
because motifs are not uniformly distributed in natural positions.

Our corpus is balanced by **phase** (200 opening / 200 middlegame / 200
endgame) and by nothing else. Nothing in the repository tagged themes, so
every headline number so far — top-1 0.2683, mean first-move regret 146cp —
was an average over an uncontrolled motif mix.

## Method

`tools/tag_calibration_themes.py` tags each position from the **teacher's**
MultiPV scores and `python-chess` board logic. Tagging never looks at the
model being evaluated, so it cannot be contaminated by what it is used to
judge. Each theme is a mechanically checkable property, not a subjective
label.

## The corpus is heavily skewed

| theme | count | share |
|---|---:|---:|
| quiet | 451 | 75.2% |
| many_good_moves | 221 | 36.8% |
| capture | 120 | 20.0% |
| mate_available | 57 | 9.5% |
| hanging_piece | 51 | 8.5% |
| only_good_move | 49 | 8.2% |
| check | 44 | 7.3% |
| endgame_technique | 35 | 5.8% |
| fork | 19 | 3.2% |

Three quarters of the corpus is quiet, and forks appear 19 times. This is what
natural game positions look like; it is not a sampling bug. But it means the
aggregate is dominated by one motif class.

## The headline number hides a 5.5x spread

| theme | n | top-1 | mean regret (cp) |
|---|---:|---:|---:|
| capture | 120 | **0.7583** | 38.9 |
| hanging_piece | 51 | **0.7255** | 43.3 |
| only_good_move | 49 | 0.6327 | 195.3 |
| fork | 19 | 0.2632 | 229.7 |
| check | 44 | 0.2273 | 332.4 |
| mate_available | 57 | 0.2105 | 408.7 |
| endgame_technique | 35 | 0.2000 | 141.7 |
| many_good_moves | 221 | 0.1538 | 127.4 |
| quiet | 451 | **0.1375** | 155.3 |

The reported 0.2683 is the average of a **0.758 vs 0.138 split** — a 5.5x
ratio between the model's best and worst motif classes.

Concretely: **57% of all the model's correct first-move predictions come from
the 20% of positions that involve a capture.** Its apparent competence is
concentrated in one motif.

Worst absolute regret is `mate_available` (408.7cp) — when a forced mate
exists, the policy's first choice is usually not it. That is consistent with
the round-8 finding that the checkpoint ranks a forced back-rank mate 10th of
17.

## Why this matters more than it looks

Captures are precisely what the engine **already gets for free**. Running the
same partition through the existing MVV-LVA heuristic:

| group | n | neural | MVV-LVA | gain |
|---|---:|---:|---:|---:|
| capture | 120 | 0.7583 | 0.6417 | +0.117 |
| quiet | 451 | 0.1375 | 0.0310 | +0.107 |

On captures the 9.72ms forward pass buys **+0.117 over a heuristic costing
nothing** — the model is largely re-deriving MVV-LVA. Its genuinely
unique contribution is quiet positions, where MVV-LVA is nearly blind (0.031)
and the model is 4.4x better.

Across the corpus the forward pass yields **62 extra correct first moves out
of 600 (10.3%)** over the free heuristic, and **77% of that margin comes from
quiet positions**.

This *refines* rather than overturns
`docs/unarchitectured-v1-why-the-hint-costs-elo.md`. That analysis showed the
policy is a better orderer than the real baseline on every metric, and that
the cost lands where the benefit is worth least. This adds: the benefit is
also **concentrated in one motif the engine cannot already handle**, and much
of the headline accuracy is duplicated work.

## Implications

- **Aggregate policy accuracy should not be quoted on its own again.** The
  per-theme table is the honest summary; 0.2683 flatters the model by
  averaging a strong capture case into a weak quiet one.
- **Nothing here justifies enabling `UnarchitecturedHint`.** If anything it
  weakens the case: over half the apparent skill duplicates a free heuristic,
  and no configuration ever trended positive across four SPRT batches.
- **If the model is ever retrained, this says where to aim.** Theme-balanced
  sampling (the paper's actual recommendation) would target quiet positions,
  mates and forks — the classes where the model is weak *and* where no cheap
  heuristic exists. Training on more captures would buy nothing the engine
  does not already have.
- **A cheap hybrid is worth considering before more model work.** Since the
  model only clearly beats free heuristics on quiet positions, a policy that
  consults the net *only* when no capture or check is available would keep
  most of the unique value at a fraction of the average cost. That is a real
  design, not a measured result, and it would need its own SPRT.

## Honest limits

- **Offline analysis, not games.** It explains and qualifies existing
  measurements; it does not replace an SPRT. No `cutechess-cli` in this
  sandbox.
- **Theme definitions are mechanical and somewhat arbitrary at the margins.**
  `fork` counts two attacked valuable pieces after the move, which will catch
  some non-forks and miss some real ones. They are consistent and
  reproducible, not authoritative.
- **Small cells.** `fork` has n=19; its 0.2632 is indicative only. The large
  cells (quiet 451, many_good_moves 221, capture 120) carry the argument.
- The MVV-LVA comparison uses the existing `heuristic_move_score`, which is
  MVV-LVA plus promotion and check bonuses — the same baseline the round-6
  calibration used.
- Corpus provenance is source-population-disjoint only, per
  `docs/unarchitectured-v1-calibration.md`.
