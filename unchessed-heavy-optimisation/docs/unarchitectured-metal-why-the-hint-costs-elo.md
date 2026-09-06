# Why the Unarchitectured Metal root hint costs Elo

Round 7 ran four real SPRT batches and every one landed negative, never
positive. It also showed the obvious explanation is wrong, and said so: depth
loss from the hint was negligible (mostly 0, occasionally +1, rarely -1), so
"the clock tax reduces depth" is not supported by the project's own data. The
round-7 writeup left the cause explicitly unexplained and offered one
hypothesis — that the ordering itself steers the search toward worse lines.

**That hypothesis is now tested and false.** This document records what the
data actually shows.

## Method

`tools/analyse_unarchitectured_metal_ordering_risk.py` replays the real exported
checkpoint over the 600-position provenance-disjoint calibration corpus, which
already carries full Stockfish MultiPV per-move scores. Average accuracy cannot
answer the question, because alpha-beta does not pay for the average — it pays
for searching a bad move *first*. So the tool measures the quantities ordering
actually bills: first-move regret in centipawns, blunder tail rates, and where
the true best move sits in the proposed order.

Crucially it compares against **the real baseline**. Reading
`go_with_root_hints`, the hint applies only at `depth == start_depth`; on that
first pass every root score is still `-MATE`, so the `sort_by_key(-score)`
fallback is a no-op and the true baseline is **movegen order** — not MVV-LVA.
All three are reported.

Mate labels arrive as ±100000 and would otherwise dominate every mean (a single
missed mate outweighing hundreds of positions), so they are clamped to ±2000.
Blunder and disaster *rates* are unaffected, both thresholds sitting far below
the clamp.

## Result: the policy is a genuinely better orderer

600 positions, full corpus, no skips.

| metric | neural | MVV-LVA | movegen (real baseline) |
|---|---:|---:|---:|
| top-1 accuracy | **0.2683** | 0.1683 | 0.0683 |
| mean first-move regret (cp) | **146.3** | 282.7 | 290.5 |
| median first-move regret (cp) | **32.5** | 182.0 | 185.5 |
| p90 first-move regret (cp) | **462.3** | 658.2 | 613.1 |
| blunder rate (≥200cp) | **0.2317** | 0.4817 | 0.4800 |
| disaster rate (≥500cp) | **0.0867** | 0.2383 | 0.2033 |
| mean rank of best move | **5.99** | 12.71 | 15.34 |
| best move in top 3 | **0.4500** | 0.2817 | 0.1650 |
| best move in bottom half | **0.1617** | 0.4250 | 0.5317 |
| confidently-wrong blunder rate | **0.2423** | 0.5034 | 0.4800 |

The neural policy wins **every** metric, including the tail ones that ordering
cost is most sensitive to. It halves the blunder rate against movegen order and
cuts mean rank of the best move from 15.3 to 6.0. There is no "confidently
wrong is worse" effect: its confident cases are *better* than its average.

So the hint is not steering the search into worse lines. The round-7
hypothesis is falsified.

## So where does the Elo go?

Two structural facts, neither about ordering quality:

**1. The benefit is confined to one pass that the transposition table has
already made cheap.** The hint sorts only `depth == start_depth`. Iterative
deepening then overwrites ordering with real alpha-beta scores, and the TT
supplies a best move from the previous iteration at every later depth. Better
ordering on the *first* pass of an iterative deepening search is worth very
little: depth 1 is microseconds of work, and everything after it is ordered by
information the search produced itself. The measured ordering advantage is
real, and almost entirely spent on the cheapest pass in the search.

**2. The cost is charged against the move budget, and scales inversely with
the clock.** `preprocessing_elapsed` is subtracted from the deadline. Using
round 7's measured 9.72ms forward pass on real hardware against this engine's
own budget formula:

| time control | clock | soft budget | hint cost |
|---|---:|---:|---:|
| tc=5+0.05 | 5000ms | 108ms | **9.0%** |
| tc=5+0.05 | 2500ms | 66ms | **14.7%** |
| tc=60+0.6 | 60000ms | 2450ms | **0.4%** |
| tc=60+0.6 | 30000ms | 1450ms | **0.7%** |

This reproduces round 7's monotonic trend exactly, from arithmetic rather than
games:

| config | measured Elo | hint cost share |
|---|---:|---:|
| MinTime=1000, tc=5+0.05 | -26.1 | ~9-15% |
| MinTime=1000, tc=5+0.05 (replication) | -15.1 | ~9-15% |
| MinTime=30000, tc=60+0.6 | -5.8 | ~0.4-0.7% |

**The feature pays a real cost for a real benefit that lands almost entirely
where it is worth the least.** At fast controls the cost is ~10% of the move
budget; at slow controls it approaches zero and so does the measured harm.
That is the whole trend, and it needs no appeal to bad move ordering.

Note this also explains why depth barely moved: 9.72ms out of a 108ms budget is
usually not enough to lose a whole iteration, so the loss shows up as slightly
less time at the final depth rather than a lower depth number. A depth
histogram was always going to under-report this.

## What this implies

- **The negative results are explained by cost placement, not model quality.**
  The policy is a better orderer than anything the engine has for free.
- **Enabling the hint by default remains unjustified** — unchanged. A better
  first-pass ordering that no configuration ever converted into a positive
  result is not worth 9-15% of the budget at fast controls.
- **If the feature is ever revived, the fix is structural, not numerical.** The
  hint would need to reach the passes where ordering is actually expensive
  (deep re-searches), or cost near zero. Options: reuse one forward pass across
  the whole game tree rather than per `go()`, or seed the TT instead of only
  the root sort. Both are real work with no evidence of payoff, which is
  exactly what "retiring the feature" means.
- **The remaining round-8 item — the isolated `MinTime` retest — is worth
  less now.** It would separate two confounded variables whose mechanism this
  analysis already explains from the budget arithmetic. Still nice to have,
  no longer diagnostic.

## Honest limits

- This is a **mechanism analysis on offline data, not a game result**. It
  explains the SPRT numbers; it does not replace them. No games were played
  here — `cutechess-cli` is unavailable in this sandbox.
- Ordering quality is measured against Stockfish's per-move scores at the
  labelling node count, which is ground truth for "which move is best", not a
  direct simulation of alpha-beta node counts. A true node-count measurement
  would require instrumenting the Rust search, which cannot be compiled here.
- The budget table is arithmetic from this engine's own `Limits::budget` plus
  round 7's measured 9.72ms on the reviewer's Core Ultra 9 285H. **That
  9.72ms is host-specific**; the percentages move with the hardware.
- The corpus is provenance-disjoint at the source-population level only, as
  documented in `docs/unarchitectured-metal-calibration.md`.
