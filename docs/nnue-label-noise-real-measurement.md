# NNUE label noise: a real, paired measurement (2026-09-04)

Reviewer-run, on real hardware, in response to Manus's Tier 1 finding
(`docs/reinforcement/08-label-noise.md`) that a real MAE/Pearson
comparison was blocked because the existing 108M-position shards store
only mover-perspective bitboards — no castling rights, no en passant
square, no full replayable state — so they can't be legally re-searched.
Manus was right to stop there rather than approximate. This sidesteps
that blocker entirely: instead of trying to re-search old shard records,
this generates fresh paired labels *live*, during normal PGN replay,
where the full legal `Position` (real castling rights, real en passant)
is already in memory before it gets flattened to bitboards for storage.

## What was measured

Added two tiny env-var-gated hooks to `unchessed-datagen`, both no-ops
when unset (default behavior byte-identical, confirmed by
`cargo test --workspace --release` staying at 118/118 → 123/123 across
this and the prior Manus merge, no regressions):

- `UNCHESSED_NNUE_LABEL_NODES` — overrides the primary label search's
  node budget (was hardcoded `5000`).
- `UNCHESSED_NNUE_LABEL_NODES_COMPARE` — when set, runs a **second**
  search at this node budget on the exact same, already-quiet-filtered,
  fully-legal position immediately after the primary label search,
  printing `LABEL_COMPARE old=<primary> new=<compare>` — a pure
  diagnostic side-channel that never touches the real output file.

### A real methodology trap, caught before trusting a wrong number

The first attempt ran `unchessed-datagen` twice — once at 5000 nodes,
once at 50000 — on the same PGN file with the same deterministic RNG
seed, assuming that would select the same positions. It didn't: **288 of
300 output records had different bitboards between the two runs.**
Root cause: the M2 quiet filter (`|static − search_score| > margin`)
uses the *same* label search whose node budget was being varied, so
changing the budget changes which positions pass the filter, not just
their recorded score. Comparing those two runs gave a nonsense result
(MAE 159cp, Pearson 0.008 — i.e. "no relationship between the labels"),
which would have been a real self-inflicted false finding. The
`LABEL_COMPARE` hook above exists specifically to avoid this: it runs
*after* the position has already been accepted by the primary search's
filters, so the position set is identical by construction.

## Result

Four samples, four different PGN sources, two depth multipliers:

| Source | Nodes (multiplier) | n | MAE | RMS | Sign flips | Pearson |
|---|---|---|---|---|---|---|
| `data/training-elo/elo-1700.pgn` | 50000 (10x) | 300 | 20.96cp | 31.94cp | 11 (3.7%) | 0.983 |
| `data/training/leagues/bl0607.pgn` | 50000 (10x) | 1000 | 17.41cp | 26.90cp | 66 (6.6%) | 0.957 |
| `data/training-elo/elo-2500.pgn` | 100000 (20x) | 800 | 21.20cp | 34.20cp | 70 (8.8%) | 0.929 |
| `data/training/players/Carlsen.pgn` | 100000 (20x) | 800 | 21.99cp | 36.05cp | 63 (7.9%) | 0.940 |

Consistent across four independent sources and two depth multipliers:
**~17-22cp MAE, Pearson 0.93-0.98** between a 5000-node label and the
same position searched 10-20x deeper. Going from 10x to 20x depth moved
MAE only slightly (≈+1-4cp) while Pearson eased down modestly
(0.96-0.98 → 0.93-0.94) and sign-flips ticked up (3.7-6.6% → 7.9-8.8%)
— a mild, sub-linear growth pattern consistent with real but bounded
search instability, not a runaway blowup that would suggest the 5000
vs 50000/100000 gap is somehow a special case.

## Why this matters: it contradicts the working theory

`docs/ieee-low-cp-val-mae-and-persona.md` (round 14, simulation-based)
modeled the 5000-node HCE label noise floor at **~50-56cp** (`σ ≈ 70cp`
assumed teacher noise, Gaussian Bayes floor `σ√(2/π) ≈ 55.85cp`), and
used that to argue the NNUE's measured best val-MAE (47.8-57.4cp across
the data-scaling ladder) was already sitting near the label-noise floor
— i.e., more data or more training wouldn't help because the *labels*
were the ceiling. That was always flagged as a *modeling assumption*,
not a measurement (`docs/ieee-low-cp-val-mae-and-persona.md`, Section
VII: "Teacher σ ... is a modelling assumption, not a measurement of
5000-node HCE vs Stockfish on this engine").

This is that measurement, and it doesn't support the assumption at the
scale tested: **real label noise from a 10x depth increase (~17-21cp)
is well below the assumed ~50-56cp floor**, and also well below the
NNUE's actual measured val-MAE (47.8-57.4cp). If the 5000-node search
were the dominant noise source, going 10x deeper should have moved
scores by something comparable to that assumed floor; it moved them by
roughly a third of it.

## What this does and doesn't establish

**Does**: the specific `σ≈70cp` assumption in the round-14 analysis
is not supported by a real measurement at 10x depth, on these two
samples. The label-noise explanation for the NNUE's strength plateau
needs to be revisited, not treated as settled.

**Does not**: prove NNUE strength is capped by something else instead
(architecture, data volume, or a different label-quality axis entirely).
Two samples of 300 and 1000 positions, one depth multiplier, one engine,
is a real but narrow measurement — not a replacement for the full
`nnue_relabel_existing.py` pipeline's `compare`/`apply` workflow at real
corpus scale. It also doesn't measure noise *within* the 5000-node
search itself (e.g., from any non-determinism) — this diagnostic
only compares two point-estimates at two depths, not repeated runs at
the same depth.

**Also doesn't** yet answer whether the 10x-deeper search's evaluation
is actually *more correct* — sign flips (3.7-6.6%) show real
disagreement exists even where 10x deeper mostly agrees; whether the
deeper search is the more trustworthy one on those disagreements is a
different, unaddressed question.

## What's actually next

1. **Partially done, could go further.** Four samples across four PGN
   sources and two depth multipliers (10x, 20x) now agree with each
   other — that's enough to call the ~50-56cp assumption genuinely
   contradicted at this scale, not a fluke. What's still untested: a
   much larger depth multiplier (the label-noise theory's own numbers
   assumed something closer to Stockfish-strength search, i.e. far
   beyond 20x) — worth doing only if there's a specific reason to think
   the relationship changes qualitatively at that range, not as a
   default next step.
2. **If the low-noise finding holds at scale**, the NNUE strength
   ceiling is more likely architecture, effective-data-volume, or
   something not yet isolated — not primarily label noise from search
   depth. That reopens directions round 14's analysis had deprioritized.
3. **This does not change any default or justify a retrain by itself.**
   It's a real, surprising, and unusually cheap (minutes, no cloud) data
   point that the working theory the project had been operating under
   needs a second look, not a finished conclusion.

## Reproducing this

```
UNCHESSED_NNUE_MIN_BASE_SECS=0 UNCHESSED_NNUE_LABEL_NODES_COMPARE=50000 \
  ./target/release/unchessed-datagen nnue out.bin 0 1 <n> <pgn file(s)>
```
`LABEL_COMPARE old=.. new=..` lines go to stderr, one per accepted
sample, in generation order. `UNCHESSED_NNUE_LABEL_NODES` (without
`_COMPARE`) instead changes the *primary* label search itself — do not
use that one for a paired comparison, per the methodology trap above.
