# NNUE coverage-vs-capacity diagnostic: a small, real, negative result (2026-09-04)

Reviewer-run, on real hardware, following the design in
`docs/reinforcement/13-nnue-ceiling.md` — scoped down to the minimum
that still tests the leading hypothesis honestly. Coverage vs. raw
sampling at **fixed capacity** (the current v4 width). The capacity
axis (narrower/wider model) is deliberately deferred, matching that
doc's own priority order: "do not spend on [capacity] until the
narrow/current comparison is informative."

## Setup

- **Train pool**: 240,000 quiet-filtered NNUE samples generated from 47
  of the 49 committed PGN files (16 workers, `UNCHESSED_NNUE_MIN_BASE_SECS=0`).
- **Held-out set**: 30,000 samples from the **other 2 PGN files**
  (`lichess-2022-10-05/elo-2000-2300.pgn`, `players/Nakamura.pgn`) —
  file-disjoint from the train pool, so no game overlap by construction.
- **RAW set**: 100,000 records, uniform random sample from the pool.
- **BALANCED set**: 100,000 records, stratified evenly across the 8
  piece-count buckets used by the shipped v4 output head
  (`(pieces-1)//4` clamped to 7) — 12,500 per bucket, drawn with
  replacement where a bucket has fewer than 12,500 unique records.
- **Recipe**: identical for both — the round-13 defended recipe
  (best-checkpoint export, early-stop patience 3, batch 16384).
- **Evaluation**: both trained nets scored against the *same* held-out
  set, broken down by piece-count bucket, using
  `tools/train_nnue.py`'s own validated export-consistent manual
  feature functions (the ones its own `selfcheck` cross-checks the fast
  path against to 1e-8) — not a re-derived index scheme.

## The pool's real coverage imbalance (measured, not assumed)

| Bucket | Records in 240k pool |
|---|---|
| 0 (≤4 pieces) | **2** |
| 1 (5-8 pieces) | 143 |
| 2 (9-12 pieces) | 2,409 |
| 3 (13-16 pieces) | 12,755 |
| 4 (17-20 pieces) | 29,081 |
| 5 (21-24 pieces) | 48,033 |
| 6 (25-28 pieces) | 70,665 |
| 7 (29-32 pieces) | 76,912 |

A >38,000x gap between the rarest and most common bucket. This is
real, not a simulation — exactly the kind of imbalance the coverage
hypothesis is about.

## Result

| Bucket | n (held-out) | RAW MAE | BALANCED MAE | Δ | Relative change |
|---|---|---|---|---|---|
| 1 | 26 | 148.48cp | 208.25cp | +59.77 | **−40.3%** |
| 2 | 307 | 173.50cp | 159.21cp | −14.29 | **+8.2%** |
| 3 | 1,548 | 131.94cp | 136.00cp | +4.06 | −3.1% |
| 4 | 3,643 | 115.29cp | 128.65cp | +13.36 | −11.6% |
| 5 | 6,029 | 89.16cp | 109.82cp | +20.66 | −23.2% |
| 6 | 8,772 | 58.13cp | 74.83cp | +16.70 | **−28.7%** |
| 7 | 9,675 | 34.14cp | 44.12cp | +9.98 | **−29.2%** |
| **Overall** | 30,000 | 68.64cp | 82.63cp | +13.99 | −20.4% |

(Bucket 0 had zero held-out records — no deep-endgame positions
appeared in either held-out PGN file.)

## Applying the predeclared decision rule

`13-nnue-ceiling.md` predeclared the rule before any result existed —
worth quoting exactly rather than rounding to a convenient answer:

> Coverage supports the hypothesis if B improves held-out error on
> rare/deployment-weighted strata by a practically meaningful margin
> (suggested threshold: ≥5% relative MAE improvement in at least two
> predeclared strata) while aggregate common-stratum error does not
> materially regress.

**This does not clear that bar.** Only one bucket (2) improved by
≥5% (+8.2%). Bucket 1 got dramatically worse (−40.3%). The two most
common, most deployment-relevant strata (6 and 7, together 61% of the
held-out set) both regressed by ~29% — a real, material regression, not
noise. **At this scale, naive piece-count-balanced resampling made the
net worse, not better, including in exactly the positions it will see
most often in real games.**

## A real confound worth being honest about, not a reason to discard the result

Buckets 0-2 have so few genuinely distinct positions in this small pool
(2, 143, 2,409) that "balancing" them to 12,500 records each meant
heavy sampling *with replacement* — bucket 0 and 1 especially are
mostly the same handful of positions repeated hundreds of times. That
is not the same intervention the coverage hypothesis actually proposes
(exposing the net to more *genuinely distinct* rare positions); it's
closer to over-weighting a tiny, duplicated sample, which is a
plausible mechanism for the regression seen in the common strata (the
net spends training capacity fitting repeated rare examples instead of
the well-covered, high-frequency positions it will mostly be evaluated
on). **This experiment is a real, honest test of "does hard-duplicated
piece-count rebalancing on this specific small corpus help" — not
necessarily a clean test of "would genuine, non-duplicated rare-position
diversity help."** Those are different questions, and this result only
answers the first one.

## What this does and doesn't establish

**Does**: naive equal-weight-by-duplication rebalancing, at this scale,
on this corpus, made the net measurably worse across the board,
including where it matters most (the common strata). The specific
mechanism proposed here is not a free win.

**Does not**: settle whether coverage matters at all. The confound above
means a fair test needs either (a) a genuinely larger, more diverse pool
where rare buckets have enough unique positions that "balanced" doesn't
mean "duplicated" — the full 178M self-play corpus almost certainly has
far more distinct rare-bucket positions than this 240k slice of 49 PGN
files — or (b) a soft reweighting during training (loss weighting, not
hard resampling) that doesn't force literal duplication.

## What's actually next

Not a bigger version of this exact experiment by default. Two real
options, in order of cost:
1. **Re-run this same design against the real 178M self-play corpus**
   (already on the reviewer's WSL disk from earlier rounds) instead of
   the small committed PGN set — cheap, no cloud, and removes the
   duplication confound if the rare buckets there have real depth.
2. **If that still regresses**, the coverage-via-hard-resampling idea is
   probably not the fix, and the project should look at soft
   reweighting or revisit whether capacity (the axis this round
   deliberately deferred) is worth testing after all.

This does not change any default, does not justify a retrain, and is
not evidence for or against a cloud spend by itself — same standing
rules as every other result this round.

## Reproducing this

```
bash scripts/research/coverage_capacity_gen.sh <out_dir>
python scripts/research/coverage_capacity_sample.py <out_dir>/train_pool 100000 <out_dir>/sets
EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 BATCH_SIZE=16384 DEVICE=cpu \
  python tools/train_nnue.py <out_dir>/nnue_raw.bin 20 <out_dir>/sets/raw.bin
EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 BATCH_SIZE=16384 DEVICE=cpu \
  python tools/train_nnue.py <out_dir>/nnue_balanced.bin 20 <out_dir>/sets/balanced.bin
python scripts/research/coverage_capacity_eval.py <out_dir>/nnue_raw.bin <out_dir>/heldout/heldout.bin
python scripts/research/coverage_capacity_eval.py <out_dir>/nnue_balanced.bin <out_dir>/heldout/heldout.bin
```
