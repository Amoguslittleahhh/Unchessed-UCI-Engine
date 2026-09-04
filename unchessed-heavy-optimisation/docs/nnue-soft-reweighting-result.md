# NNUE soft reweighting: predeclared design tested, real negative result (2026-09-03)

Executes the design predeclared in
`docs/reinforcement/32-nnue-soft-reweighting.md`. Implementation: an
optional `weight` argument added to `tools/train_nnue.py`'s `wdl_loss`
(off by default — existing call sites pass `weight=None` and reproduce
today's behavior bit for bit), applied via `NNUE_BUCKET_WEIGHTS=1` with
the predeclared inverse-pool-frequency table, clipped to `[0.25x, 20x]`:
`[20.0, 20.0, 11.7, 2.0, 0.87, 0.58, 0.45, 0.44]` for buckets 0-7.

## Setup

Same RAW training set (`sets/raw.bin`, 100,000 records) and same
held-out set (`heldout.bin`, 30,000 records from `shard3`) as the
178M-corpus coverage-vs-capacity run, reused unchanged so the only
variable is the loss-weighting change. Same recipe (early-stop patience
3, batch 16384, 20-epoch cap, CPU). Validation-MAE-based early
stopping is unaffected by the weighting — `evaluate_iter` computes MAE
directly, not through `wdl_loss`, so it stays an unweighted measure of
true error throughout training.

## Result

| Bucket | RAW MAE | BALANCED MAE | REWEIGHTED MAE | Reweighted vs RAW |
|---|---|---|---|---|
| 1 | 340.10cp | 213.45cp | 349.09cp | **+2.6%** (worse) |
| 2 | 239.96cp | 171.41cp | 286.64cp | **+19.5%** (worse) |
| 3 | 210.25cp | 185.57cp | 232.79cp | **+10.7%** (worse) |
| 4 | 191.78cp | 180.91cp | 191.84cp | +0.0% (flat) |
| 5 | 136.44cp | 140.22cp | 135.79cp | −0.5% (slightly better) |
| 6 | 85.71cp | 95.01cp | 83.05cp | −3.1% (better) |
| 7 | 39.12cp | 44.57cp | 37.93cp | −3.0% (better) |
| **Overall** | 107.44cp | 108.59cp | 108.14cp | +0.65% |

**This is the opposite of what the design predicted, and the opposite
of what hard resampling did.** Soft reweighting made the rare/mid
strata it was meant to help *worse* — bucket 2 by nearly 20% — while
slightly improving the common strata it was meant to protect from
being starved.

## Why, verified rather than guessed at

Checked the actual per-bucket composition of the 100,000-record RAW
training set directly (not assumed from pool proportions):

| Bucket | Records in 100k RAW training set |
|---|---|
| 0 | 0 |
| 1 | 44 |
| 2 | 1,030 |
| 3 | 6,145 |
| 4 | 14,525 |
| 5 | 21,629 |
| 6 | 28,101 |
| 7 | 28,526 |

Bucket 1 has 44 examples in the *entire* training set (about 43 after
the validation split) — not 44 per batch, 44 total. Upweighting those
same ~43 specific positions' loss by 20x, epoch after epoch, doesn't
teach the net "rare positions in general" — it teaches the net those
~43 examples' specific idiosyncrasies, disproportionately hard, with no
new information to generalize from. Held-out bucket-1 positions are
different specific instances, so that overfitting doesn't transfer,
and the held-out error on that bucket gets worse, not better. The same
mechanism, weaker, likely explains bucket 2's regression (1,030 real
examples is still small enough for 11.7x weighting to overfit rather
than generalize).

This is the mirror image of why hard resampling *did* help those same
buckets: hard resampling adds more *distinct* rare-bucket records into
the training set (drawing with replacement from a real pool of 12,656
and 288,201 candidates respectively, not just repeating the same ~44
already-selected ones), which is a genuine diversity increase. Soft
reweighting on an unchanged RAW sample has no such mechanism — it can
only amplify whichever handful of rare examples happened to be drawn,
which is a much weaker (and here, actively harmful) intervention than
changing what's in the training set at all.

## Applying the predeclared decision rule

From `32-nnue-soft-reweighting.md`, quoted exactly:

> If soft reweighting regresses common strata by more than hard
> resampling did, or fails to move rare strata by at least half of hard
> resampling's gain, the mechanism itself (not just the resampling
> implementation) is likely not worth pursuing further at this scale,
> and the project should stop investigating piece-count-bucket
> reweighting.

**This branch is triggered, unambiguously.** Soft reweighting didn't
recover any of hard-BALANCED's rare-stratum gain — it moved three of
the four rare/mid strata in the wrong direction, with bucket 2
regressing worse than hard-BALANCED had ever regressed a common
stratum. Per the predeclared rule, this closes out piece-count-bucket
loss reweighting as a direction at this data scale.

## What this does and doesn't establish

**Does**: on a training set where rare buckets have only tens to low
thousands of real examples, loss reweighting doesn't substitute for
actual data diversity — it amplifies overfitting on whatever few
examples exist rather than approximating the effect of having more of
them. The two interventions (resample vs. reweight) are not
interchangeable ways to reach the same outcome, even though they're
often treated as such in ML folklore.

**Does not**: rule out soft reweighting in general — a corpus where
even the rare buckets have, say, hundreds of thousands of distinct
examples (rather than tens to low-thousands) might behave completely
differently, since the overfitting mechanism identified here depends
specifically on rare-bucket example counts being small. That's a
different, larger-scale question this experiment wasn't designed to
answer and doesn't claim to.

**No default change, no retrain, no cloud spend** — same standing rule
as every prior round in this series. Per the decision rule this
experiment predeclared for itself, the coverage/reweighting research
thread on this specific mechanism is done; further NNUE-ceiling work
should look elsewhere (e.g. the capacity axis `13-nnue-ceiling.md`
deliberately deferred, or a genuinely larger rare-bucket sample if the
project wants to revisit hard resampling with even more real diversity
than the 178M-corpus run already had).

## Code change

`tools/train_nnue.py`: `wdl_loss` gained an optional `weight=None`
parameter (weighted mean when given, identical to today's `.mean()`
when omitted); `train()` gained `NNUE_BUCKET_WEIGHTS=1`-gated per-batch
weight lookup using a new `BUCKET_WEIGHT` constant. Both default off;
`selfcheck` was re-run and still passes bit-for-bit
(`numpy forward matches model, max diff 1.49e-08`), confirming the
unweighted default path is unaffected.

## Reproducing this

```
NNUE_BUCKET_WEIGHTS=1 EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 BATCH_SIZE=16384 DEVICE=cpu \
  python tools/train_nnue.py <out_dir>/nnue_reweighted.bin 20 <out_dir>/sets/raw.bin
python scripts/research/coverage_capacity_eval.py <out_dir>/nnue_reweighted.bin <out_dir>/heldout.bin
```
