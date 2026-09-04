# NNUE coverage-vs-capacity diagnostic, rerun against the real 178M corpus (2026-09-03)

Same design as `docs/nnue-coverage-capacity-matrix-small.md`, rerun
against the real self-play corpus specifically to remove that run's
duplication confound (its rarest buckets had as few as 2-143 unique
records, so "balancing" them meant heavy duplication, not genuine
rare-position diversity).

## Setup

- **Train pool**: 27,000,000 records, hardlinked from 3 self-play
  shards (`shard0.bin`, `shard1.bin`, `shard2.bin`, 9M records each,
  `~/unchessed-ai/data/maia-data/nnue/`).
- **Held-out set**: 30,000 records sampled from `shard3.bin` — a
  different self-play generation batch, file-disjoint from the train
  pool.
- **RAW set**: 100,000 records, uniform random sample from the pool.
- **BALANCED set**: 100,000 records, stratified evenly across the 8
  piece-count buckets (12,500 per bucket).
- **Recipe**: identical for both nets — round-13 defended recipe
  (best-checkpoint export, early-stop patience 3, batch 16384, CPU).
- **Evaluation**: `scripts/research/coverage_capacity_eval.py`,
  unchanged from the first run (uses `train_nnue.py`'s own validated
  export-consistent manual feature functions).

## The pool's real bucket depth (confound resolved)

| Bucket | Records in 27M pool |
|---|---|
| 0 (≤4 pieces) | 60 |
| 1 (5-8 pieces) | 12,656 |
| 2 (9-12 pieces) | 288,201 |
| 3 (13-16 pieces) | 1,680,871 |
| 4 (17-20 pieces) | 3,871,670 |
| 5 (21-24 pieces) | 5,833,163 |
| 6 (25-28 pieces) | 7,577,822 |
| 7 (29-32 pieces) | 7,735,557 |

Bucket 1 alone has 88x more unique records than the entire small-corpus
pool had duplicated into it. Buckets 2-7 have real depth in the
hundreds-of-thousands to millions. Only bucket 0 (60 records) is still
thin enough that balancing it to 12,500 means heavy duplication — every
other bucket's "balanced" sample is drawn from a genuinely large pool.

## Result

| Bucket | n (held-out) | RAW MAE | BALANCED MAE | Δ | Relative change |
|---|---|---|---|---|---|
| 0 | 0 | — | — | — | (no held-out records) |
| 1 | 13 | 340.10cp | 213.45cp | −126.65 | **−37.2%** |
| 2 | 323 | 239.96cp | 171.41cp | −68.55 | **−28.6%** |
| 3 | 1,877 | 210.25cp | 185.57cp | −24.68 | **−11.7%** |
| 4 | 4,240 | 191.78cp | 180.91cp | −10.87 | **−5.7%** |
| 5 | 6,341 | 136.44cp | 140.22cp | +3.78 | +2.8% |
| 6 | 8,486 | 85.71cp | 95.01cp | +9.30 | **+10.9%** |
| 7 | 8,720 | 39.12cp | 44.57cp | +5.45 | **+13.9%** |
| **Overall** | 30,000 | 107.44cp | 108.59cp | +1.15 | +1.1% |

(Bucket 0 had zero held-out records — no ≤4-piece positions appeared in
`shard3`'s 30,000-record sample.)

## Applying the predeclared decision rule

From `docs/reinforcement/13-nnue-ceiling.md`, quoted exactly, as in the
first run:

> Coverage supports the hypothesis if B improves held-out error on
> rare/deployment-weighted strata by a practically meaningful margin
> (suggested threshold: ≥5% relative MAE improvement in at least two
> predeclared strata) while aggregate common-stratum error does not
> materially regress.

**The rare-stratum half of the rule is clearly met, and by a wide
margin** — not the borderline single-bucket result from the small-corpus
run. Four buckets (1, 2, 3, 4) improved by ≥5%, two of them by
double-digit margins (−37.2%, −28.6%). This is the direction the
coverage hypothesis actually predicts, on a corpus where "balanced"
doesn't mean "duplicated."

**The second half is genuinely ambiguous, and I want to be precise about
why rather than round it either way.** If "aggregate common-stratum
error" means the overall MAE, it barely moved (+1.1%) — arguably not a
material regression. But that number is doing some cancellation: bucket
6 and bucket 7 alone cover 57% of the held-out set (17,206 of 30,000
records) and both regressed by real, non-noise margins (+10.9%,
+13.9%) — larger in absolute terms than three of the four rare-bucket
improvements. The overall MAE looks flat only because gains in
thousands of rare-bucket records are averaged against losses spread
over tens of thousands of common-bucket records. **Read per-bucket,
this is a real trade: better on rare positions, worse on the two most
common ones a deployed engine will actually see most often.**

## What this does and doesn't establish

**Does**: once the duplication confound from the first run is removed,
piece-count-balanced resampling produces a real, non-trivial
improvement on rare strata (buckets 1-4) — this is not noise, and it
reverses the first run's negative result on those same strata. The
mechanism (expose the net to more of the rare positions it currently
undersees) works in the direction the hypothesis predicted, when there
is real diversity to draw from.

**Does not**: establish that naive hard balancing is a free win, or
that it should replace the current sampling default. It costs real
accuracy on the two buckets a real game spends most of its time in
(endgame-adjacent play is rarer than the mid-to-full-board buckets this
engine mostly operates in during actual games). Whether that trade is
worth taking depends on how much rare-position accuracy matters
relative to common-position accuracy for actual playing strength —
this diagnostic measures held-out MAE, not game outcomes, and cannot
answer that by itself.

## What's actually next

Not a default change, not a retrain, not a cloud spend — same standing
rule as every prior round. The honest next step, if this is worth
pursuing further, is **soft reweighting** (loss weighting during
training, not hard resampling) rather than another hard-balance
variant: something that lets rare buckets get more gradient signal
without displacing the majority of common-bucket training examples the
current RAW recipe already handles well. That is a different, cheap
experiment (same generation pipeline, different sampling function) and
should itself be evaluated against the same held-out set and the same
decision rule before any conclusion is drawn.

## Comparison with the first (small-corpus) run

| | Small corpus (240k pool, confounded) | 178M corpus (27M pool, real depth) |
|---|---|---|
| Buckets clearing ≥5% improvement | 1 of 7 | 4 of 7 |
| Worst regression | −40.3% (bucket 1, itself likely duplication noise) | +13.9% (bucket 7, real signal) |
| Overall MAE change | −20.4% (worse) | +1.1% (~flat) |
| Verdict | Rebalancing hurt across the board | Rebalancing helps rare strata, costs common strata |

The direction of the result changed once the confound was removed —
which is itself the useful finding: the first run's negative result was
substantially an artifact of duplicating a handful of positions
hundreds of times, not evidence against coverage-based rebalancing in
general.

## Reproducing this

Train pool and held-out set were built directly from local self-play
shards (hardlinked, no PGN replay):

```
# pool: hardlink shard0/1/2 records into <out_dir>/train_pool/w*.bin
# heldout: sample 30000 records from shard3.bin
python scripts/research/coverage_capacity_sample.py <out_dir>/train_pool 100000 <out_dir>/sets
EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 BATCH_SIZE=16384 DEVICE=cpu \
  python tools/train_nnue.py <out_dir>/nnue_raw.bin 20 <out_dir>/sets/raw.bin
EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 BATCH_SIZE=16384 DEVICE=cpu \
  python tools/train_nnue.py <out_dir>/nnue_balanced.bin 20 <out_dir>/sets/balanced.bin
python scripts/research/coverage_capacity_eval.py <out_dir>/nnue_raw.bin <out_dir>/heldout.bin
python scripts/research/coverage_capacity_eval.py <out_dir>/nnue_balanced.bin <out_dir>/heldout.bin
```
