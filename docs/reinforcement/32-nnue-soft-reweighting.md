# NNUE soft reweighting by piece-count bucket: predeclared design (2026-09-03)

Follow-up to `docs/nnue-coverage-capacity-matrix-178m.md`. That
diagnostic found that **hard resampling** (piece-count-balanced
training sets) genuinely helps rare/mid buckets (1-4: −5.7% to −37.2%
relative MAE) once given real data depth, but costs the two most
common buckets (6, 7: +10.9%, +13.9%) — a real trade-off, not a clean
win. This doc predeclares a cheaper, less destructive alternative
before running it, so the decision rule can't be shaped by the result.

## Why soft reweighting instead of another hard-resampling variant

Hard resampling changes *which* positions the net sees — rare buckets
are drawn from the same pool repeatedly (with real diversity now, not
duplication) until they match the majority buckets' count, which means
the majority buckets' full, well-covered example set is discarded down
to the same size. That's the likely mechanism for the bucket 6/7
regression: the net trains on fewer distinct common-bucket examples
than it otherwise would, even though the common bucket itself has
millions available.

Soft reweighting keeps the natural RAW sample (every bucket
contributes its natural share of a large uniform draw, so common
buckets keep their full effective coverage) and instead scales each
sample's **loss contribution** up or down by a per-bucket weight. Rare
buckets get more gradient signal per example without displacing common
bucket examples from the training set. This is a strictly smaller,
more surgical intervention than hard resampling, and it's cheap: same
generation pipeline, same RAW sampling script, one code change.

## Implementation sketch (not yet made)

`tools/train_nnue.py`'s `wdl_loss` (line 179) currently computes an
unweighted `.mean()` over the batch:

```python
def wdl_loss(raw, target):
    diff = torch.sigmoid(raw) - target
    return (diff * diff * (diff.abs() + 1e-8).sqrt()).mean()
```

The training loop already has each sample's output bucket available
(`bkt`, passed into `model(...)` at every call site — see lines 499,
501, 657) since the model's output head is itself bucket-indexed. The
minimal change is to accept an optional per-sample weight tensor and
use a weighted mean instead of a plain one:

```python
def wdl_loss(raw, target, weight=None):
    diff = torch.sigmoid(raw) - target
    per_sample = diff * diff * (diff.abs() + 1e-8).sqrt()
    if weight is None:
        return per_sample.mean()
    return (per_sample * weight).sum() / weight.sum()
```

with `weight = BUCKET_WEIGHT[bkt]` looked up from a fixed per-bucket
weight table at each call site. This is additive and backward
compatible — `weight=None` reproduces today's exact behavior bit for
bit, so it cannot regress the current default path.

## Choosing the weight table

Not tuned against held-out error before the fact — that would be
fitting the evaluation metric. Instead, derive it directly from the
measured pool imbalance in `docs/nnue-coverage-capacity-matrix-178m.md`
so the weight table is a predeclared function of known data, not a
free parameter search:

```
weight[bucket] = (total_pool_size / n_buckets) / pool_count[bucket]
```

i.e. inverse-frequency weighting, normalized so the average weight
across a natural RAW-sampled batch is 1 (leaves overall loss scale
roughly unchanged). Using the measured 27M-pool counts:

| Bucket | Pool count | Raw inverse weight | Notes |
|---|---|---|---|
| 0 | 60 | ~faceted, extreme (450,000x) | **excluded — clip, see below** |
| 1 | 12,656 | ~267x | |
| 2 | 288,201 | ~11.7x | |
| 3 | 1,680,871 | ~2.0x | |
| 4 | 3,871,670 | ~0.87x | |
| 5 | 5,833,163 | ~0.58x | |
| 6 | 7,577,822 | ~0.45x | |
| 7 | 7,735,557 | ~0.44x | |

Bucket 0's true inverse weight is absurd (60 records out of 27M) and
would let a handful of positions dominate the gradient — the same
failure mode as hard-duplication resampling, just moved into the loss
function instead of the sampler. **Clip all weights to a fixed band,
predeclared before running anything**: `[0.25x, 20x]`. This caps
bucket 0/1's influence while still giving bucket 1 (the strongest real
win in the hard-resampling run, −37.2%) a meaningful boost, and caps
how far buckets 6/7 can be *down*-weighted so they can't be starved the
way hard resampling implicitly starved them.

Clipped table to actually use:

| Bucket | Weight |
|---|---|
| 0 | 20.0 (clipped) |
| 1 | 20.0 (clipped) |
| 2 | 11.7 |
| 3 | 2.0 |
| 4 | 0.87 |
| 5 | 0.58 |
| 6 | 0.45 |
| 7 | 0.44 |

## Predeclared decision rule

Same shape as `13-nnue-ceiling.md`'s rule, adapted for a three-way
comparison (RAW baseline, hard-BALANCED from the prior run, and the new
soft-REWEIGHTED net), evaluated on the *same* held-out set
(`shard3`-derived, already built and reusable):

> Soft reweighting is preferred over hard resampling if it recovers at
> least half of hard-BALANCED's rare-stratum gain (buckets 1-4, judged
> per-bucket against RAW) while regressing the common strata (6, 7) by
> less than half of what hard-BALANCED cost them (i.e. under ~+5.5% and
> +7%, respectively, vs. hard-BALANCED's +10.9%/+13.9%).
>
> If soft reweighting matches or beats hard-BALANCED on rare strata
> AND does not regress common strata beyond RAW's own noise floor, that
> is a stronger result and should be reported as such rather than
> rounded down to "partial win."
>
> If soft reweighting regresses common strata by more than hard
> resampling did, or fails to move rare strata by at least half of hard
> resampling's gain, the mechanism itself (not just the resampling
> implementation) is likely not worth pursuing further at this scale,
> and the project should stop investigating piece-count-bucket
> reweighting.

## Setup for the actual run (once this design isn't being re-litigated)

- Training set: same 100,000-record RAW sample already used in the
  178M-corpus run (`sets/raw.bin`) — reused unchanged, not regenerated,
  so any difference in result is attributable to the loss-weighting
  change alone, not sampling variance.
- Held-out set: same `heldout.bin` (30,000 records from `shard3`),
  reused unchanged for direct comparability with both prior nets.
- Recipe: identical hyperparameters (early-stop patience 3, batch
  16384, 20-epoch cap, CPU) — only the loss function's weight argument
  differs from the RAW baseline run.
- Evaluation: `scripts/research/coverage_capacity_eval.py`, unchanged.

## What this is and isn't

This is a code change to `tools/train_nnue.py` (an added optional
`weight` argument, off by default), not a default-behavior change —
the existing unweighted call sites are untouched, so this cannot affect
the current shipped recipe until a deliberate follow-up call passes
`weight=`. No retrain of the shipped net, no cloud spend, no default
change. Running the actual experiment is the next step and should be
written up as its own finding doc, applying the rule above literally
once real numbers exist — not before.
