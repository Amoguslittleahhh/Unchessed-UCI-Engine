# NNUE v4 retrain: data-scaling root cause (2026-08-30)

**Round 13 answer:** `docs/nnue-v4-training-recipe.md` — early-stop +
best-checkpoint export; shipped launcher recipe recovered; cloud 178M is
NO-GO until a local 108M SPRT with that recipe exists.

Reviewer-run (not arena), following up on round 12's shipped 8-bucket v4
format and quiet-filtered dataset tooling. Round 12 built the format and
tooling but never trained on real data or SPRT'd a result. This is that
follow-up, run entirely on local/already-present hardware — no cloud spend.

## What was tested

Trained the v4 format (8 piece-count output buckets, quiet-filtered
targets) at three data scales, using `tools/train_nnue.py` unmodified
except for record count, and SPRT'd each resulting net against the
current shipped default (`unchessed-nnue.bin`, v3 format) at
`tc=10+0.1`, `elo0=0 elo1=10`, on real hardware (WSL, 14-core):

| Positions | Source | Epochs | Final val-MAE | SPRT vs. default |
|---|---|---|---|---|
| 959,102 | committed git corpus (49 human-game PGNs), quiet-filtered | 20 | 83.6cp (climbing from 57.4cp) | **−796.5 Elo** (98-1-0) |
| 9,000,000 | 1 real self-play shard | 8 | 59.3cp (climbing from 55.3cp) | **−383.5 Elo** (92-7-7) |
| 27,000,000 | 3 real self-play shards | 8 | 54.3cp (climbing from 51.1cp) | **−307.1 Elo** (97-12-11) |

The committed git corpus (used for round 12's own quiet-filter
measurements) is **0.54% the size of the real self-play corpus** that
trained the shipped default — confirmed on disk: the original self-play
shard set totals exactly 108,000,000 records (matching the documented
figure), with 5 more monthly shards bringing the current total to
178,000,000.

## Root cause, isolated

Ran a controlled ablation to separate "is it the 8-bucket head" from "is
it the dataset size": retrained a single-shared-head (v3-style,
`N_OUT_BUCKETS=1`) net on the *exact same* 959,102-position dataset, same
20 epochs. Result: val-loss 0.012229, val-MAE 86.7cp — statistically the
same pathological curve as the 8-bucket run (val-loss 0.012471, val-MAE
83.6cp), not meaningfully better. **The bucket-head change is not the
cause.** Both the 8-bucket format and the quiet-position filters are
structurally sound; round 12's tests (110/110 passing, ABI cross-check
exact) correctly validated the code. What round 12 never did was train on
enough real data to draw a strength conclusion.

## The scaling trend is diminishing returns, not linear

Plotting Elo gap against log(positions):

- 940k → 9M (10x data): gap fell 796.5 → 383.5 (−413 Elo)
- 9M → 27M (3x data): gap fell 383.5 → 307.1 (−76 Elo)

The slope per log-unit roughly halved between the two steps. Naively
extrapolating that decaying trend to the full 178M corpus (another ~6.6x
from 27M) lands around a **150-250 Elo gap remaining** — not full parity
with the shipped default. This is the reviewer's own back-of-envelope
extrapolation, not a measured result; it could be wrong in either
direction (data quality effects, LR-schedule interactions, or the
decay rate itself changing at larger scale are all real possibilities).

## What's being asked of arena

All three diagnostic runs above deliberately used a cheap, fixed 8-epoch
schedule (20 for the first) purely to get a fast directional read, not
because that's a considered training recipe. The honest conclusion is:
**data volume is real and load-bearing, but "point the existing recipe at
more shards" is not yet a validated production recipe.** Before this
project spends real cloud money on a full-178M-position training run, we
want that recipe actually worked out and defended, specifically:

1. **What epoch/step count is actually appropriate at 178M scale?** Fixed
   epoch count is the wrong lever once data volume changes by 2+ orders
   of magnitude (round 8's audit doc, `docs/full-scale-bug-audit-2026-08-21.md`
   F-01, already flags exactly this failure mode for a different
   training pipeline — the same reasoning likely applies here). Determine
   this from validation-loss convergence behavior, not a copied constant.
2. **Does the diminishing-returns trend above hold, worsen, or reverse**
   at a fourth, larger local data point (e.g. all 12 original shards,
   108M) before committing to the full 178M + a proper epoch schedule?
   This is still free/local — no need to wait for cloud access to get one
   more real point on the curve.
3. **Is the current shipped default's own training recipe documented
   anywhere** (epoch count, LR schedule, exact corpus used)? If it's
   available, use it as the reference point instead of re-deriving one
   from scratch — the goal is a recipe that's at least as good as
   whatever produced the thing already in production, not merely "more
   data than round 12's diagnostic."
4. **A real go/no-go recommendation**: either a concrete training recipe
   (corpus size, epoch/step count, LR schedule) with a stated expected
   outcome, or an honest "the full corpus likely isn't enough to beat the
   shipped default without also changing X" if that's what the further
   local scaling point suggests. No cloud spend should happen until this
   recipe is written down and defended with evidence, per this project's
   standing discipline on real-money decisions.

## Reproducing this

- `scripts/nnue-pipeline/local_regen.sh` — regenerates the quiet-filtered
  committed-corpus dataset (the 959k-position run).
- `scripts/research/wsl_sprt_nnue_v4.sh`, `wsl_sprt_nnue_real9m.sh`,
  `wsl_sprt_nnue_real27m.sh` — the three SPRT configs above, assuming the
  same WSL layout as prior rounds (`~/unchessed-sprt-build`,
  `~/unchessed-ai/data/cutechess`, `~/unchessed-ai/data/maia-data`).
- Real self-play shards used: `~/unchessed-ai/data/maia-data/nnue/shard{0,1,2}.bin`.

## Honest negatives

- No net from this round is being proposed for shipping. All three are
  confirmed worse than the current default, some far worse.
- The 150-250 Elo extrapolation is a rough estimate from two slope
  measurements, not a prediction to plan a budget around.
- 8 epochs was chosen for turnaround speed, not because it's known to be
  sufficient for any of these data scales — the rising val-MAE within
  each individual run (even the healthiest, 27M) suggests none of these
  three runs has actually converged yet either.
