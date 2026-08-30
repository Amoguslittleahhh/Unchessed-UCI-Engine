# NNUE v4 full-scale training recipe (2026-08-30)

Answers the four questions in `docs/nnue-v4-retrain-data-scaling-finding.md`.
No net was trained at 108M or 178M this round — those shards are not in
git, and this sandbox is 2 vCPU / 3.8 GB / no GPU. The recipe is what
must be true before any cloud spend; the 108M point is still a local
reviewer-machine job.

## 1. What epoch/step count is appropriate at 178M?

**Do not copy 8 or 15.** Treat the CLI epoch argument as a *cap*. Stop
when val-MAE has not improved by ≥0.1 cp for 3 epochs, and export the
**best** checkpoint, not the last.

That is the F-01/F-03 failure mode from
`docs/full-scale-bug-audit-2026-08-21.md` applied to this trainer: a
fixed epoch count independent of data size, plus no protection against
worsening validation burning the rest of the run. The three diagnostic
SPRT nets all exported a *worse* checkpoint than the run had already
seen:

| Unique positions | Cap | Final val-MAE | Best val-MAE in the same run |
|---|---|---|---|
| 959,102 | 20 | 83.6 cp | 57.4 cp |
| 9,000,000 | 8 | 59.3 cp | 55.3 cp |
| 27,000,000 | 8 | 54.3 cp | 51.1 cp |

Val-MAE was climbing in every run, including the healthiest (27M). More
epochs at those scales would have made the exported net *worse*. The
right lever is "stop after the minimum", not "train longer".

`tools/train_nnue.py` now does this (defaults: `EARLY_STOP_PATIENCE=3`,
`EARLY_STOP_MIN_DELTA=0.1`; patience `0` disables stop but still exports
best). The production 60%/80% LR step-decay is unchanged and is still
keyed off the *cap*, not the realised length: if early-stop fires
before the first drop, LR never drops, which is correct — we already
started overfitting at the high LR.

A 15-epoch cap at 178M is ~2.67B samples seen at batch 65536, vs ~1.62B
for the 108M × 15 production shape. That cap is a ceiling to match the
shipped launcher, not a target. Early-stop decides the actual length.

nnue-pytorch is *not* the bar to copy. Its "epoch" is a fixed 100M-position
superbatch (`--epoch-size=100000000`) with `--max_epochs=400`, cyclic
reads, and they **save intermediates and pick the best by playing games**
(Theoria's public writeup: 150 epochs trained, epoch 139 selected). Our
8 epochs of 27M is 216M samples seen ≈ two of their epochs. Matching
Stockfish's 400-superbatch schedule is a different project.

## 2. Fourth local data point (108M) — not run here

Cannot. The 12 original self-play shards live on the reviewer's disk
(`~/unchessed-ai/data/maia-data/nnue/shard*.bin`); they are not in this
repo (108M × 104 B ≈ 11 GB). This sandbox also cannot hold them in the
3.8 GB RAM the host-resident trainer needs.

The 150–250 Elo extrapolation in the finding assumed the *same cheap
8-epoch last-checkpoint recipe* at 178M. That forecast does not transfer
to this recipe, and it was already labelled a back-of-envelope. Do not
budget cloud spend against it.

The 108M run is still the right next measurement. Command:
`scripts/nnue-pipeline/train_recipe.sh` (15-epoch cap, early-stop 3,
best-ckpt, `BATCH_SIZE=65536`). Then SPRT with the same harness as
`scripts/research/wsl_sprt_nnue_real27m.sh`.

## 3. Shipped default's own recipe — recovered from the launchers

No training log for the specific shipped `unchessed-nnue.bin` is in the
repo, so we cannot certify that *that file* was a 15-epoch run versus an
earlier ad-hoc one. What *is* committed is the pipeline that produces
nets in this family:

| | Local WSL (`scripts/nnue-pipeline/full_pipeline.sh`) | A100 (`full_pipeline_cloud.sh`) |
|---|---|---|
| Epoch cap | **15** | **15** |
| Batch | **65536** | **131072** |
| Device | cpu, 14 threads | cuda, GPU-resident |
| Optimiser | Adam `lr=1e-3`, ×0.3 at 60% and 80% of the cap | same |
| Trainer | `tools/train_nnue.py` | same |
| Corpus | combined shards in `~/unchessed-ai/data/maia-data/nnue/` (originally 108M; monthly add-ons target ~178M, hard cap 200M local / 500M cloud) | same shards |

The v3 research brief (`nnue-shards-safe/v3_research_brief.md`) used the
same Adam / 1e-3 / 60–80% schedule; its failed 15-epoch SPRT was a
different *architecture*, not a different optimiser. Diagnostic runs
used the trainer default `BATCH_SIZE=16384` (4× smaller than the CPU
production batch) and 8 epochs — two more recipe mismatches on top of
data volume.

## 4. Go / no-go

**NO-GO for cloud 178M spend.** The three diagnostic SPRTs proved data
volume is load-bearing and that the 8-bucket head is not the cause.
They did **not** validate a production recipe: last-checkpoint export,
8-epoch cap, and batch 16384 are all different from the shipped
launcher. Spending money on 1.65× more unique positions with the same
broken export path is the thing the finding asked us not to do.

**GO for a local 108M run with this recipe**, on the machine that
already has the shards:

```
cd <repo>
env OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 DEVICE=cpu BATCH_SIZE=65536 \
  EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 \
  python3 tools/train_nnue.py nnue_v4_108m.bin 15 \
    ~/unchessed-ai/data/maia-data/nnue/shard{0..11}.bin
```

(`scripts/nnue-pipeline/train_recipe.sh` wraps that.) Then SPRT vs
`unchessed-nnue.bin` at `tc=10+0.1`, `elo0=0 elo1=10`, same as the
diagnostics.

Decision tree after that SPRT — no Elo number invented here:

- **Still >100 Elo behind the default.** 178M is 1.65× the unique
  positions of 108M. The measured 9M→27M step (3× unique) only recovered
  76 Elo under a *worse* recipe. Do not spend cloud on 178M expecting it
  to close a triple-digit gap. Something else has to change (quiet-filter
  regen of the 108M shards, a different labeler, a longer *unique*-data
  collection), and that is a new finding, not this recipe.
- **Within ~50 Elo, or positive.** Then the 178M A100 run with the same
  recipe (`DEVICE=cuda BATCH_SIZE=131072`, 15-epoch cap, early-stop 3)
  is the justified next spend. Any resulting net still needs a fresh
  SPRT before it can touch the default evaluation.

Not in this recipe, on purpose: weight decay, random-FEN-skipping,
changing the loss, changing the 8-bucket head. No evidence from the
three diagnostic runs supports adding them, and the bucket-vs-single-head
ablation already says the head is not the problem.

`UnarchitecturedHint` stays default-off. No search integration.
