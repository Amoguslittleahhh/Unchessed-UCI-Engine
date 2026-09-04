# NNUE v4: 108M-record recipe-validated SPRT (2026-09-01)

Reviewer-run (not arena), on a rented Verda `CPU.32V.128G` box (FIN-03,
$0.384/hr, ~2.5 hours total, VM + volume deleted immediately after).
Closes the ask in `docs/nnue-v4-training-recipe.md`: a local 108M SPRT
with the defended recipe, as the gate before any 178M cloud decision.

## What was run

All 12 original self-play shards (108,000,000 records, the real corpus
behind the documented "108M" figure) through the round-13 recipe:
15-epoch cap, early-stop patience 3 / min-delta 0.1cp, best-checkpoint
export, batch 65536, Adam 1e-3 with the 60/80% step-decay.

Early-stopped at epoch 6; best checkpoint epoch 3, **val-MAE 47.8cp** —
the lowest in this entire investigation, and in line with the IEEE
analysis's own forecast (~49.6cp) from last round.

SPRT vs the shipped default (`unchessed-nnue.bin`, v3), same harness as
every prior round (`tc=10+0.1`, `elo0=0 elo1=10`, real hardware, WSL):
**114-35-39 over 188 completed games, Elo difference −155.6 ± 47.7.**
(`scripts/research/wsl_sprt_nnue_108m.sh`.)

## Full trend, all real SPRTs this investigation

| Positions | Recipe | Best val-MAE | SPRT vs shipped net |
|---|---|---|---|
| 959,102 | old (last-epoch export) | 57.4cp | **−796.5 Elo** |
| 9,000,000 | old | 55.3cp | **−383.5 Elo** |
| 27,000,000 | old | 51.1cp | **−307.1 Elo** |
| 27,000,000 | **new** (best-ckpt, early-stop, batch 65536) | 49.3cp | **−244.7 Elo** |
| 108,000,000 | new | **47.8cp** | **−155.6 ± 47.7 Elo** |

Both levers (recipe fix, data volume) are independently real and
additive: the recipe fix alone bought ~62 Elo at fixed 27M data; going
27M→108M under the fixed recipe bought another ~89 Elo.

## Go/no-go on cloud 178M, applying round 13's own pre-committed rule

Round 13 stated the decision tree in advance, specifically so this
result couldn't be rationalized after the fact:

> Still >100 Elo behind the default. ... Do not spend cloud on 178M
> expecting it to close a triple-digit gap.
> Within ~50 Elo, or positive. Then the 178M A100 run ... is the
> justified next spend.

Even the optimistic end of this result's confidence interval
(−155.6 + 47.7 = **−107.9 Elo**) is still over 100 Elo behind. **By the
project's own pre-agreed rule, cloud 178M remains NO-GO.** This is a
clean, honest application of a rule set before the result was known —
not a post-hoc rationalization in either direction.

## Why more data alone probably won't close this

Last round's IEEE-style analysis (`docs/ieee-low-cp-val-mae-and-persona.md`)
modeled the current 5000-node HCE labeling process as having a Bayes
noise floor around 50-56cp. This net's best val-MAE (47.8cp) is now
*below* that estimated floor, in the same neighborhood as the forecast
for 178M (48.7cp) and even 500M (46.9cp) positions under the same
labels. If that model is right, **the remaining ~108-156 Elo gap is
probably not a data-volume problem anymore** — it's some combination of
label quality (HCE search depth), architecture capacity, or something
not yet isolated. More of the same labels at 178M is unlikely to move
the number much, which is exactly why round 13's rule (require getting
within ~50 Elo before funding it) already protects against wasting that
spend.

## What's actually next

Not cloud spend. Candidates, unordered, for whoever picks this back up:

1. **Stronger labels**: deeper HCE search, or self-distillation from the
   shipped net at higher node count, for the *existing* 108-178M corpus
   — cheaper than more raw self-play volume if the noise-floor model is
   right.
2. **Isolate label noise from architecture/capacity** before assuming
   either one. The three-point SPRT ladder here never varied
   architecture; the ablation in the original finding only checked
   8-bucket vs single-head at tiny scale, not at 108M.
3. **A different question entirely**: is −107 to −156 Elo actually
   close enough to be worth the engineering cost of chasing further, or
   is the shipped v3 net simply near a local optimum for this labeling
   pipeline? That is a real, honest question this data supports asking.

No net from this investigation is being proposed for shipping. The
shipped `unchessed-nnue.bin` (v3) remains the default; nothing about the
default evaluation path changed at any point in this investigation.
