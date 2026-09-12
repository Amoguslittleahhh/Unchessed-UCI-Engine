# hf-eval-full809m: full-corpus retrain after fixing the mate-label bug + OOMs

Second real attempt at training on the Lichess/chess-position-evaluations
corpus, after `hf-eval-284m` (SPRT-rejected at -178.1 +/- 52.3 Elo) traced
its loss to a flat +-30000cp mate-position label and two training-time
OOMs that had forced a 284M-record subset instead of the full corpus.
Both were fixed in commit 8e360a5 (graduated mate-distance scoring,
host-RAM `ascontiguousarray` fix, VRAM `randperm` fix, auto GPU-residency
fallback) -- this run trains on the corrected labels and the **full**
809,065,000-record train split, no subsetting.

## Infrastructure

Verda H200 SXM5 141GB VM (FIN-02, $4/hr), chosen specifically because the
full 809M-record train split needs ~86.6GB of VRAM (99 bytes/record data
+ 8 bytes/record train_idx) -- doesn't fit an 80GB A100 even with the OOM
fixes, but fits an H200 with ~55GB to spare. `BATCH_SIZE=262144` (2x the
A100 recipe) to use the extra headroom for higher throughput; all other
speed flags (TF32, bf16 AMP, fused Adam, cudnn.benchmark) at their
defaults. `torch.compile` deliberately left off (untested for this
model's EmbeddingBag ops, not worth the risk on a billed run).

Corpus was regenerated locally with the fixed `hf_eval_to_shard.py`
(verified: 138 distinct graduated mate-score values in a 2M-record
sample, not a flat +-30000) and transferred to the VM as just the 20
`*-train.bin` shards (79GB -- valid/test splits weren't needed since
`train_nnue.py` carves its own 200k-record held-out set internally).
Total round cost: **$8.17** (VM + volume, torn down immediately after).

## Real training numbers

15-epoch cap, early-stopped at epoch 7 (best epoch 4). **151s/epoch,
5.36M samples/sec** -- more than 2x the A100 run's 2.27M samples/sec,
confirming the H200 was the right call for a dataset this size. Full
training wall time: ~18 minutes.

Ordinary-position val-MAE (the metric that now correctly excludes the
mate-labeled slice, per this same commit's evaluate_iter split): best
**223.2cp**. Mate-position val-MAE stayed high (~28,700cp) as expected --
sigmoid(cp/400) saturates well before +-30000cp regardless of label
scheme, so mate positions are inherently hard to hit on this raw-cp
metric; this is not a regression, see train_nnue.py's evaluate_iter
docstring.

## Real SPRT gate result: REJECTED, but the fix mattered

vs the shipped `unchessed-nnue.bin` (v4), tc=10+0.1, elo0=0/elo1=10,
same WSL/cutechess harness as every other gate in this project:

| | hf-eval-284m (buggy mate labels) | hf-eval-full809m (fixed) |
|---|---|---|
| Elo vs shipped v4 | -178.1 +/- 52.3 | **-80.1 +/- 31.8** |
| LOS | 100% | 100% |
| Games | 197 decisive | 424 (LLR hit ubound at 2.96) |

The mate-label fix and full-corpus training roughly **halved the Elo
gap** -- a real, measurable improvement from the bug fixes, not noise.
But it's still a decisive loss to the shipped net despite a genuinely
good 223.2cp ordinary-position MAE.

## What this means

Two real attempts on this dataset (284M subset with a labeling bug, then
the full corrected 809M corpus) have both lost to the shipped net by a
wide margin, despite low held-out MAE on the dataset's own positions.
The remaining gap is very unlikely to be another labeling/OOM bug --
low MAE on Lichess's browser-triggered analysis positions doesn't
guarantee playing strength against real opponents, since that position
distribution (arbitrary positions users happened to analyze) differs
from the shipped net's training distribution (curated Stockfish
self-play / real game corpora actually representative of played chess).

**Do not promote this net.** Shipped `unchessed-nnue.bin` (v4) is
unchanged. This dataset is being set aside as a standalone eval-net
data source; further work in this direction (if any) should mix it with
existing game-derived corpora (e.g. `scale-v5`) rather than train on it
alone.
