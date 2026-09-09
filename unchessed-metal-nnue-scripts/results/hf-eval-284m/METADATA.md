# hf-eval-284m: real A100 GPU-trained net (partial corpus, mate-labeling bug)

Real training run on a Verda A100 80GB VM (`a100-nnue-train`, FIN-03,
$1.79/hr, ~1hr wall time including two real OOM crashes and their fixes,
destroyed immediately after this run).

## Data

284,352,787 real positions (files f0000-f0007 of the 20-file
`local-data/hf-chess-position-evaluations/` corpus -- see that
directory's own METADATA.md for full provenance), out of the full
809M-record corpus. Cut down from an original attempt at the full 809M,
then 500M, due to two real OOM failures:

1. Host RAM OOM at 121GB RSS training on all 809M records -- the
   `np.ascontiguousarray()` step in `train_nnue.py`'s GPU-resident
   loader creates a large transient copy of the bitboard column on top
   of the already-loaded data.
2. CUDA VRAM OOM at ~72.8GB/79.25GB training on a 500M-record subset --
   `torch.randperm` for the epoch shuffle needed more headroom than was
   left after the dataset itself occupied most of the A100's 80GB.

284M records (8 of 20 files) left enough headroom on both host RAM and
VRAM to train cleanly.

## Real training numbers

15/15 epochs, no early-stop (LR decay at epochs 10 and 13 kept
squeezing out real >=0.1cp improvements). ~124-125s/epoch, 2.27M
samples/sec, ~31 minutes total wall time. `SAFE_MAX_RECORDS` in
`tools/nnue_cloud_runtime.py` raised from 500M to 1B for this run (see
that file's own commit).

Raw val-MAE reported by the trainer: 3554.0cp -- **misleading**, caused
by a real bug (below), not the actual eval quality.

## Real bug found: mate-score labeling

`unchessed-metal-nnue-scripts/hf_eval_to_shard.py` assigns mate
positions a flat +-30000cp regardless of actual mate distance. Verified
via a real post-training eval through the actual Rust inference path
(`unchessed-adapter` + `EvalFile`, not a Python reimplementation),
splitting the held-out `f0000-valid.bin` by `abs(score) >= 29000`:

| bucket | n | real MAE |
|---|---|---|
| ordinary | 4,302 | **175.6cp** |
| mate (flat +-30000 label) | 698 (~14%) | 28,391.7cp |

The **ordinary-position MAE (175.6cp) is the best result of this whole
project** -- better than both earlier CPU pilots (223.9cp on
2018-01+2026-01, 377.7cp on the original 6,750-position pilot). The
raw blended val-MAE the trainer reported is dominated by the ~14% of
positions with a broken mate label, not a reflection of overall net
quality.

## What's NOT done here

- Mate-score labeling is not fixed. Either exclude mate positions from
  training (matching the original Stockfish-teacher pipeline's
  convention) or use a graduated distance-to-mate score, then retrain.
- Only 284M of 809M records used. The host-RAM and VRAM OOM causes are
  understood but not fixed in `train_nnue.py` itself (no host-resident
  streaming loader exists yet for datasets this large on this class of
  hardware).
- No SPRT / real-game validation against this net yet -- 175.6cp
  ordinary MAE is a promising real signal, not a proven Elo gain.
