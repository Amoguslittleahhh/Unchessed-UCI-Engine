# Unchessed NNUE handoff to Manus

Original work, self-contained: no third-party proprietary weights or
data. Safe to share and already committed to the repo.

## 1. The NNUE file

`unchessed-nnue.bin`, repo root, 23,071,768 bytes, magic `UNCHNNUE`.
Committed at `18a72ad` ("Add the trained NNUE weights and the scripts
that produced them"), currently shipped as default eval since `6c5431f`
("Ship NNUE v4 as the default eval, SPRT-validated: +26.1 +/- 12.4 Elo").

**One correction worth flagging up front**: despite the "v4" name in
commit messages/docs, the file's own on-disk header (read directly,
not inferred from docs) is wire-format **version 3**:

```
magic=UNCHNNUE version=3 ft_in=22528 acc=256
```

Version 3 is the *non-bucketed single output head* variant. Version 4
in `tools/train_nnue.py` refers to an 8-way piece-count-bucketed output
head experiment that the Rust runtime can also load (`out_buckets=8`
path exists in `nnue.rs`), but that is not what's in the shipped file.
"v4" in project naming = the *feature-scheme* fix (mirroring + own-king
+ factorized training), which shipped using the older non-bucketed
output head, not the bucketed one. Wanted to catch this before it
became a documentation mismatch on your end too.

## 2. Architecture description and layer dimensions

Two-perspective (STM + NSTM) linear feature transformer, no hidden
layers, no bucketing in the shipped file:

- Feature transformer: `[22528, 256]` weight matrix + `[256]` bias, one
  row per active feature, summed per perspective (incremental accumulator).
- Activation: SCReLU, `clamp(x, 0, 1)^2`, applied per perspective.
- Concatenate `[STM(256), NSTM(256)]` -> `[512]`.
- Output: single `Linear(512, 1)` (`out_w[512]`, `out_b[1]`), plain dot
  product, no hidden layers, no PSQT split, no piece-count bucket
  selection.
- Inference precision: **f32 throughout, not quantized**. No int8/int16
  fixed-point scaling anywhere in the shipped format -- this is a real,
  meaningful difference from real Stockfish's quantized nets and from
  the SFNNv16 diagram you're presumably calibrating against.

## 3. Feature-set definition

`HalfKAv2_hm`, ported to match `official-stockfish/nnue-pytorch`'s
`model/modules/features/halfka_v2_hm.py` and Stockfish's own
`src/nnue/features/half_ka_v2_hm.h` (verbatim algorithm, independently
reimplemented in Rust, not copied source):

- From perspective P: compute P's own king square in P's frame
  (vertical-flip s^56 if P is Black).
- Horizontal mirror: if that king square is on files a-d, flip the
  *whole board* (every square `s -> s^7`) so the king ends up on files
  e-h.
- Look up one of 32 king buckets for the (now always e-h) oriented king
  square (`KING_BUCKETS` table, identical values to Stockfish's).
- Non-king pieces (5 own + 5 opp) contribute
  `bucket*704 + p_idx*64 + sq`, `p_idx = piece_type*2 + (0 if own else 1)`
  (own/opp interleaved per type, matching Stockfish's `PieceSquareIndex`).
- Both kings share one 64-wide block at `p_idx=10` (features
  `[640,703]` within the bucket): opponent king contributes
  `bucket*704+640+opp_king_sq`; own king *additionally* contributes
  `bucket*704+640+own_king_sq` (same square used to pick the bucket) --
  always two distinct rows.
- `ft_in = 32 buckets * 704 (11 piece-types * 64 squares) = 22528`.

## 4. Quantization and activation details

No quantization -- plain `f32` weights and activations end to end (see
#2). Activation is `SCReLU = clamp(x, 0, 1)^2` on both perspective
accumulators before the output dot product. No ClippedReLU stage, no
squared-clipped-ReLU hidden layers, no fixed-point rescaling constants
(Stockfish's `OneV=258/128`, `QA`/`QB` scales -- none of that exists in
this format, since there's no quantization to calibrate).

## 5. Training command / configuration

`tools/train_nnue.py`, PyTorch, CPU or CUDA. Honest caveat first: **no
training log for this specific shipped file is in the repo**, so the
exact run (epoch count actually completed, exact machine) isn't
certified -- only the recipe family it came from is documented.

Recipe family (`docs/nnue-v4-training-recipe.md`):
- Optimiser: Adam, `lr=1e-3`, x0.3 decay at 60% and 80% of the epoch cap.
- Epoch cap: 15 (a ceiling, not a target -- best-val-MAE checkpoint is
  exported, not the last epoch).
- Early stop: patience 3 epochs, min-delta 0.1 cp val-MAE improvement.
- Batch size: 65536 (CPU/local) or 131072 (CUDA/cloud).
- Loss: `|sigmoid(raw) - target|^2.5`,
  `target = 0.7*sigmoid(cp/400) + 0.3*(wdl/2)`.
- Feature factorization during training only (main `[24576,256]` table
  + shared `[768,256]` "virtual" table for gradient density on
  sparse rows), coalesced into the single `[22528,256]` export table --
  export-time has no concept of factorization.
- Train/val split: random permutation, `n_val = min(200_000, max(1, n/50))`
  (2%, capped at 200k records) held out as validation; no separate
  third test split beyond that.

Command shape (`scripts/nnue-pipeline/train_recipe.sh` wraps this):
```
env OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 DEVICE=cpu BATCH_SIZE=65536 \
  EARLY_STOP_PATIENCE=3 EARLY_STOP_MIN_DELTA=0.1 \
  python3 tools/train_nnue.py <out.bin> 15 <shard0.bin> [shard1.bin ...]
```

## 6. Dataset provenance and train/validation/test split

Positions: real Lichess PGN dumps, replayed with this project's own
move generator (`unchessed-datagen`). Lichess's game database is under
Lichess's own open license for this kind of use, not proprietary.

Record format (104 bytes, `unchessed-datagen/src/main.rs`):
`12 x u64 LE bitboards` (side-to-move normalized, mover planes 0-5,
opponent 6-11) + `i16 LE search-score cp` (STM pov) + `u8 WDL`
(2=win/1=draw/0=loss, STM pov) + 5 padding bytes.

Shard location: `~/unchessed-ai/data/maia-data/nnue/shard*.bin` on the
training machine -- **not in this git repo** (108M positions x 104
bytes =~ 11 GB, too large to check in). Only the resulting weights file
and the pipeline that produces shards from PGNs are committed.

Split: see #5 (2%/200k-cap random holdout as validation; no distinct
test set on top of that documented anywhere in this pipeline).

## 7. Label source and reference version

**Not Stockfish-labeled.** Labels are self-generated: this project's
own hand-crafted eval (HCE) running a shallow, fixed-node search on
each quiet position (5000 nodes by default,
`unchessed-datagen/src/main.rs`'s `nnue_label_nodes()`). No external
engine, Stockfish version, or third-party evaluation was used to
produce the training labels -- the net is trained to imitate this
project's own HCE search output, not to match any Stockfish network's
output. Worth being explicit about this since your parity-with-SFNNv16
research direction might have assumed Stockfish-derived labels.

## 8. May the weights be committed?

Already committed (`18a72ad`, currently shipped as default at
`6c5431f`) and safe to share as-is -- original project work, Lichess
game positions (openly licensed for this), self-generated HCE labels,
no third-party proprietary weights or data anywhere in the pipeline.

## What this does NOT establish

Per your own note: this doesn't claim parity with anything. It's a
from-scratch net trained to imitate this project's own shallow HCE
search on Lichess positions -- useful for your load-correctness,
incremental-accumulator, and architecture/quantization-compatibility
checks, but the *reference* it was trained against is this project's
own evaluator, not Stockfish or SFNNv16, so a direct Stockfish-error
measurement will show whatever gap that implies rather than a
training-quality signal by itself.

Source: `unchessed-core/src/nnue.rs` (module doc comment + `TrainingRecord`
loader), `tools/train_nnue.py`, `unchessed-datagen/src/main.rs`,
`docs/nnue-v4-training-recipe.md`, current `main` at `918fccc`.
