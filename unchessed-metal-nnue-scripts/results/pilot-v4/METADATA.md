# unchessed-metal-nnue-pilot-v4.bin

Real, working NNUE trained via unchessed-metal-nnue-scripts/ (the
audited-and-fixed pipeline), not a mockup or placeholder.

## What it is
- Wire format: UNCHNNUE version=4 (piece-count-bucketed 8-way output
  head), ft_in=22528, acc=256 -- same HalfKAv2_hm feature scheme as
  the currently-shipped default net, but with the bucketed head
  variant tools/train_nnue.py can also produce.
- Trained on 6,208 positions (after 2% internal validation split),
  early-stopped at epoch 9/12 (15-epoch cap), val-MAE 323.8cp on the
  internal validation slice.

## Real labels, real reference
- Labeled by the actual official Stockfish 19 Linux binary
  (official-stockfish/Stockfish release sf_19), depth 12, 1 thread,
  64MiB hash, real WDL (UCI_ShowWDL=true) -- not this project's own
  HCE self-labels like the currently-shipped default.
- Positions: 6,750 real quiet (ply>=16, not in check, every 8th ply)
  positions from 545 real, distinct games across this project's own
  PGN corpora (tools/classics.pgn + real SPRT self-play game
  archives) -- genuinely game-disjoint train/valid/test split
  (485/26/28 games respectively), not a random position-level split.

## Real held-out test result (never touched during training)
345 real Stockfish-19-labeled positions, inference run through the
ACTUAL Rust unchessed-adapter (EvalFile-loaded, real production
inference path, not a Python re-implementation):

    mae_cp=377.7  median=327.5  max=1597  positions=332 (ordinary)
    mate_mae_cp=29742 (13 mate-labeled positions, reported
      separately -- see unchessed-metal-nnue-scripts/evaluate_gates.py's
      fix for why these must never be averaged into the ordinary MAE)
    mae_gate_pass=false (does not clear the strict 100cp gate)

## Honest scope
This is a real pilot at real (if small) scale -- proof that the full
label_uci.py -> jsonl_to_shard.py -> train_nnue.py -> student_infer.py
-> evaluate_gates.py pipeline works end to end against the real
official Stockfish 19 binary, produces a real loadable net, and
measures it through the real Rust inference path. It is NOT a claim
of parity or even of beating the shipped default (148.7cp MAE on
Manus's own measurement) -- 6,208 training positions is roughly three
orders of magnitude smaller than what the shipped default trained on.
The next real step is the same pipeline at the corpus scale the
training procedure doc (docs/sfnnv16-training-procedure.md) already
specifies (millions of positions, ideally on the A100 host it
describes), not further pilot-scale tuning.

## Reproducing this run
All source data included in this bundle: train.jsonl/valid.jsonl/
test.jsonl (real Stockfish-19 labels with game_id for the split),
student_test.jsonl (the real Rust adapter's predictions on the test
set). Recipe: BATCH_SIZE=512 DEVICE=cpu EARLY_STOP_PATIENCE=3
EARLY_STOP_MIN_DELTA=0.1, tools/train_nnue.py <out> 15 <train+valid
shards concatenated>, matching docs/nnue-v4-training-recipe.md's
documented recipe family (batch size scaled down for this pilot's
much smaller corpus).
