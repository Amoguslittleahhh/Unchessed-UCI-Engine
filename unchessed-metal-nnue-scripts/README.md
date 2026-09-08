# Unchessed Metal NNUE Scripts

This directory contains reproducible, original orchestration scripts for the Lichess + LC0/Leela + Maia-3 + Stockfish research pipeline. It does not contain third-party source code, model weights, raw game archives, or generated labels. Keep those external and record their checksums in the manifests.

## Short procedure

```text
Lichess/LC0 position sources
  -> make_manifest.py
  -> split_manifest.py (game-disjoint)
  -> label_uci.py with Stockfish, LC0, and optional Maia-3 UCI engines
  -> train_portable.sh using the root original trainer
  -> evaluate_gates.py on an untouched test JSONL
```

Run the pilot first:

```bash
python3 unchessed-metal-nnue-scripts/make_manifest.py \
  --input positions.fen --out manifest.jsonl --source lichess --game-field none
python3 unchessed-metal-nnue-scripts/split_manifest.py \
  --input manifest.jsonl --out-dir splits --seed 42
python3 unchessed-metal-nnue-scripts/label_uci.py \
  --engine /path/to/stockfish --manifest splits/train.jsonl \
  --out labels/stockfish_train.jsonl --depth 12 --teacher stockfish
DEVICE=cuda BATCH_SIZE=131072 THREADS=16 \
  unchessed-metal-nnue-scripts/train_portable.sh \
  checkpoints/pilot.bin 15 /data/shards/shard*.bin
python3 unchessed-metal-nnue-scripts/evaluate_gates.py \
  --reference labels/stockfish_test.jsonl --student student_outputs.jsonl
```

## Target separation

Stockfish labels are the parity and tactical reference. LC0/Leela labels are strategic value/policy supervision. Maia-3 labels are rating-conditioned human policy supervision. Do not average their scalar values into one unlabeled target. Store each teacher in separate fields and select loss weights explicitly in the training configuration.

## Hardware policy

`train_portable.sh` delegates to the root `scripts/train_nnue_portable.sh`. GPU jobs default to a bounded 16 host threads; explicit CPU jobs use all logical threads by default. Set `THREADS` after profiling. Do not disable thermal protections or force unsafe power/clock settings.

## Provenance and licensing

Record the source URL, license, model name, commit or revision, checkpoint checksum, engine command, depth/nodes, options, and date in the run manifest. Maia-3 is AGPLv3 according to its official repository/model card; keep its process and weights separate unless the distribution and licensing obligations are intentionally handled.
