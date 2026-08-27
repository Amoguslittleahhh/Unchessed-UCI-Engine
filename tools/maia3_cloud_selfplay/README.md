# Maia-3 self-play at scale (cloud / Verda AI)

Generates **2,000,000 games** of Maia-3 vs Maia-3, each side's UCI elo
drawn independently and uniformly as an integer from **100-3200
(1-elo accuracy)** from one seed, ready for immediate level-conditioned
training. This is the many-core scaling of the verified pipeline in
`tools/selfplay_elo_mixer.py` — same `Maia3` inference class, same
4352-move indexing, same temperature-1 sampling — re-orchestrated with
per-game deterministic substreams, per-worker shards, fsync'd
checkpoints (resume-safe), and a built-in full validation +
conditioning-calibration pass.

## What you get

```
<out>/pgn/shard-NNNNN.pgn        PGN; WhiteElo/BlackElo = the drawn limits
<out>/labels/shard-NNNNN.jsonl   one row per move:
                                 {game, elo_white, elo_black, fen,
                                  move_uci, move_ply, side, elo_self,
                                  elo_oppo, top1_prob, ldw}
<out>/progress/shard-NNNNN.json  checkpoints (delete after success)
<out>/manifest.json              seed, ranges, per-shard sha256, model
                                 provenance (file sha256 + source commit),
                                 backend, worker layout
<out>/calibration.json           full-replay validation result +
                                 mean top1 probability per 100-elo band
                                 (the conditioning check)
```

~2,000,000 games × ~65 plies ≈ **131M label rows ≈ 31-35 GB** + ~1.2 GB
PGN. Provision **≥ 100 GB** of storage. The label schema is identical to
`data/selfplay/maia3-100-3200-labels.jsonl`, so the level-conditioned
pipeline (`tools/build_level_conditioned_moves.py`) consumes it without
changes.

## Reproducibility contract

- **Elo plan**: `rng = random.Random(seed)`; per game, `randint(elo_min,
  elo_max)` for White then Black, in game order. The plan's game 0 is
  anchored to the committed reference set `data/selfplay/` (seed 42:
  both streams are fresh `Random(42)` before their first two draws —
  tested). Later pairs deliberately differ: the reference pipeline
  interleaved move sampling into the same stream, while this generator
  uses independent per-game substreams (documented layout change, same
  uniform integer 100..3200 distribution).
- **Move sampling**: each game g samples from its own substream
  `random.Random(sha256(f"{seed}:{g}"))` — games are independent, so any
  subset can be regenerated or resumed exactly. (The small committed
  reference used one global stream; this layout is what makes sharding
  at 2M games possible — same model, same sampling mechanism,
  documented stream.)
- **Reproducibility across backends**: use one backend for a given set.
  CPU inference is byte-deterministic across machines; CPU vs GPU can
  flip a move only in near-tie samples (documented, not a defect).

## Run on a Verda AI instance

Verda gives you bare CUDA-equipped VMs with `verda` CLI / Terraform /
web UI (docs: https://docs.verda.com). Flow:

1. **Create the instance** (from your laptop, `verda` CLI installed):

   ```sh
   # one H100, 32 cores, 100 GB volume (~$3.25/h on-demand, ~$1.14-1.63 spot)
   verda vm create --kind gpu \
       --instance-type 1H100.80S.32V \
       --os ubuntu-24.04-cuda-12.8-open-docker \
       --os-volume-size 100 \
       --hostname maia3-2m
   # faster: 4H100.80S.176V (176 cores, ~$9-13/h) or 8H100.80S.176V
   # budget: 1V100.6V (~$0.17/h, CPU generation is viable)
   ```

2. **SSH in and set up** (GitHub must be reachable from the instance —
   Verda instances have normal internet):

   ```sh
   verda ssh maia3-2m
   git clone <your-remote>/Unchessed-UCI-Engine.git
   cd Unchessed-UCI-Engine
   python3 -m venv venv
   venv/bin/pip install -r tools/maia3_cloud_selfplay/requirements.txt
   # GPU (recommended on a CUDA image):
   venv/bin/pip uninstall -y onnxruntime
   venv/bin/pip install onnxruntime-gpu        # match the image CUDA
   ```

3. **Stage the model** (pinned mirror, 45.7 MB):

   ```sh
   venv/bin/python tools/selfplay_elo_mixer.py fetch-model --out /tmp/maia3-onnx
   ```

4. **Generate** (2M games). Two sensible layouts:

   ```sh
   # GPU path (4 GPUs, one worker per GPU):
   venv/bin/python tools/maia3_cloud_selfplay/generate.py \
       --model /tmp/maia3-onnx/simple-maia3-inference/simple_maia3_inference/maia3_simplified.onnx \
       --out /data/maia3-2m --games 2000000 --seed 20260827 --gpus 4

   # CPU path (uses all cores; fully byte-reproducible):
   venv/bin/python tools/maia3_cloud_selfplay/generate.py \
       --model ... --out /data/maia3-2m --games 2000000 --seed 20260827 \
       --workers 30
   ```

   The command **validates the whole set when it finishes** (every move
   replayed legal, headers/labels cross-checked, conditioning gradient
   reported to `calibration.json`) and exits non-zero on any hard
   failure. If the instance is killed mid-run, re-run with `--resume`.

5. **Ship the data back** (rsync/scp from the instance, or point your
   training job at the instance volume if your stack supports it).

## Expected duration and cost (honest, anchored)

Anchor measured in this repo's 2-core sandbox: ~13.5 positions/s per
process (CPU), ~65 plies/game.

| Layout | Throughput (est.) | 2M games | Cost (est.) |
|---|---|---|---|
| 1×H100.80S.32V, CPU (30 workers) | ~8-12 games/s | ~5-8 h | ~$20-30 |
| 1×H100, `--gpus 1` | ~2-5 games/s | ~12-24 h | ~$40-80 |
| 4×H100.80S.176V, CPU (170 workers) | ~40-60 games/s | ~1-1.5 h | ~$12-20 |
| 4×H100, `--gpus 4` | ~8-20 games/s | ~3-8 h | ~$40-100 |

(GPU throughput on this 5M-param model is dominated by the python-chess
position overhead, so a big CPU instance is often the sweet spot and is
the byte-reproducible option; GPU wins once you add more workers per
GPU or bigger models.)

## Calibration and tests

- `calibration.json` reports mean top-1 probability per 100-elo band.
  The committed 200-game reference measures **0.323 at 100-199 rising
  to ~0.52-0.62 at 1900-3200**; a 2M set should reproduce the shape
  (the validation pass asserts low-end < high-end concentration and
  fails otherwise).
- `tools/test_maia3_cloud_selfplay.py` (runnable in the sandbox, no
  cloud needed): plan determinism + exact match of the first 200 elo
  pairs against the committed reference, model anchor outputs
  (startpos top-1 0.6165 @1500/1500, 0.1666 @200/3000 ±0.002), and a
  4-worker end-to-end mini-run (8 games) through generate → validate.

## Honest limits

- Same model caveats as `data/selfplay/README.md`: "simplified"
  single-position export (no history/clock inputs), sampling without
  the official UCI's one-ply ranking; model-generated, not human.
- 2M games ≈ 131M rows is a lot of disk; shard it into object storage
  or your training volume early.
- Spot instances can be preempted — `--resume` is the safety net;
  checkpoints are fsync'd per game.
