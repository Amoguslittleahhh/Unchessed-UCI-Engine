# Mixed-engine self-play at scale (cloud / Verda AI)

Generates **5,000,000 games** (default) from a pool of four engines,
each side of every game drawn independently as (engine, elo) — engine
uniform over the pool, elo a uniform integer inside the range that
engine's strength mechanism supports. Ready for immediate
level-conditioned training. This is the many-core scaling of the
verified pipeline in `tools/selfplay_elo_mixer.py` — same `Maia3`
inference class, same 4352-move indexing, same temperature-1 sampling
— re-orchestrated with per-game deterministic substreams, resident
per-worker engine pools, per-worker shards, batched fsync'd
checkpoints (resume-safe), and a built-in full validation +
conditioning-calibration pass.

## The engine pool and what "elo" means for each

Each engine gets the strength mechanism it **actually** supports
(verified from source and, where the sandbox's egress allowed, from a
live binary):

| Engine | Version (pinned) | Strength lever | Elo range | Label quality |
|---|---|---|---|---|
| Maia-3 | official platform ONNX (pinned commit) | **native UCI_Elo model conditioning** — the elo input is a real, measured part of the model (0.323 → 0.52-0.62 top-1 gradient) | 100-3200 | `calibrated` |
| Stockfish | sf_18 (cb3d4ee9) | **native UCI_Elo** (Skill mechanism; min 1320 / max 3190 verified in `sf/src/search.h` and live `uci` output) + fixed 100 ms movetime | 1320-3190 | `native` |
| LC0 | v0.32.1 (fd71a2d9) | **thinking-budget ladder** — LC0 has *no* UCI_Elo option (verified in v0.32.1 source); movetime 2→500 ms monotone in target elo | 600-3200 | `approximate` |
| RubiChess | master 2026-08-27 (29ddf13d) | **LimitNps ladder** — RubiChess has *no* UCI_Elo option (verified in its option list); NPS cap 100→uncapped monotone in target elo, 250 ms movetime for the cap to bind | 600-3200 | `approximate` |

Games played by LC0/RubiChess carry `WhiteEloQuality`/`BlackEloQuality
= "approximate"` in the PGN and `elo_quality: "approximate"` in the
label rows: their elo values are **monotone in strength but
uncalibrated in exact value** (the lever ordering is guaranteed, the
number on the label is a documented approximation). Downstream
level-conditioned training can filter or down-weight by `elo_quality`
(e.g. use `calibrated`+`native` rows for the conditioning signal,
everything for move diversity).

## What you get

```
<out>/pgn/shard-NNNNN.pgn        PGN; WhiteElo/BlackElo + WhiteEngine/
                                 BlackEngine + WhiteEloQuality/
                                 BlackEloQuality per game
<out>/labels/shard-NNNNN.jsonl   one row per move:
                                 {game, elo_white, elo_black, fen,
                                  move_uci, move_ply, side, elo_self,
                                  elo_oppo, engine, elo_quality,
                                  top1_prob, ldw}
                                 (top1_prob/ldw are null on
                                  non-Maia-3 moves)
<out>/progress/shard-NNNNN.json  checkpoints (delete after success)
<out>/manifest.json              seed, ranges, per-shard sha256, full
                                 engine provenance (pinned git refs,
                                 binary/net sha256, engine id strings)
<out>/calibration.json           full-replay validation result,
                                 per-engine stats, mean top1 per 100-elo
                                 band over the Maia-3 rows
```

5,000,000 games × ~65 plies (measured on the committed 200-game
reference) ≈ **327M label rows ≈ 85-90 GB**. Provision **≥ 150 GB** of
storage. The Maia-3 rows keep the schema of
`data/selfplay/maia3-100-3200-labels.jsonl`, so the level-conditioned
pipeline (`tools/build_level_conditioned_moves.py`) consumes the PGN
without changes.

## Reproducibility contract

- **Elo plan**: `rng = random.Random(seed)`; per game, per side:
  `choice(engine)` then `randint(engine_lo, engine_hi)` — the
  single-engine pool (`--engines maia3`) skips the choice, making its
  stream **byte-identical to the original 2M generator's `plan_elo`**
  (tested). Game 0 of the maia3-only stream is anchored to the
  committed reference set `data/selfplay/` (seed 42 — tested).
- **Move sampling**: Maia-3 moves sample from each game's own
  substream `random.Random(sha256(f"{seed}:{g}"))` — games are
  independent, so any subset can be regenerated or resumed exactly.
  UCI-engine moves are the deterministic `bestmove` of a fixed
  movetime budget (same seed ⇒ same moves, same binaries).
- **Reproducibility across backends**: use one backend per set. CPU
  inference is byte-deterministic across machines; CPU vs GPU can
  flip a move only in near-tie samples (documented, not a defect).

## Run on a Verda AI instance

Verda gives you bare VMs with `verda` CLI / Terraform / web UI
(docs: https://docs.verda.com). Flow:

1. **Create the instance** — the picked shape is the **CPU Node,
   180x vCPU / 720 GB RAM** (instance `CPU.180V.720G`, FIN-03,
   **$2.160/h** = $0.012/vCPU/h): web UI → *Compute type: CPU
   instance* → *CPU Node* → **180x**, or CLI:

   ```sh
   verda vm create --kind cpu \
       --instance-type CPU.180V.720G \
       --os ubuntu-24.04 \
       --os-volume-size 150 \
       --hostname mixed-5m
   ```

   (check `verda vm create --help` for the exact CPU flag names on
   your CLI version; the UI path above is unambiguous)

   Purchase checklist (the compute-configuration screen does not show
   all of these):

   * **Storage >= 150 GB** — the run writes ~87 GB of shards plus the
     engine sources/nets (~5 GB) and the venv; confirm the volume on
     the storage step.
   * **On-demand, not spot** — the job costs ~$200-240 total; a spot
     preemption kills the instance *and* its local disk, wiping the
     `--resume` checkpoints.
   * **OS: any Ubuntu** (22.04/24.04) — CPU mode needs no CUDA image
     and no Docker.
   * **RAM is over-provisioned** (720 GB; the mixed pool keeps ~8
     engine processes + one ONNX session per worker, ~90 workers ≈
     720 processes, tens to a few hundred GB depending on the LC0
     net) — no action needed.

2. **SSH in and set up** (GitHub must be reachable from the instance —
   Verda instances have normal internet):

   ```sh
   verda ssh mixed-5m
   git clone <your-remote>/Unchessed-UCI-Engine.git
   cd Unchessed-UCI-Engine
   python3 -m venv venv
   venv/bin/pip install -r tools/maia3_cloud_selfplay/requirements.txt
   # build toolchain for fetch-engines (LC0 needs meson+ninja):
   venv/bin/pip install meson ninja
   ```

3. **Stage the Maia-3 model** (pinned mirror, 45.7 MB):

   ```sh
   venv/bin/python tools/selfplay_elo_mixer.py fetch-model --out /tmp/maia3-onnx
   ```

4. **Fetch the engines** (pinned git refs + build + nets + a one-move
   self-test per engine that hard-fails the whole fetch if any engine
   cannot move):

   ```sh
   venv/bin/python tools/maia3_cloud_selfplay/generate.py \
       fetch-engines --engines-dir /data/engines
   ```

   Builds: Stockfish 18 (cmake/make, few minutes), RubiChess (make,
   few minutes), LC0 (meson, the slow one — ~10-20 min on 90 cores).
   Nets: Stockfish's NNUE from the official tests host (SF18 is
   NNUE-only and refuses to run without it), RubiChess's NNUE from
   `Matthies/NN` (pinned), LC0's T8 net from `sergiovieri/lc0-training`
   (pinned, sha256-verified). Skip an engine with `--engines ...`.

5. **Pilot first** (≈1-3 min, gives the real rate on this exact box —
   the tool logs rate + ETA per shard every 30 s):

   ```sh
   venv/bin/python tools/maia3_cloud_selfplay/generate.py generate \
       --model /tmp/maia3-onnx/simple-maia3-inference/simple_maia3_inference/maia3_simplified.onnx \
       --engines-dir /data/engines --out /data/pilot \
       --games 1000 --workers 90
   ```

6. **Generate the 5M set** (mixed pool, `--workers 90` — each worker
   drives two engines at once, so ~2 active cores/worker on 180
   vCPUs; maia3-only runs use `--workers 170`):

   ```sh
   venv/bin/python tools/maia3_cloud_selfplay/generate.py generate \
       --model /tmp/maia3-onnx/simple-maia3-inference/simple_maia3_inference/maia3_simplified.onnx \
       --engines-dir /data/engines --out /data/mixed-5m \
       --games 5000000 --seed 20260827 --workers 90
   ```

   The command **validates the whole set when it finishes** (every
   move replayed legal, headers/labels/engine/quality cross-checked,
   per-engine stats + Maia-3 conditioning gradient reported to
   `calibration.json`) and exits non-zero on any hard failure. If the
   instance dies mid-run, re-run with `--resume` (crash loses at most
   `--fsync-every` games per shard, default 10).

7. **Ship the data back** (rsync/scp from the instance, or point your
   training job at the instance volume if your stack supports it).

## Expected duration and cost (measured inputs, honest)

Per-ply inputs, **measured** (not guessed):

  * Maia-3 ONNX forward, CPU, 1 thread: **29.5 ms/ply** on this
    repo's 2-core sandbox (Xeon, Sapphire Rapids class; the earlier
    sandbox measured 98 ms/ply — server CPUs vary ~3x, so plan on a
    30-100 ms range). 2 intra-op threads measured 26.8 ms (1.10x —
    not worth the core it steals; `--maia-threads 1` stays the
    default). ORT dynamic int8 quantization of this export produces an
    invalid graph (measured), so there is no int8 fast path.
  * UCI engines: 100 ms movetime (Stockfish), 250 ms + NPS cap
    (RubiChess), LC0 ladder 2-500 ms (mean ≈ 100 ms) + a few ms of
    UCI round-trip overhead.
  * Average game length: **65.4 plies** (13,076 labels / 200 games in
    the committed reference set).
  * **Correction of the earlier estimate**: the previous README's
    "~2-4 h / $5-9 for 2M" was wrong by ~3-10x (it under-assumed game
    length and per-ply time). The numbers below use the measured
    values.

| Run | Workers | Est. duration | Est. cost @ $2.16/h |
|---|---|---|---|
| **5M mixed pool (default)** | 90 | **~95-110 h** | **~$205-240** |
| 5M maia3-only | 170 | ~15-49 h | ~$33-106 |
| 2M maia3-only | 170 | ~6-20 h | ~$13-43 |

Read: per-ply CPU inference is the bottleneck (Maia-3 forward is
~90%+ of a move's cost), so the big flat-core node is still the sweet
spot — and it is priced per vCPU, so the same core count inside an
H100 box costs ~5-6x more for the same CPU throughput. A GPU only
wins decisively with **batched cross-game inference** (one forward per
batch of in-flight games), which is not implemented yet; with it, a
4-GPU box would do the 5M in roughly 4-8 h. **Trust the pilot's
measured rate over this table** — the first 1,000 games nail the ETA
for the run (deterministic plan, no variance).

## Calibration and tests

- `calibration.json` reports per-engine move counts and mean top-1
  probability per 100-elo band **over the Maia-3 rows**. The committed
  200-game reference measures **0.323 at 100-199 rising to
  ~0.52-0.62 at 1900-3200**; a full set should reproduce the shape
  (the validation pass asserts low-end < high-end concentration with
  ≥1,000 Maia-3 moves and fails otherwise).
- `tools/test_maia3_cloud_selfplay.py` (12 tests, runnable in the
  sandbox, no cloud needed): plan determinism (both streams),
  single-engine stream == legacy `plan_elo`, ladder monotonicity,
  pool/elo-bound filtering, model anchor outputs (startpos top-1
  0.6165 @1500/1500, 0.1666 @200/3000 ±0.002), a 4-worker maia3-only
  end-to-end mini-run (8 games) through generate → validate, and a
  mixed maia3+RubiChess end-to-end run with real engine subprocesses
  (skips when the scratch build is absent).

## What is and isn't verified in the sandbox (honest)

The sandbox's egress is restricted to GitHub git + PyPI, so:

- **Verified here** (built + moved in real games): Maia-3 ONNX
  (anchors + e2e), RubiChess 2026-08-27 (NNUE via `NNUENetpath` —
  note there is no `--nnue` CLI; a failed net load with
  `Use_NNUE=true` hangs the search, so the generator forces
  `Use_NNUE=false` when no net is present), the UCI protocol layer,
  the plan/label/validation pipeline end-to-end.
- **Verified from source only**: Stockfish 18's UCI_Elo range
  (1320-3190) and LC0's option set (no UCI_Elo; net via the
  `WeightsFile` UCI option). SF18's net lives on
  `tests.stockfishchess.org` (not the GitHub CDN) and LC0's meson
  subprojects download from a release CDN — both are blocked from
  this sandbox, so **both engines' fetch + first real move happen in
  `fetch-engines` on the instance**, whose per-engine self-test
  hard-fails the fetch if the engine cannot play.

## Honest limits

- Same model caveats as `data/selfplay/README.md`: "simplified"
  single-position export (no history/clock inputs), sampling without
  the official UCI's one-ply ranking; model-generated, not human.
- LC0/RubiChess elo labels are monotone but uncalibrated (see the
  pool table) — filter by `elo_quality` for the conditioning signal.
- The LC0 net is a 2023-era T8 net (`net40`, ~88 MB) from a pinned
  community repo: fine for level diversity, older than the current
  best nets (its "3200" label means "full uncapped budget", not a
  measured 3200 player).
- 5M games ≈ 327M rows ≈ 87 GB — shard it into object storage or your
  training volume early.
- Spot instances can be preempted — `--resume` is the safety net;
  checkpoints fsync every 10 games (tune with `--fsync-every`).
