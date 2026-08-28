# Move-prediction pretrain — plan, pipeline, and verification

**Status:** CPU + GPU stages built and sandbox-verified (2026-08-28).
The probe (below) is evidence the *objective* makes dual-elo
conditioning informative; the stage-1/2 A100 runs are the next
actions, on the GPU box, starting with the selfcheck.

## The idea

The level-conditioned retrain needs a model that, given
(board, elo_self, elo_oppo), predicts how a player at that level
moves. The standard way to get that is a two-stage design:

1. **Stage 1 — pretrain (next-move prediction).** Behavioral
   cloning: cross-entropy over the position's *legal* actions,
   trained on the whole mixed corpus (the 1M-5M cloud set,
   `data/selfplay/`, `data/training-elo/`), conditioned on dual elo.
   Because the same position appears under many (elo_self,
   elo_oppo) labels with different played moves, **the loss can only
   fall if the conditioning is actually used** — the force the
   canonical v1's single scalar rating never had (0/200 sweep,
   `rating-conditioning-finding.md`).
2. **Stage 2 — finetune (level alignment).** The same objective on
   the *trusted-only* subset (games where every row is calibrated /
   native / human — no approximate ladder rows) at a lower LR,
   resumed from the stage-1 best checkpoint: aligns the level axis
   and pulls in human style (the mixed corpus is engine-styled).

Stage 1+2 run the **Unarchitectured v1 oracle architecture** with
two retrain-round changes:

- **dual-elo conditioning** — the single-scalar rating path in
  `history_context` (measured inert) is replaced by two learned
  projections (elo_self, elo_oppo) (`UnarchitecturedV1OracleDualElo`),
- **GAB widened to the paper's 5M configuration** —
  `gab_token_projection` 16 → 32 (`gab-capacity-finding.md`: ours was
  4× smaller than the paper's smallest).

Measured parameter count of the dual-elo oracle with widened GAB:
**58,486,415** (pinned in `config/pretrain_v1_training.json`).

## The pipeline (CPU work / GPU work)

`scripts/pretrain-pipeline/` splits the work by machine:

| Stage | Box | Tool | Notes |
|---|---|---|---|
| games (1M-5M mixed) | CPU.180V.720G | `tools/maia3_cloud_selfplay/generate.py` | see its README for cost (5M mixed ≈ 95-110 h; pilot nails the rate) |
| **CPU stage** | CPU | `cpu_stage.sh` → `tools/pretrain_v5_data.py` | PGN → v5 dual-elo shards (full + trusted-only) + validation; pure python-chess |
| handoff | — | rsync both shard dirs | the GPU box needs no generator; the CPU box needs no torch |
| **GPU stage** | A100/H100 | `gpu_stage.sh` → `tools/pretrain_v1_a100.py` | selfcheck (first!) → stage 1 (24 epochs) → stage 2 (8 epochs, LR 5e-5, resumed); single GPU |

### v5 data format (`UNCHD5R0`)

The frozen v4 wire record (1088 B) with its 48 reserved bytes
redefined as `elo_oppo:u16 + pretrain_quality:u1 + pad:45`, a new
magic/version/schema-SHA. The mover's own elo stays in the existing
`rating` field (it was always the mover's rating), so each record
carries the dual-elo pair. Quality: 0 calibrated (maia3), 1 native
(stockfish), 2 approximate (lc0/rubichess ladders — monotone but
uncalibrated value; down-weighted 0.5× in stage 1, excluded from
stage 2), 3 human. `wdl` is a **game-outcome proxy** (stage 1 trains
policy CE only and does not use it — do not treat it as a
per-position value label). Train/val split is by **game** (never by
row). Records: board bitboards (STM-normalized, unchessed-datagen
convention), 16-bit action encoding (from|to<<6|promo<<12|kind<<14),
legal set, previous-8-plies history, policy_kind, ply.

### The conditioning sweep (the gate metric)

Every epoch, on held-out positions: mover elo swept 600 → 3200
(opponent fixed at the position's own elo_oppo), count how many
change their predicted top-1 action. Canonical v1: **0/200** (inert).
A working pretrain: substantial flips **and** high-elo play more
concentrated (`top1prob@3200 > top1prob@600`). Both numbers are in
the epoch log and the checkpoint metrics.

## Sandbox probe (objective proof, small)

`tools/pretrain_move_dataset.py` + `tools/pretrain_move_predictor.py`
(NumPy MLP, 256-wide) on the committed 13,076-row self-play
reference — `benchmarks/unarchitectured-v1/pretrain-probe-2026-08-28.json`:

| Metric | Value |
|---|---|
| best val CE | 2.848 (initial 3.82) |
| val top-1 accuracy | 0.1687 vs 0.0879 uniform-legal baseline (1.9×) |
| **sweep flips** | **118/200** (extremes 115/200) — canonical v1: **0/200** |
| mean top-1 @600 → @3200 | **0.156 → 0.259** (right direction; ground truth: 0.323 → 0.52-0.62) |

The probe validates objective/encoding/diagnostic, not strength
(13k rows, 256-wide MLP).

## What is verified, and what is not (honest status)

- **Verified in the sandbox (CPU, this commit):**
  - v5 builder on real PGNs (dual-elo swap by side, quality from
    engine headers, target-in-legal hard guard, game-disjoint split,
    ep/castling/wdl domains) — `tools/test_pretrain_v5.py`,
  - v5 wire round-trip + magic rejection + exact 16-bit action
    encoding values,
  - GPU trainer selfcheck (small dual-elo oracle, 2 steps, sweep),
  - real-data GPU-trainer smoke: v5 shards → mmap loader → 2+
    optimizer steps of the dual-elo oracle → conditioning sweep,
    loss finite, **logits differ across elos for the same
    position** (dual-elo path is live),
  - trusted-only quality filter (all-rows semantics),
  - the 58,486,415 parameter count of the production config (built
    and counted).
- **Not verified here (no CUDA in the sandbox):** the CUDA path of
  the trainer (precision/compile/memory) — the selfcheck is the
  first command on the box for exactly this reason. Multi-GPU DDP is
  not implemented (single GPU).
- **Not yet wired (next round):** distillation to a dual-elo student
  + UNARCHV1 packaging + the Rust runtime change for dual-elo inputs.
  Checkpoints mark `dual_elo: true`; the distill path refuses
  silently-wrong checkpoints. Nothing here touches the engine
  binary; the hint stays default-off and the SPRT gate is unchanged.

## Sizes

1088 B/record: 5M mixed games (~65 plies/game) ≈ 327M records ≈
356 GB (mmap'd on the GPU box); 1M mixed ≈ 65M ≈ 71 GB; the
200-game self-play reference = 13,076 records ≈ 14 MB.
