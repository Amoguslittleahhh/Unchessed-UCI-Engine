# Move-prediction pretrain — plan and sandbox probe

**Status:** plan + working sandbox probe (2026-08-28). The probe is
evidence that the *objective* makes dual-elo conditioning
informative; it is not a deployment result and has no Elo meaning.

## The idea

The level-conditioned retrain (roadmap item 1) needs a model that,
given (board, elo_self, elo_oppo), predicts how a player at that level
moves. The standard way to get that is a two-stage design:

1. **Pretrain — next-move prediction.** Behavioral cloning:
   cross-entropy over the position's *legal* actions (the
   4096×5 = 20480 move-promotion space of the Unarchitectured v1
   student), trained on the whole mixed corpus — the 1M–5M
   cloud self-play set (four engines × their elo ranges),
   `data/selfplay/`, and `data/training-elo/` (35,812 real rated
   games). Every row teaches "at this level, from this position, this
   is what was played." Because the same position appears under many
   different (elo_self, elo_oppo) labels with different played moves,
   **the loss can only go down if the model actually uses the dual-elo
   conditioning** — this is what the canonical v1's single `rating`
   field (0/200 sweep, `rating-conditioning-finding.md`) never
   forced.
2. **Fine-tune — level alignment.** Continue from the pretrain
   checkpoint on the *trusted* rows only (quality `calibrated` +
   `native` + real humans), with the conditioning inputs at full
   weight, to sharpen the level axis and pull the style toward
   humans (the mixed corpus is engine-styled; the fine-tune data is
   where human style enters).

Both stages run through the existing `train_unarchitectured_v1_a100.py`
oracle→student pipeline (the pretrain is stage 1 of the retrain, not a
parallel architecture). The one ABI extension the retrain round needs:
the frozen v3 base record carries a single `rating:u16`; the
pretrain matrix carries **both** elos (see the bridge below).

## The tools (in `tools/`)

- `pretrain_move_dataset.py` — bridge: label JSONL (cloud output /
  `data/selfplay/`) and plain PGN with WhiteElo/BlackElo headers
  (`data/training-elo/`) → sharded npz arrays. Board encoding is the
  STM-normalized 12-plane bitboard convention of
  `unchessed-datagen` (mover planes 0-5, opponent 6-11, a1=0);
  actions in the 20480 move-promotion space; **target and legal set
  are built in the same mirrored view** (a double mirror would put
  the target outside the legal set for every black-to-move row —
  caught by a hard guard in `build()`); train/val split is
  **game-disjoint** (leakage guard). Rows carry `engine` +
  `quality` (calibrated / native / approximate / human) for
  per-stage weighting.
- `pretrain_move_predictor.py` — the sandbox-runnable probe: a small
  NumPy MLP (774-d features: 768 board bits + 4 castling bits +
  elo_self/100 + elo_oppo/100 → 2×hidden → 20480 logits, legal-only
  softmax, Adam + L2) trained on the shards, plus the diagnostics:
  per-100-elo-band top-1 accuracy and mean top-1 probability, and
  **the conditioning sweep** — on fixed held-out positions, sweep
  mover elo 600→3200 (opponent fixed) and count how many change
  their predicted top-1 action. The v1 finding was **0/200**
  (inert); a working pretrain must show substantial flips.
  Backprop verified against central-difference finite differences
  (test: `test_pretrain_move.py::test_numerical_gradient_toy`).

## Sandbox probe (committed reference set, 13,076 rows)

Command:

```sh
python3 tools/pretrain_move_dataset.py \
    --labels data/selfplay/maia3-100-3200-labels.jsonl \
    --out /tmp/pretrain-selfplay --val-games 50
python3 tools/pretrain_move_predictor.py \
    --data /tmp/pretrain-selfplay --epochs 15 --width 256 \
    --report /tmp/pretrain-selfplay/report.json
```

Results (2026-08-28, this sandbox, seed 20260828, 15 epochs, 256-wide;
full JSON: `benchmarks/unarchitectured-v1/pretrain-probe-2026-08-28.json`):

| Metric | Value |
|---|---|
| train / val rows | 9,946 / 3,130 (50 game-disjoint val games) |
| initial CE (random init) | 3.82 |
| best val CE | 2.848 |
| val top-1 accuracy | **0.1687** (baseline ≈ mean 1/legal_count = 0.0879 → 1.9×) |
| sweep flips (any, 200 positions) | **118/200** — the canonical v1 finding was **0/200** |
| sweep flips (extremes 600 vs 3200) | 115/200 |
| mean top-1 @600 vs @3200 | **0.1561 → 0.2591** (right direction; ground truth on the same data: 0.323 → 0.52-0.62) |

Reading: the objective demonstrably makes the dual-elo conditioning
informative (118/200 vs 0/200) and concentrates high-elo play more
than low-elo play, with top-1 accuracy nearly 2× the uniform-legal
baseline — from 13k rows in a 256-wide MLP. Per-band accuracy is
noisy at this scale (some bands n<100); the 1M+ row mixed corpus is
where the signal sharpens.

The probe is small (13k rows, 256-wide MLP) on purpose: it exists to
validate the objective + encoding + sweep diagnostic, not to be good.
The 1M+ row mixed corpus is where the model itself gets strong; the
A100 pipeline is where it is trained.

## Reading the sweep (what "works" means)

- **Flips:** a large fraction of the 200 positions must change top-1
  between elo 600 and 3200 (v1: 0/200). 30-80% is the expected
  healthy range for a model that has learned level conditioning;
  ~100% would suggest it has learned *elo noise* rather than style
  (check per-band accuracy for that: low-elo accuracy should lag
  high-elo accuracy).
- **Concentration:** mean top-1 probability at 3200 must exceed the
  mean at 600 (the committed Maia-3 reference shows the ground-truth
  shape: 0.323 → 0.52–0.62).
- **Per-band accuracy** should be lowest at the extreme low bands
  (human play there is the least predictable) and rise toward the
  middle/high bands.

## What this is NOT (honest limits)

- Engine-style ≠ human style: the mixed corpus is what the four
  engines do at each level. The fine-tune stage (calibrated + human
  rows) is where human-ness enters; the pretrain stage buys the
  *level axis*, the fine-tune buys the *style axis*.
- `approximate` rows (LC0/RubiChess ladders) are monotone but
  uncalibrated in elo value: they should carry reduced loss weight in
  stage 1 (suggested: 0.5×) and none in stage 2, or be filtered out
  entirely via the `quality` field.
- The probe model is a stand-in MLP; the real pretrain runs the v1
  oracle (16-layer board trunk + GAB + legal decoder) on the same
  objective. The GAB-capacity finding
  (`gab-capacity-finding.md`) still applies to the real retrain:
  widen GAB in the same round.
- **Nothing here touches search or the engine binary.** The hint
  stays default-off; a pretrain+fine-tune net enters the engine only
  through the standing gate: conditioning diagnostics pass, then a
  fresh paired-game SPRT (`No search integration without a fresh
  SPRT gate`).
