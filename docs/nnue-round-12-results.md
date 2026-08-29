# NNUE round 12 — results (2026-08-28)

Honest results for the round-12 request
(`scripts/research/arena_agent_unarchitectured_v1_runtime_speed_prompt.md`).
Diff vs `main` (`8f63321`, the re-imported root commit): 8 files,
+689/−87 — all listed below. All work pushed to
`arena/01a02efe-unchessed-uci-engine` at `0f59057`.

## What this round actually did

Round 11 was merged, so the round-12 asks were the "what's needed next"
list. Status per item:

| Item | Status |
|---|---|
| NNUE scrutiny (quiet filters + 8 output buckets) | **Done** (implementation + measurement); the *retrain* itself remains the owner's call |
| DiffusionBlocks labeling-oracle question | **Answered**: defer/drop at current scale |
| Oracle-side rating conditioning | **Narrowed**: one hypothesis ruled out with repo evidence; deciding experiment requires the oracle checkpoint (not in repo) |
| Retrain decision (GAB capacity, rating conditioning, int8 clipping) | **Not started by design** — owner's call, real cloud spend |
| Isolated `MinTime` retest (open from round 7) | **Not started** — optional polish; would need real-hardware SPRT, and must now state its `UnarchitecturedHintExit` (new standing rule) |

## 1. NNUE quiet-position dataset filters — measured, blocker fixed

Round 4 built the two filters (arXiv:2412.17948) but they were never run,
because the fail-closed `NNUE_MIN_BASE_SECS=180` gate rejected 100% of
candidates: **no committed corpus carries TimeControl headers** (verified
across `data/training`, `data/training-elo`, `data/selfplay`). Fix:
`UNCHESSED_NNUE_MIN_BASE_SECS` env override (default 180, `0` disables),
plus `UNCHESSED_QUIET_HISTOGRAM=1` reject-curve histograms with per-filter
candidate counts (`unchessed-datagen/src/main.rs`).

Measured on `data/training/lichess-2022-10-05/elo-1700-2000.pgn` (4,465
games seen, 1 thread, 121 s, ~358 samples/s), margins 60/70:

- M1 (|static−quiescence| > 60): **17,021 / 79,267 = 21.5%** rejected,
  before the label search (cheap).
- M2 (|static−search| > 70): **8,543 / 51,818 = 16.5%** rejected, after
  the 5000-node label search (each reject paid for the search).
- **43,275 accepted.** Both reject curves are long-tailed out to 150 cp
  (full table in `docs/nnue-dataset-quiet-filters.md`) — no elbow arguing
  for different margins. **60/70 kept as the default.**
- Filter overhead: ~3% throughput (308 vs 300 samples/s, unfiltered vs
  filtered A/B, same 20k-record cap).
- Full committed corpus ≈ 95k games ≈ ~1M candidate positions ≈ **~1 hour
  single-threaded on this 2-vCPU host**, minutes on the 180-vCPU cloud
  host. Regenerating training data is now directly affordable here.

One correction to an earlier session note: the M1 candidate count is
79,267 (measured with the new counters), not the 60,296 carried in
session memory — the 21.5% rate (not 28.2%) is the correct one.

## 2. NNUE 8 piece-count output buckets — implemented, retrain gated

The one surviving recommendation from
`docs/research-notes-moe-2507.11181.md`, implemented end-to-end:

- **Runtime** (`unchessed-core/src/nnue.rs`): file format **version 4** =
  HalfKAv2_hm features + `out_w` [8][512] f32 (per bucket: STM half |
  NSTM half) + `out_b` [8] f32. Bucket = `clamp((pieces−1)/4, 0, 7)` from
  the occupied-squares count at eval time; the incremental state
  (two accumulators) is unchanged — captures, the only bucket-changing
  event, recompute the bucket from the new position, so `update_state`
  needs no bookkeeping. Versions 1–3 remain loadable (single head,
  bit-identical behavior).
- **Tests** (4 new): per-bucket probe net (zero head weights, per-bucket
  biases 0..=7) asserts the exact bucket for 3/5/11/15/18/21/26/32 pieces
  through both `eval` and `eval_with_state`; a 29→28-piece capture asserts
  the eval crosses bucket 7→6 by exactly the per-bucket bias delta on both
  the full-refresh and incremental paths; plus v4 dummy-net load and
  move-pair incremental checks.
- **Trainer** (`tools/train_nnue.py`): `Linear(2×ACC, 8)` head, bucket
  from the popcount of the 12 input planes, v4 export, selfcheck
  **ALL PASS** (numpy manual forward matches the model to 2.98e-08).
- **Trainer→runtime ABI cross-check**: the trainer-exported net loads in
  the Rust runtime and reproduces the independent Python manual forward
  exactly (startpos: raw 0.031979, cp 12, bucket 7).

**Not done, on purpose:** the retrain. It needs the owner's call (real
compute) and any resulting net goes through SPRT before it can touch the
default evaluation. The default search path is unchanged; nothing
shipping depends on the v4 format until a v4 net exists.

## 3. DiffusionBlocks for a labeling-oracle retrain — answered

`docs/research-notes-diffusionblocks-2506.14202.md`, four sections for the
four questions:

1. **Fits architecturally, no I/O rework** — a chess value oracle
   (encoder → uniform residual stack → scalar head) is the same shape as
   the paper's own ViT demo; the diffused quantity is the internal
   representation, not the board. Caveats: **no published regression
   results** (all results are classification/generation) and the public
   code is ViT/CIFAR-100 only.
2. **Memory at real scales**: the published chess transformers are
   ChessMimic ~9M, ChessFormer 100.7M, ChessBench 270M (16L/d=1024).
   Standard mixed-precision + Adam footprint ≈ 16 B/param fixed
   (0.2/1.6/4.3 GB) plus activations — **all three fit a single 80 GB
   card at practical batch sizes.** The B× savings become load-bearing
   only beyond ~500M–1B params or on sub-48 GB cards. (No public
   training-memory figures exist for these chess transformers; the
   activation numbers in the doc are labeled estimates.)
3. **One-time oracle economics**: porting a ViT-only reference to a
   custom encoder + value head, with untested regression parity, for a
   model used once and discarded — versus renting 1–4× H100/A100-80 for
   a few hundred dollars. The honest answer is the prompt's own escape
   hatch: **rent the GPU and use ordinary backprop**; gradient
   checkpointing covers the small-card case with no architectural risk.
4. **Recommendation: defer — effectively drop at current scale**, with
   explicit revisit conditions (≥~500M–1B oracle, single ≤48 GB card,
   checkpointing insufficient) and a smallest-useful-experiment sketch
   (16L/d=512 ~30–40M value transformer on 1M Stockfish-labeled
   positions, end-to-end vs B=4, go/no-go on ≥3× peak-memory reduction
   with parity within ~2–5%).

## 4. Oracle-side rating conditioning — bug location narrowed

`docs/rating-conditioning-finding.md` gained an oracle-side section:

- **Ruled out:** "the student never saw rating variation" — the
  distillation dataset carries per-move dual elo
  (`tools/pretrain_move_dataset.py`), so the student trained on
  rating-varying inputs.
- **Consistent with, not decisive:** the measured response is cleanly
  linear in rating — the signature of the single-scalar injection path
  with a small learned weight, which is what a network learns when its
  (oracle-supplied) targets are rating-invariant.
- **Live hypothesis:** distillation-time (rating-invariant teacher
  targets). Only the oracle-side sweep decides it — and it is cheap when
  the checkpoint is available (200 positions × 7 ratings, CPU-minutes).
- **Both outcomes are actionable:** oracle varies → student-architecture
  fix (the planned retrain already replaces the single-scalar path with
  dual-elo conditioning and uses this sweep as its gate); oracle
  invariant → fix the oracle first, the student retrain inherits it.

## Verification (this host: cargo 1.88.0, torch 2.5.1 CPU)

- `cargo test --workspace --release`: **unchessed-core 110 passed / 0
  failed / 6 ignored** (106 before the round, +4 new v4 tests); the other
  three workspace crates have 0 tests.
- `pytest tools/`: **337 passed, 4 skipped, 352 subtests** — unchanged
  from the round-11 baseline.
- `train_nnue.py selfcheck`: ALL PASS.
- Trainer→runtime ABI cross-check: exact match (raw 0.031979 / cp 12).
- Round-4 quiet-filter unit test (`static_and_quiescence_separates_quiet_
  from_hanging_positions`, `unchessed-core/src/search.rs`) executed: pass.
- The quiet-filter reject numbers were reproduced twice on this host
  (original run + rerun: identical 43,275 samples, identical grid curves).

## Honest negatives and open items

- **No net was retrained and no engine was SPRT'd this round.** The
  quiet-filter benefit and the 8-bucket benefit are the paper's and the
  MoE note's, not measured playing-strength deltas here. The honest
  current state: better dataset tooling + a verified new net format,
  zero shipping change.
- `MinTime` retest: not started. If it happens, it must state its
  `UnarchitecturedHintExit` — all four SPRT batches so far used the
  hard-coded 2/128 exit (worst-calibrated, top-1 0.185 vs 0.255 at 8/256).
- `UnarchitecturedHint` remains default-off, as directed.
- The retrain decision (GAB capacity ¼ of the paper config, rating
  conditioning, int8 weight clipping 2.06× out of range) remains the
  project owner's call with round 10's infrastructure in place.

## Files (diff vs `main`)

- `unchessed-core/src/nnue.rs` (+230/−…) v4 format, bucket eval, 4 tests
- `tools/train_nnue.py` 8-bucket head, v4 export, selfcheck
- `unchessed-datagen/src/main.rs` base-seconds override, histograms,
  candidate counters
- `docs/nnue-dataset-quiet-filters.md` measured retuning section
- `docs/research-notes-moe-2507.11181.md` implementation status
- `docs/research-notes-diffusionblocks-2506.14202.md` new (the answer)
- `docs/rating-conditioning-finding.md` oracle-side section
- `scripts/research/200_research_ideas.md` item 153 status pointer
