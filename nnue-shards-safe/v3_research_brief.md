# NNUE v3 (HalfKA) Regression — Research Brief

## Project context

Unchessed AI is a from-scratch UCI chess engine (Rust core + Python training tooling). It currently has a working, SPRT-validated NNUE eval (**v1**) deployed as the default:

- **v1 architecture**: flat 768-input features (`plane*64 + sq`, own/opp × 6 piece types × 64 squares), `EmbeddingBag(768, 256, sum)` feature transformer, SCReLU activation `clamp(x,0,1)^2`, concat both perspectives (512-wide) → `Linear(512, 1)`.
- **v1 result**: SPRT-validated **+107.1 Elo** vs. the classical hand-crafted eval it replaced (the single biggest gain in this project's history). Trained on ~108M labeled positions.

We just attempted **v3**, a from-scratch retrain with a new architecture, and it **decisively failed** its SPRT gate against v1. This document is everything relevant to diagnosing why, for a one-shot external research pass — no follow-up round is available, so please be thorough and cite specific mechanisms, not just general NNUE folklore.

## What v3 changed (two things at once — this is itself a known methodological problem, see "Known issues" below)

### Change 1: Feature scheme — flat-768 → HalfKA-style, king-relative, NO mirroring

- **Feature index formula (as implemented)**: `king_sq * 704 + piece_idx * 64 + sq`, where `king_sq` is the **own** king's square in that perspective's frame (0-63, no mirroring/bucketing), and `piece_idx` (0-10) enumerates the 11 non-own-king (color, piece-type) pairs in order: own P,N,B,R,Q (0-4), then opp P,N,B,R,Q,K (5-10).
- `FT_IN = 64 * 11 * 64 = 45,056` (up from 768 — a **~59x** increase in embedding-table size, i.e. `EmbeddingBag(45056, 256)` = ~11.5M params in just the feature transformer, up from ~196K).
- **The own king is NOT itself an active feature** — it's used purely as the multiplier/anchor for indexing everyone else's features, and contributes nothing to the active-feature set for that perspective.
- No horizontal mirroring, no king-bucketing (unlike Stockfish's production `HalfKAv2_hm`, which halves the feature count via mirroring — see "What we already checked" below).
- Both perspectives (STM/NSTM) computed via: unpack all 12 raw bitboard planes into one-hot bits, derive NSTM from STM via a plane-group swap + rank-flip (`reshape(n,12,8,8).flip(dims=(2,))`), independent of feature-index math itself.

### Change 2: Output head — SCReLU-only, 512-wide → SCReLU+ClippedReLU concat, 1024-wide

- v1: `concat([screlu(acc_stm), screlu(acc_nstm)])` (512) → `Linear(512, 1)`.
- v3: `concat([screlu(acc_stm), crelu(acc_stm), screlu(acc_nstm), crelu(acc_nstm)])` (1024) → `Linear(1024, 1)`, where `crelu(x) = clamp(x, 0, 1)` (no squaring). This is the "SFNNv5 trick" — concatenating both the squared and linear activation shape of the same accumulator.
- Same 256-wide per-perspective accumulator width as v1 (not widened independently).

## Training setup (unchanged from v1 except epoch count in the rerun)

- **Loss**: `|sigmoid(raw) - target|^2.5` (not plain MSE), computed as `diff^2 * sqrt(|diff|+1e-8)` for numerical stability. `target = 0.7*sigmoid(cp/400) + 0.3*(wdl/2)`.
- **Optimizer**: Adam, `lr=1e-3`, step-decayed `*0.3` at 60% and 80% of total epochs (unchanged formula regardless of total epoch count — i.e. for a 15-epoch run, drops happen at epoch 9 and 12; for the 100-epoch rerun, at epoch 60 and 80).
- **Batch size**: 131,072, GPU-resident (whole dataset — 268.8M records, ~28GB raw — transferred to VRAM once, all epoch shuffling done as GPU tensor ops).
- **Weight init**: `EmbeddingBag.weight` uniform(-0.05, 0.05) (same init range used for both v1's 768-row table and v3's 45,056-row table — not rescaled for the new table's different fan-in/fan-out characteristics).
- **No regularization**: no dropout, no weight decay, no gradient clipping anywhere in the training loop.
- **No feature factorization ("virtual features")** — each of the 45,056 rows is trained completely independently; there is no shared/coarser auxiliary embedding folded in during training to give sparse rows a denser gradient signal.
- **Dataset**: 268.8M positions total (5 months of real Lichess rated games, fixed-node HCE search-labeled — same labeling methodology as v1, just more months). Split 268.6M train / 200K val.
- Hardware: NVIDIA A100 80GB SXM4, DEVICE=cuda, GPU-resident training path (custom-built this session, not from nnue-pytorch).

## Empirical results

### SPRT gate (15-epoch v3 vs. v1, same binary/engine, only `EvalFile` differs)

```
Score: 216-367-173 (756 decided games)
Elo difference: -70.3 +/- 22.1
LOS: 0.0%
SPRT: llr -2.96 crossed lower bound -2.94 -> H0 accepted (DECISIVE FAIL, not borderline)
```

Settings: tc=5+0.05, concurrency=13, elo0=0/elo1=5, alpha=beta=0.05 — same settings used for every prior SPRT gate in this project's history (SEE +19.4, futility pruning +31.7, magic bitboards +21, Lazy SMP +181, all passed cleanly at these settings; several extension-family techniques failed decisively at these same settings too, so the harness is trusted/calibrated).

### 100-epoch rerun (same everything, just more epochs, to test "was it just undertrained?")

Key epoch snapshots (val-mae in centipawns, val-loss, train-loss):

| Epoch | train-loss | val-loss | val-mae |
|---|---|---|---|
| 1 | 0.009108 | 0.021056 | 56.9cp |
| 13-14 | ~0.00816 | ~0.02059 | **53.6cp (best point of the whole run)** |
| 20 | 0.008112 | 0.020585 | 54.8cp |
| 30 | 0.008064 | 0.020583 | 55.5cp |
| 45 | 0.008022 | 0.020606 | 56.4cp |

**Pattern**: val-mae bottoms out around epoch 13-15, then *rises* steadily through epoch 45 (56.4cp — worse than epoch 1!) while train-loss keeps monotonically decreasing the entire time. This is a textbook overfitting signature, not an undertraining one — training was stopped at epoch 50 given the clear, sustained (30+ epoch) downward trend, before reaching the full 100 epochs, since continuing would only get worse and the script has no mid-run checkpoint/export capability (weights only export after the full requested epoch count completes, so no checkpoint was salvaged from this run).

**Conclusion so far**: more epochs alone made things *worse*, not better — this rules out simple undertraining as the explanation.

## What we already checked against real Stockfish source (`official-stockfish/Stockfish`, `src/nnue/features/half_ka_v2_hm.h`)

- Real Stockfish's `HalfKAv2_hm`: `Dimensions = SQUARE_NB * PS_NB / 2` = 64 × 704 / 2 = **22,528** (half of our 45,056 — they mirror horizontally, always orienting the king to files e-h; we don't mirror at all).
- **`PS_NB` has 11 categories: 5 piece types × 2 colors (10) + ONE shared `PS_KING` category covering BOTH kings.** From the `PieceSquareIndex` table, both `W_KING` and `B_KING` map to the same `PS_KING` slot — meaning **the own king DOES generate its own active feature** in real Stockfish (at the king's own square, in the shared KING plane), on top of being the indexing anchor. We explicitly excluded this in v3 ("own king is the anchor, never itself an active feature").
- `MaxActiveDimensions = 32` in real Stockfish (our nnz per position is ~28-30, same ballpark).
- We did NOT check nnue-pytorch (the actual training-side repo) in this session — only the inference-side feature definition in the main Stockfish repo. nnue-pytorch would have the actual training recipe (LR schedule, factorization implementation, regularization) and hasn't been consulted yet.

## Known/suspected issues, ranked by how confident we are

1. **(Methodological, certain)** Two independent architecture changes (feature scheme + output head) were bundled into one test. This project has an established, previously learned-the-hard-way rule against this (an earlier SPSA campaign on search-engineering techniques explicitly warned against conflating multiple simultaneous changes). Whatever the root cause turns out to be, this v3 attempt violated that rule, so we don't cleanly know which change (or interaction) is responsible.
2. **(Suspected, matches evidence)** No feature factorization. Each of 45,056 rows gets gradient updates only when its exact `(king_sq, piece, sq)` combination occurs — much sparser per-row signal than v1's 768 rows ever had, even with 2.5x more total training data (268.8M vs ~108M). This is consistent with the observed pattern: the model can still fit the training set arbitrarily well (train-loss never stops falling) while individual rows are too undertrained to generalize, and MORE epochs just lets already-well-covered rows overfit harder rather than fixing the sparse ones.
3. **(Suspected, verified against source, effect size unknown)** Own king excluded as an active feature, contrary to real Stockfish's design (see above). Effect on generalization untested — could be meaningfully informative (self-referential king-plane signal) or could be genuinely redundant with the king_sq-based indexing that already exists regardless. Needs direct evidence, not assumption either way.
4. **(Plausible, untested)** No horizontal mirroring — the unmirrored 45,056-dim table means the model has to separately (and redundantly) learn mirror-symmetric weights for e.g. `king_sq=e1` and `king_sq=d1` cases that should share structure, doubling the effective sparsity problem beyond what a mirrored implementation would face.
5. **(Plausible, untested)** Hyperparameters (LR=1e-3, decay schedule, Adam, no weight decay, same init range) were carried over unchanged from a model with a 196K-param feature transformer to one with an 11.5M-param feature transformer. No adjustment was made for the much larger table's different capacity/overfitting characteristics.

## Specific research questions

1. **How exactly does nnue-pytorch implement feature factorization for HalfKA/HalfKP-family features?** What's the exact mechanism (shared coarser embedding table folded in via addition before/after training, or some other scheme), and how does it get "unfolded" into the final exported sparse weights? Would this directly explain fixing our overfitting pattern?
2. **What does including vs. excluding the own-king active feature actually change, mechanistically?** Is there existing ablation/discussion (Stockfish commit history, nnue-pytorch issues/docs, or the "Neural Networks for Chess" book referenced earlier in this project's own research) on this specific design choice?
3. **What LR schedule, optimizer, weight decay, and regularization does nnue-pytorch's actual reference trainer use for HalfKA-scale networks**, and how does it differ in kind (not just magnitude) from a simple step-decayed Adam with no regularization?
4. **Given a fixed ~270M-position dataset, is there a known rule of thumb for how many epochs / how much data a HalfKA-scale (10M+ param) embedding table needs to converge**, versus what a flat 768-input table needs? Are there published convergence curves or guidance from Stockfish's own training documentation?
5. **Is there evidence that skipping horizontal mirroring specifically causes the kind of degraded generalization we saw**, independent of the factorization/own-king questions — i.e., if you fixed factorization and own-king inclusion but left mirroring out, would that alone still cause a regression like this?

## What NOT to re-litigate

- Loss function (`|sigmoid-target|^2.5` WDL-blended) — already independently confirmed correct against Stockfish's own documented recommendation, not a suspect here.
- GPU-resident training pipeline mechanics — verified bit-exact against the host-resident (numpy) path via direct tensor comparison, not a suspect (this is an infrastructure/plumbing layer, not a modeling choice).
- Labeling methodology (fixed-node HCE search, quiet-position filtering) — unchanged from v1's successful methodology, not a suspect.
