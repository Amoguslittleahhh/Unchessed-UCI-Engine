# Research note — "Mixture of Experts in Large Language Models" (arXiv:2507.11181)

Assessment of Zhang et al., *Mixture of Experts in Large Language Models*
(v2, 22 Dec 2025), for relevance to this engine.

## What the paper is

A **survey**, not a new technique. It reviews the MoE design space: gating and
routing (top-k, entropy-based, load-balanced, attention-based), hierarchical and
sparse configurations, parameter-efficient variants (LoRA-MoE, Nexus), and
deployment considerations. It contributes taxonomy and synthesis, not an
algorithm to port.

The core MoE claim is: **grow capacity without growing per-inference compute**,
by activating only a small subset of parameters per input.

## Verdict

**Mostly not applicable, with one genuinely relevant exception.**

Two of the paper's three pillars do not transfer:

1. **Sparse conditional compute doesn't buy what it buys in LLMs.** MoE wins
   when a model is so large that activating all of it is the bottleneck. This
   engine's NNUE is a 22528x256 feature transformer plus a *512x1 output layer*
   (verified: `unchessed-nnue.bin` is v3, `ft_in=22528`, `acc=256`, single
   output head, `mult=2`). The output layer is already tiny. Routing it to one
   of N experts saves nothing — the accumulator, which is the real cost, would
   still be fully computed.
2. **Routing overhead lands on the wrong side of the ledger.** An MoE router is
   a per-inference gate. Here inference happens at *every search node*, so a
   gate costs more than the layer it selects. This is round 0's lesson in a new
   costume: per-node neural overhead is the thing that already lost 0-20-0.

**What does transfer: the paper's observation that MoE is most effective as
"a practical control for compute allocation" — which in chess terms is
output-head bucketing, a standard and well-proven NNUE technique this engine
does not currently have.**

## The one real opportunity: bucketed output heads

Modern Stockfish-lineage NNUEs use **8 output buckets** selected by piece count
(`(popcount(occupied) - 1) / 4`). Each bucket is its own small output layer.
This is precisely MoE's structure, with the routing degeneracy that makes it
free:

| | LLM MoE | NNUE output buckets |
|---|---|---|
| Router | learned gate over experts | `(piece_count - 1) / 4` |
| Router cost | a real matmul per token | one `popcnt` + shift |
| Load balancing | needs an auxiliary loss | inherent (game phase is naturally spread) |
| Expert collapse | a live failure mode | impossible, routing is not learned |

The paper spends much of its length on routing instability, expert collapse, and
load-balancing losses. **A fixed, hand-specified router makes every one of those
problems vanish.** That is the actual insight worth taking: the benefit of
conditional compute is available here *without* the machinery that causes MoE's
documented failure modes.

Why it helps in chess specifically: a single output head must learn one mapping
from accumulator to centipawns that is simultaneously correct for a 32-piece
opening and a 5-piece endgame, where material means very different things.
Bucketing gives each phase its own head. This is standard practice in strong
engines and is typically worth real Elo.

### Cost

- **Inference:** unchanged. Exactly one head is evaluated per eval, the same
  512-wide dot product as today. The only added work is a `popcnt` — and
  `pos.occ` is already maintained, so it is one instruction.
- **Memory:** +7 heads x 512 floats = ~14 KB. Negligible against the 23 MB
  feature transformer.
- **File format:** needs a v4 (`out_w` becomes `[buckets * mult * ACC]`), and
  `train_nnue.py`'s `nn.Linear(2*ACC, 1)` becomes `nn.Linear(2*ACC, buckets)`
  with the loss reading the active bucket.

### Why this is not a code change today

It requires **retraining**. The shipped `unchessed-nnue.bin` has one head; you
cannot synthesize eight from it. The work is:

1. bump the trainer to a bucketed head and the format to v4;
2. retrain;
3. SPRT the result against the current net.

That is a training project, not a runtime optimization, and it must not be
started by silently breaking the v3 loader that the engine currently depends on.
So this note records the design rather than shipping a half-version.

## Ideas from the paper explicitly rejected

- **Learned routing over eval experts.** Adds a per-node gate, risks expert
  collapse, and needs a load-balancing loss — all cost, no compensating win at
  this model size.
- **Hierarchical / two-level MoE (H-MoE, MixER).** Multi-stage routing for a
  512x1 layer is strictly overhead.
- **Sparse expert scaling for Unarchitectured Metal.** The 4.2M-param student is
  already too slow to pay per move (round 0), and its measured policy top-1 is
  0.255. More capacity is not the blocker; the blocker is that the forward pass
  costs more than it returns. MoE would raise capacity while *increasing*
  latency variance — the wrong direction.
- **LoRA-MoE.** Already effectively present: Unarchitectured Metal has rank-16
  human/guide policy adapters routed by `policy_kind`, which is exactly the
  LoRA-MoE pattern the paper describes (`POLICY_ADAPTER_RANK = 16`,
  `policy_body/source/target.adapter_a/b`). Nothing to add.

## Recommendation

Do not implement MoE. Do consider **8 piece-count output buckets** in the next
NNUE training run — it is the same "conditional compute" idea with a free
router, it is standard in strong engines, and it is the only part of this survey
that survives contact with a per-node evaluation budget.

Priority-wise it sits behind validating the already-written SIMD work
(`docs/performance-round-1-implementation.md`), which is pure speed at zero
training cost and still needs `cargo test` on a machine with a toolchain.

## Status (2026-08-28) — 8 buckets implemented, retrain still gated

The 8 piece-count output buckets are now **implemented** on both sides,
default-inert until a retrain:

- **Runtime** (`unchessed-core/src/nnue.rs`): file format **version 4** =
  HalfKAv2_hm features + per-bucket output head (`out_w` = 8 × [512] f32,
  `out_b` = 8 f32, per bucket [STM half | NSTM half]). The bucket is
  `clamp((pieces−1)/4, 0, 7)` from the position's occupied-squares count at
  eval time — no state bookkeeping needed, because the incremental state is
  just the two accumulators and captures (the only bucket-changing event)
  recompute the bucket from the new position. Versions 1–3 remain loadable
  (`out_buckets = 1`, single shared head). Tests: per-bucket constant-head
  probe net asserts the exact bucket for piece counts 3/5/11/15/18/21/26/32,
  and a 29→28-piece capture asserts the eval crosses bucket 7→6 by exactly
  the per-bucket bias delta through both the full-refresh and incremental
  state paths.
- **Trainer** (`tools/train_nnue.py`): `Linear(2×ACC, 8)` head, bucket from
  the popcount of the 12 input planes, v4 export, selfcheck ALL PASS
  (numpy manual forward matches the model to 2.98e-08), and a
  trainer→runtime ABI cross-check: the trainer-exported net loads in the
  Rust runtime and reproduces the Python manual forward exactly
  (startpos: raw 0.031979, cp 12, bucket 7).

**Not done, on purpose:** the retrain itself (filtered corpus per
`docs/nnue-dataset-quiet-filters.md` + this head) is the owner's call —
real compute — and any resulting net goes through SPRT before it can touch
the default evaluation. Nothing in the default search path has changed.
