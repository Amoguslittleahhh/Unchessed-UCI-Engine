# Unchessed XT-NNUE and Chessformer/alpha-beta hybrid

## Honest status

This document defines and begins implementing two research architectures. It is
**not** an accuracy or Elo claim: there is no trained XT-NNUE or Chessformer
asset in the repository yet. The concrete code added with this design is:

- a bounded, deterministic 32,400-dimensional threat-relation extractor;
- a fixed 64-square Chessformer token contract;
- dynamic geometric-attention relation keys; and
- a conservative persona-to-search-backend router with unit tests.

The shipped v3 NNUE and alpha-beta search remain unchanged. Promotion requires
trained weights, validation loss, quantization drift, fixed-node timing, and
paired game evidence.

## Research basis

Stockfish SFNNv10 combined `HalfKAv2_hm` with FullThreats, reduced its large L1
transform to 1,024 outputs, and changed feature-transformer quantization to 255.
Its threat inputs represent selected attacker-square/attacked-piece-square
relationships. Stockfish 18 used a 1,024 -> 16 -> 32 -> 1 family of layer
stacks. These facts are documented by the Stockfish NNUE guide and SFNNv10
commit:

- <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
- <https://github.com/official-stockfish/Stockfish/commit/8e5392d79a36aba5b997cf6fb590937e3e624e80>

The Unchessed design below is independent and deliberately smaller. It does not
copy Stockfish's feature-index map, network file format, update code, or
weights.

Chessformer treats the 64 squares as tokens, uses geometry-aware attention, and
an attention-based source/destination policy head. The ICLR 2026 paper reports
that this model family supports both engine strength and human move matching:

- <https://arxiv.org/abs/2605.19091>
- <https://github.com/CSSLab/maia3>

## Breakthrough candidate 1: factorized threat-residual NNUE

A direct SFNNv10-scale port is poorly matched to this engine. A roughly
80k-feature x 1,024 int16 transformer is large, expensive to refresh, and would
undo much of the latency work already completed. Unchessed XT-NNUE instead
keeps the proven fast positional path and adds a low-rank residual.

### Proposed v4 architecture

```text
HalfKAv2_hm positional features (22,528)
    -> incremental int16 accumulator, width 256

Relative occupied-target threat relations (32,400)
    -> refreshable int16 residual accumulator, width 32

For side to move and non-side to move:
    positional SCReLU + positional CReLU + threat clipped activation
    -> phase/material stack (8 stacks)
    -> sparse affine 16
    -> SCReLU and CReLU concatenation 32
    -> affine 32
    -> clipped activation
    -> scalar evaluation
```

The compact threat feature is:

```text
(attacker piece type + ownership: 12)
  x (target piece type + ownership: 12)
  x (relative file/rank delta: 15 x 15)
  = 32,400 dimensions
```

Both attacks and defenses are included. Repeated geometric relations are a
multiset, not accidentally deduplicated. The positional accumulator supplies
absolute square and king-bucket context, while the residual supplies explicit
piece interaction.

At width 32, the quantized residual table costs about 2.07 MB. Combined with
the existing 11.53 MB positional transformer, the runtime transformer tables
remain around 13.6 MB before small heads and alignment: dramatically below a
full 1,024-wide threat transformer.

### Why it may improve accuracy

The current linear SCReLU head must infer pins, overloaded defenders, hanging
pieces, batteries, and king-ring interactions indirectly from piece-square
features. Explicit occupied-target relations expose this information before the
small output head. Relative deltas preserve geometry while avoiding a table for
every absolute attacker/target pair.

### Why it may preserve speed

- the 256-wide positional path stays incrementally updated;
- the threat path is only 32 wide;
- active relations live in a fixed 256-entry stack buffer;
- a worst-case bounded refresh is 256 x 32 int16 additions;
- typical positions have far fewer occupied targets than attacked empty
  squares;
- scale 255 provides accumulator headroom and byte-friendly activations;
- the dense head should be fully integer and use AVX2/AVX-512 dot products;
- eight stack heads are selected once by material phase, not all evaluated.

A release-mode standalone microbenchmark over the committed twelve-position
suite and both perspectives observed 38.6 active relations per call and about
295 ns per full extraction plus synthetic 32-wide int16 accumulation on the
single-core sandbox host. This is promising for a relationship feature, but it
would still consume roughly a quarter of one core at 900k evaluations/s if run
at every node. The measurement excludes the output head and search effects and
is not an NPS or Elo result.

That finding makes a second optimization necessary: **selective residual
evaluation**. The base evaluator can serve ordinary non-PV pruning nodes; the
threat residual is enabled at the root, PV nodes, in-check nodes, and positions
near pruning margins. Alternatively, dirty-ray incremental updates must reduce
the amortized cost below 100 ns/node. Both approaches must be measured carefully
because using two static evaluators can destabilize search. They remain gated
experiments, not the default design.

### Incremental threat update plan

The first measurable implementation should refresh the 32-wide residual on
each child. Only after profiling should it add dirty updates. The dirty closure
contains:

1. moved-from, moved-to, captured, and en-passant capture squares;
2. kings and all leapers incident to those squares;
3. first sliders in each ray through every dirty square; and
4. relationships whose occupied target or line-of-sight status changed.

A full refresh fallback is mandatory when the closure is ambiguous. Exact
full-refresh versus dirty-update tests must cover castling, promotions, en
passant, discovered attacks, pins, and depth-three legal trees.

### Training objective

Use quantization-aware training with scale 255 and a multi-task objective:

```text
L = 0.55 * WDL cross entropy
  + 0.30 * robust teacher-score loss
  + 0.10 * pairwise move-order loss
  + 0.05 * threat auxiliary loss
```

Targets should combine a stronger teacher's WDL/score with game result. The
threat auxiliary task predicts attacked-piece value, hanging status, and king
zone pressure; it is discarded at inference. Train/validation splits must be
game-disjoint and future-month held out.

## Breakthrough candidate 2: persona-routed Chessformer/alpha-beta

The transformer should not replace alpha-beta globally. The useful hybrid is a
**root policy and persona architecture**, with alpha-beta retained as a tactical
and legality authority.

### Two model sizes

```text
CF-Lite-Human:
  64 tokens, 4 encoder blocks, d_model 96, 4 heads, FFN 192
  continuous Elo/time/persona embeddings
  source/destination policy + WDL head
  target: one root inference under short clocks

CF-Guide:
  64 tokens, 8 encoder blocks, d_model 192, 6 heads, FFN 384
  stronger policy/value distillation
  target: policy-guided analysis when clock permits
```

Both use dynamic geometry keys containing rank/file/diagonal relation, knight
and king geometry, actual source-to-target attack, reverse attack, occupancy,
and ownership. A small per-head embedding or MLP converts these keys into an
attention bias. The code now fixes this input contract.

### Backend routing

```text
Known/suspected engine, uncertain identity, target >= 2300,
FULL, PUNISH, or DEFEND:
    pure NNUE alpha-beta

Confident average human in MATCH, target <= 2100:
    Chessformer human-policy sample
    + shallow/full alpha-beta safety veto

Confident strong human or CLINCH:
    Chessformer policy-guided alpha-beta

Missing/unverified model:
    pure alpha-beta fallback
```

The backend is selected once per move and latched conservatively. Timing cannot
select a backend. Fixed `UCI_Elo` still has absolute strength precedence.

### Human-policy guarded mode

1. run CF-Lite once at the root with target Elo context;
2. mask illegal moves and obtain policy probabilities;
3. select a heavy-tail human-error target using the existing MATCH model;
4. retain policy-plausible legal candidates;
5. alpha-beta verify candidates at common depth/nodes;
6. veto forced mate, catastrophic SEE loss, or loss above the target envelope;
7. sample deterministically under `RandomSeed`.

This creates human-like choice without allowing a searchless model to hang a
queen or miss a forced mate merely because its policy assigns that move weight.

### Policy-guided alpha-beta mode

The transformer does not provide the final value. It supplies:

- root ordering and prior tie breaks;
- top-ply quiet move ordering after TT/captures;
- optional policy entropy for time allocation;
- a bounded extension hint for one or two high-prior moves; and
- interpretability data for persona logs.

The TT move, legal captures, killers/history, and tactical constraints retain
priority. Transformer inference should be cached by Zobrist key and target-Elo
bucket. Per-node transformer calls are prohibited in the first implementation.

## Required benchmark matrix

### XT-NNUE acceptance gates

| Gate | Requirement |
|---|---|
| exactness | scalar/SIMD and full/dirty accumulators exactly agree |
| speed | no more than 10% NPS regression before selective residual mode |
| accuracy | future holdout WDL/log-loss improves with bootstrap CI excluding 0 |
| quantization | mean score drift <= 3 cp, maximum <= 25 cp on frozen suite |
| search | best-move agreement reported, not used as sole quality evidence |
| strength | paired SPRT before becoming default |

### Chessformer hybrid gates

| Gate | Requirement |
|---|---|
| latency | CF-Lite p95 root inference fits the shortest supported clock tier |
| human fit | player-disjoint move-match and calibration by Elo/time band |
| safety | zero missed forced mates from the alpha-beta veto suite |
| GM/engine | router always chooses alpha-beta under engine/uncertain evidence |
| fixed Elo | clock-tier strength remains stable at each requested Elo |
| strength | policy-guided mode evaluated separately from human-direct mode |

## Concrete implementation sequence

1. **Completed in this batch:** threat feature extractor, square tokens,
   geometry keys, and conservative router tests.
2. Implement an `UNCHNNX4` quantized file inspector and trainer export.
3. Add a scalar v4 reference evaluator behind `EvalFile`; do not ship a random
   model.
4. Train the threat residual from the frozen v3 network and compare holdout
   loss.
5. Add SIMD integer head and profile full-refresh residual cost.
6. Implement `UNCHFORM` CF-Lite inference and a deterministic fixture model.
7. Train continuous-Elo human policy and package it only after provenance and
   player-disjoint validation.
8. Wire root policy guidance behind a default-off UCI option.
9. Run game gates; only then allow persona Auto routing.

The decisive design principle is asymmetric trust: Chessformer can propose and
humanize, but alpha-beta verifies and remains authoritative whenever strength,
identity, or tactics are uncertain.
