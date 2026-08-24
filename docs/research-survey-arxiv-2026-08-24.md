# arXiv survey — chess AI and LLM research, 2026-08-24

A broad sweep of arXiv across chess AI and LLM/efficiency research, triaged for
what could actually affect **this** engine: a from-scratch Rust alpha-beta UCI
engine with an NNUE evaluator, plus the default-off Unarchitectured v1
transformer.

## How to read this

Papers are graded by **actionability for this codebase**, not by how good the
paper is. Many excellent papers are graded low here purely because this engine
is CPU alpha-beta with a hard per-move time budget, which rules out most of what
is exciting in GPU-era ML.

| Grade | Meaning |
|---|---|
| **A** | Directly actionable; would change code or training here |
| **B** | Relevant context / worth reading before related work |
| **C** | Interesting, adjacent, but not actionable for this engine |
| **D** | Same keyword, different problem — recorded to prevent re-searching |

The blunt summary: **almost nothing in current LLM research transfers**, and the
few A-grade items are chess-specific or classical search papers. That is the
honest finding, not a failure of the search.

---

## Grade A — directly actionable

### A1. Study of the Proper NNUE Dataset (2024)
<https://arxiv.org/abs/2412.17948>

The single most actionable paper found. Proposes an algorithm for generating and
filtering **"quiet" positions** — stable, free of tactical volatility — for NNUE
training, and reports real engine-strength improvements.

Why it matters here: your NNUE data pipeline (`tools/train_nnue.py`,
`tools/check_nnue_data.py`) is the thing most likely to yield Elo per unit of
effort, and this is a documented methodology for exactly that. It pairs directly
with the bucketed-output-head idea already recorded in
`docs/research-notes-moe-2507.11181.md` — both are changes to the *next* NNUE
training run.

### A2. A New Paradigm for Minimax Search / MTD(f) (Plaat et al.)
<https://arxiv.org/abs/1404.1515> · <https://arxiv.org/abs/1702.03401> ·
<https://arxiv.org/abs/1505.01603>

The canonical MTD(f) papers. MTD(f) replaces the aspiration-window loop with
repeated null-window searches driven by transposition-table storage, and the
authors report it beating aspiration NegaScout on leaf nodes, total nodes, *and*
execution time in real tournament programs.

Why it matters here: `search.rs` currently uses aspiration windows
(`aspiration_delta`, `aspiration_min_depth`). MTD(f) is a drop-in alternative at
the iterative-deepening driver level — no eval or node-level changes — and is
one of the few well-documented search changes with measured node-count wins.

**Caveat:** MTD(f) is TT-traffic heavy and interacts badly with unstable evals
and with Lazy SMP. It changes the tree, so it needs an SPRT. Treat as a
well-evidenced experiment, not a free win.

### A3. Amortized Planning with Large-Scale Transformers: ChessBench (DeepMind)
<https://arxiv.org/abs/2402.04494>

10M games / 15B data points annotated by Stockfish 16 with **legal move and
action-value labels**. Trains transformers up to 270M params; the largest
reaches ~2895 Lichess blitz Elo with *no search*.

Two concrete uses:
1. **A ready-made teacher-labelled corpus.** Round 6 step 1 built a 600-position
   Stockfish-labelled calibration set by hand; ChessBench is the same idea at
   ~7 orders of magnitude more scale, already public.
2. **A calibration reference point.** They quantify how well Stockfish's search
   can be distilled into a feedforward net — directly comparable to the
   Unarchitectured v1 question of whether a 4.2M student can carry useful policy
   signal. Their answer: good but "perfect distillation is still beyond reach",
   which is consistent with the measured top-1 of 0.255.

### A4. A different take on the best-first game tree pruning algorithms (2019)
<https://arxiv.org/abs/1911.03388>

Practical experimental comparison of depth-first vs best-first pruning with
modern memory assumptions. Useful companion to A2 for deciding whether MTD(f) is
worth the SPRT slot before spending one.

---

## Grade B — relevant context

### B1. Neural Networks for Chess (Klein, 2022) — book-length
<https://arxiv.org/abs/2209.01506>

Complete technical introduction covering AlphaZero, Leela, **and NNUE
specifically**, plus minimax/alpha-beta/MCTS. The best single orientation
document for this codebase's exact intersection.

### B2. Human-aligned Chess with a Bit of Search (Allie, 2024)
<https://arxiv.org/abs/2410.03893>

Models human play *including non-move behaviours* (pondering time,
resignations), then uses a **time-adaptive MCTS** where search length depends on
how long humans would think. Achieves a 49-Elo average skill gap across
1000–2600.

Relevant because this engine has real adaptive/persona machinery (`adapt.rs`,
`policy.rs`, human/guide policy adapters). Adaptive *time allocation by position
criticality* is the transferable idea, and it does not require MCTS to use.

### B3. Otter: Time-Aware, History-Conditioned Human Chess AI (2026)
<https://arxiv.org/abs/2608.05206>

15.3M params, 55.23% top-1 human move prediction, beating Maia-2 with far fewer
parameters, trained on one T4. Conditions on last-20-move history and clock
pressure.

Directly comparable to Unarchitectured v1's design (history conditioning, time
class, Elo conditioning are all already in your input encoding). Useful as an
external accuracy yardstick for the same conditioning signals.

### B4. Tracking vs. Deciding: Dual-Capability Bottleneck in Searchless Chess Transformers (2026)
<https://arxiv.org/abs/2603.29761>

Formalises P <= min(T,Q): a move-sequence model must both track board state and
decide well, and these want *contradictory* data (low-rated games give tracking
diversity, high-rated games give decision quality). Introduces Elo-weighted
training. 120M params, Lichess bullet 2570, 55.2% top-1.

Worth reading before any further Unarchitectured v1 training: it is a concrete
argument about **training-data mixture**, which is a live question for a student
distilled from mixed-strength Lichess data.

### B5. Learning Models of Individual Behavior in Chess (Maia individual, KDD 2022)
<https://arxiv.org/abs/2008.10086>

Fine-tuning Maia to individual players; personalized models good enough to do
stylometry. Relevant to the persona/adapter design already in the codebase.

### B6. Evidence of Learned Look-Ahead in a Chess-Playing NN (Jenner et al., 2024)
<https://arxiv.org/abs/2406.00877> · follow-up <https://arxiv.org/abs/2505.21552>

Leela internally represents future optimal moves; a probe predicts the optimal
move 2 turns ahead at 92%. Evidence that policy nets encode more than one ply of
lookahead.

### B7. The Algorithm Is Not the Behavior (2026)
<https://arxiv.org/abs/2508.21380>

Leela often computes the correct solution in intermediate layers, then
*overrides* it in the final output due to learned safety priors; steering
recovers 61.7% of these "forgotten puzzles".

A useful caution for interpreting Unarchitectured v1's weak top-1: a low output
accuracy does not by itself prove the representation lacks the information.

### B8. FP8 Quantization: The Power of the Exponent
<https://arxiv.org/abs/2208.09225>

Concludes FP8 beats INT8 for post-training quantization when outliers are
present, and that the gap disappears under quantization-aware training.

Directly relevant to the documented failure of the int8 *activation* prototype
(1.01e-2 vs the required 5e-3). This paper's diagnosis — outliers — matches that
symptom and suggests the fix is QAT or a different format, not more tuning of
post-hoc scaling.

### B9. Integer or Floating Point? MoFQ (2023)
<https://arxiv.org/abs/2305.12356>

Optimal quantization format varies **per layer**; mixing formats layer-wise beats
committing to one. Applicable to `aegis_v4_runtime.rs`, which currently applies
one scheme uniformly.

---

## Grade C — interesting, not actionable here

- **Discovering High-Quality Chess Puzzles with Offline RL** (RLC 2026) —
  <https://arxiv.org/abs/2608.14851> — 1.5B puzzle-solving histories; pedagogy,
  not engine strength.
- **Hallucinations on the Board: ACT-Eval** — <https://arxiv.org/abs/2608.04240>
  — LLM chess *commentary* factuality. Only relevant if you ever ship natural-
  language annotation.
- **Engine-Equal, Human-Unequal** — <https://arxiv.org/abs/2607.25655> —
  engine-equal positions show reproducible human outcome skew. Fascinating, and
  arguably an argument for contempt/persona tuning, but observational.
- **UniMaia: Steering Chess Policies with Language** —
  <https://arxiv.org/abs/2605.27767>
- **Toward Modeling Player-Specific Chess Behaviors** —
  <https://arxiv.org/abs/2605.11893>
- **Three-Body Alignment** — <https://arxiv.org/abs/2607.21993> — notable for
  explicitly studying NNUE-rationalising commentary.
- **DeepChess** — <https://arxiv.org/abs/1711.09667> — historical; the
  comparison-based eval predates NNUE.
- **Policy Gradient Steering** — <https://arxiv.org/abs/2607.27574>
- **When Does LLM Orchestration Pay Off?** —
  <https://arxiv.org/abs/2608.00685> — uses chess puzzles as a benchmark;
  conclusion (orchestration costs 2-4x tokens for ~4.6pp) is a useful general
  caution about paying compute for marginal accuracy.
- **EIE: Efficient Inference Engine (retrospective)** —
  <https://arxiv.org/abs/2306.09552> — sparsity/compression; hardware-oriented.
- **ABQ-LLM** — <https://arxiv.org/abs/2408.08554> — arbitrary-bit quantized
  inference, GPU/TensorCore-specific.
- **SIMD engineering references** (useful as technique, not as results):
  AVX-512 lattice QCD <https://arxiv.org/abs/1811.00893>, SIMD pseudo-Verlet
  lists <https://arxiv.org/abs/1804.06231>, SIMD modular arithmetic
  <https://arxiv.org/abs/2004.11571>, SSE/AVX performance
  <https://arxiv.org/abs/1211.0820>.
- **SimdBench** — <https://arxiv.org/abs/2507.15224> — benchmarks LLMs at
  *writing* SIMD intrinsics.

---

## Grade D — keyword collisions, recorded so nobody re-searches them

The `"NNUE"`/`"efficiently updatable"` query is heavily polluted: **online
hashing** (<https://arxiv.org/abs/1911.12125>), **covariance matrix updates**
(<https://arxiv.org/abs/2002.08831>), **truss/frame redundancy matrices**
(<https://arxiv.org/abs/2205.12264>), **FlexFlood learned indexes**
(<https://arxiv.org/abs/2411.09205>), **DGAI graph ANN indexes**
(<https://arxiv.org/abs/2510.25401>).

Likewise `"alpha-beta"` returns Finsler metrics
(<https://arxiv.org/abs/1209.0857>), skew characters
(<https://arxiv.org/abs/0806.1879>), and set theory
(<https://arxiv.org/abs/math/9706223>). Use `abs:` + a chess/game term.

---

## The LLM-side finding, stated plainly

I searched speculative decoding, test-time compute, MoE, distillation,
quantization, and sparse activation. **Essentially none of it transfers**, for
one structural reason:

> LLM efficiency research optimizes *autoregressive token generation*, where a
> large model is the bottleneck and verification can be batched. This engine's
> bottleneck is a **sequential game-tree search with a hard per-move deadline**,
> where the evaluator is already small and is called millions of times.

Concretely:
- **Speculative decoding** (<https://arxiv.org/abs/2402.01528>,
  <https://arxiv.org/abs/2504.15475>, <https://arxiv.org/abs/2607.08690>) has no
  analogue: alpha-beta already *is* an exact verifier, and the search is the
  cost, not the drafting.
- **Test-time compute scaling** is what iterative deepening already does, with
  better-understood time management.
- **MoE** — assessed separately in
  `docs/research-notes-moe-2507.11181.md`; rejected except as output-head
  bucketing.
- **Distillation** is already the Unarchitectured v1 design (58M oracle → 4.2M
  student), and round 6 measured the result: top-1 0.255. The blocker is
  latency-vs-value, not distillation technique.

The two genuine LLM-side takeaways are both about **quantization** (B8, B9), and
both point the same direction: the failed int8-activation experiment probably
needs quantization-aware training or a per-layer format choice, not more
post-hoc calibration.

## Recommended reading order

1. **2412.17948** (NNUE datasets) — most likely Elo per unit effort.
2. **1404.1515 / 1702.03401** (MTD(f)) — the one search change with real
   published node-count evidence.
3. **2402.04494** (ChessBench) — free large-scale teacher-labelled data.
4. **2603.29761** (dual-capability bottleneck) — before any further
   Unarchitectured v1 training.
5. **2208.09225** (FP8) — before retrying int8 activations.

## Priority note

None of this outranks the outstanding engineering task: the SIMD/search work in
`docs/performance-round-1-implementation.md` is **written but never compiled**,
because no Rust toolchain is reachable from this sandbox. Validating that on a
machine with `cargo` is worth more than any paper in this survey.
