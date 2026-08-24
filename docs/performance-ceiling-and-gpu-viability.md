# How fast can this engine actually get, and is a GPU worth it?

Research note answering two questions: (1) what is the realistic upper limit on
speed *without* giving up depth or move quality, and (2) is GPU computation
viable for this engine.

Short answers:

1. **Roughly 1.8x more per-node speed is available**, and the single biggest
   remaining win is **int16 quantization of the NNUE**, not wider SIMD. Beyond
   that there is a hard Amdahl ceiling of about **2.7x** even with a
   *free* evaluator, because search bookkeeping then dominates.
2. **A GPU is not viable for the main search, and would make it slower.** The
   numbers below are decisive, not marginal. There is exactly one narrow GPU use
   case (offline training), which you already do.

## Measurement conditions

All figures are `gcc -O2` C microbenchmarks of the same operation shapes, run on
this sandbox: **2 logical CPUs, Intel Xeon @ 2.60GHz**, with AVX2, FMA, **and
AVX-512** (`avx512f/dq/bw/vl/vnni`). They size the operations; they are **not**
engine NPS measurements and will not transfer 1:1. No Rust toolchain is
reachable here, so nothing was compiled in-tree (unchanged from prior rounds).

---

## Part 1 — where the remaining speed is

### 1a. The big one: int16 quantization beats wider vectors

Measured, NNUE output layer (ACC=256, both perspectives):

| Implementation | Time | vs scalar |
|---|---:|---:|
| scalar f32 (pre-round-1) | 594.1 ns | 1.0x |
| **AVX2 f32** (what round 1 shipped) | 67.6 ns | 8.8x |
| AVX-512 f32 | 50.1 ns | 11.9x |
| **AVX2 int16** | **13.5 ns** | **44x** |
| AVX-512 int16 | 11.2 ns | 53x |

Accumulator update (one feature row):

| Implementation | Time |
|---|---:|
| scalar f32 | 28.16 ns |
| **AVX2 f32** (round 1) | 13.91 ns |
| AVX-512 f32 | **26.77 ns** |
| **AVX2 int16** | **7.02 ns** |
| AVX-512 int16 | 13.67 ns |

Two conclusions, both important:

**AVX-512 is a trap on this CPU.** For the accumulator update it is *slower than
AVX2* (26.8 vs 13.9 ns) — the classic AVX-512 downclocking penalty, and exactly
what `docs/unarchitectured-v1-runtime-optimization.md` already found for the
Chessformer VNNI experiment. Do not "upgrade" the round-1 kernels to AVX-512.

**int16 at AVX2 width beats f32 at AVX-512 width, by a lot** (13.5 vs 50.1 ns).
Half-width data means twice the lanes per register *and* half the memory
traffic. This is why every strong engine quantizes. Stockfish's own NNUE
documentation is blunt about it: floating point "is not an option for achieving
maximum engine strength as it sacrifices too much speed for too little accuracy
gains."

Secondary benefit: the feature transformer halves from **23.1 MB to 11.5 MB**.
That is the difference between thrashing and mostly fitting in this host's 54 MB
L3, and it shrinks the 2 KB `EvalState` copied per node to 1 KB.

### 1b. `apply_diff` scans 12 piece planes when 1-2 change

`nnue.rs:543` loops `c in 0..2, p in 0..6` unconditionally, computing two
bitboard diffs per plane on every move, for both perspectives. A quiet move
touches exactly one plane; a capture, two.

| Plane scan (per perspective, per move) | Time |
|---|---:|
| all 12 planes (current) | 52.77 ns |
| dirty planes only | 14.07 ns |
| | **3.75x** |

At two perspectives that is roughly **77 ns wasted per move**, which is on the
order of the entire output layer. The move already knows which piece moved and
what it captured, so the dirty set is free to compute — it does not need a
bitboard scan at all.

### 1c. Realistic combined model, and the ceiling

Per-node cost model on this host (movegen/make/TT/ordering estimated at 120 ns
and **not** measured — flagged as the weakest number here):

| Component | Now (round 1) | With int16 + dirty planes |
|---|---:|---:|
| NNUE output layer | 67.6 | 13.5 |
| accumulator update (x2) | 27.8 | 14.0 |
| `apply_diff` scan (x2) | 105.6 | 28.2 |
| movegen / make / TT / ordering | 120.0 | 120.0 |
| **total** | **321.0** | **175.7** |

**≈1.8x further speedup**, with NNUE dropping from ~63% of a node to ~32%.

And then the ceiling:

> Even an **infinitely fast** evaluator only yields **2.67x** from here, because
> movegen, make, TT probes and move ordering are untouched.

That is the honest upper limit of evaluator optimization. After int16, further
eval work has almost nothing left to win, and effort should move to search
bookkeeping (or stop).

### 1d. What "retaining full quality" costs

int16 quantization is **not** free the way the round-1 SIMD work was. Round 1
was bit-exact or summation-order-only; quantization changes numbers.

- Stockfish-lineage nets quantize with **QAT-style scaling** chosen at training
  time, not by rounding a trained f32 net afterwards. Post-hoc rounding of your
  existing v3 net would lose accuracy for no good reason.
- Your own history confirms this: the int8 *activation* prototype for
  Unarchitectured v1 failed parity at 1.01e-2 vs the required 5e-3, and the
  arXiv survey's FP8 paper (2208.09225) diagnoses precisely this — outliers —
  concluding the difference vanishes under quantization-aware training.
- Therefore int16 NNUE is a **retraining project** (trainer emits quantized
  weights, file format v4, then SPRT), not a runtime patch. Same conclusion as
  the bucketed output heads, and the two should ship in the *same* retrain.

The dirty-plane fix (1b), by contrast, is a pure refactor with identical
arithmetic and can be done immediately.

---

## Part 2 — GPU viability

### 2a. For the main alpha-beta search: no, and it is not close

Two independent reasons, either one fatal.

**Reason 1: latency.** A CPU NNUE eval after the work above is ~20 ns. An
optimistic GPU round trip (launch + transfer + sync) is ~5 µs even for a
trivial payload.

| | per eval |
|---|---:|
| CPU NNUE (int16, AVX2) | ~20 ns |
| optimistic GPU round trip | ~5,000 ns |
| | **250x slower** |

At `tc=5+0.05` (~150 ms/move) that is ~8 million CPU evals versus ~30,000
sequential GPU round trips — the GPU version would not finish a shallow search.

**Reason 2: alpha-beta is sequential, so you cannot batch.** Amortizing one 5 µs
round trip needs a batch of ~250 evaluations. But alpha-beta decides *which node
to visit next based on the score of the previous one* — that is what alpha-beta
pruning **is**. You cannot batch positions you have not yet decided to search.
Trying to fill a batch means speculatively evaluating nodes the search would
have pruned, which throws away the exponential saving that makes alpha-beta
work in the first place.

This is why Leela (GPU, big net, MCTS) and Stockfish (CPU, tiny quantized net,
alpha-beta) have the architectures they do. **The evaluator and the search
algorithm are a matched pair.** Bolting a GPU onto alpha-beta gets the worst of
both.

**Reason 3, specific to you: round 0 already ran this experiment.** The
Unarchitectured v1 root hint added ~89 ms per move and lost **0-20-0**. A GPU
round trip is smaller than 89 ms but the mechanism is identical: fixed per-move
latency stolen from the move clock. That failure is the empirical version of
this argument.

### 2b. Where a GPU *is* legitimately useful

- **Training.** Already the case — `train_nnue.py` / the A100 scripts. If you
  retrain for int16 + bucketed heads, that is GPU work.
- **Offline dataset labelling.** Bulk-annotating positions is embarrassingly
  parallel and latency-insensitive. Note the arXiv survey found
  **ChessBench** (2402.04494), 15B Stockfish-labelled data points, so much of
  this is already done for you.
- **A different engine.** A Leela-style MCTS engine with a big net genuinely
  needs a GPU — MCTS batches naturally because it expands many leaves before
  needing any result. That is a rewrite, not an optimization.

### 2c. The honest caveat

**There is no GPU in this sandbox** (`/dev/nvidia*` absent, no `nvidia-smi`,
`rocm-smi`, or `clinfo`), so the 5 µs round-trip figure is a literature-standard
estimate, not something measured here. It would have to be off by **two orders
of magnitude** to change the conclusion, which is why I am comfortable stating
it plainly — but it is an estimate, and it is labelled as one.

---

## Recommended plan

**Now (no retraining, safe):**
1. **Dirty-plane `apply_diff`** — ~77 ns/move, identical arithmetic, no SPRT
   risk beyond normal review.
2. **Compile and test the round-1 SIMD work.** It is still unvalidated because
   no toolchain is reachable here. Nothing below matters until this is done.

**Next retrain (one training run, all three together):**
3. **int16 quantization** (QAT-style, chosen at training time) — the single
   biggest remaining win, ~5x on the output layer and half the memory.
4. **8 piece-count output buckets** (from the MoE note) — free router, standard
   in strong engines.
5. File format v4 covering both, then SPRT against the current net.

**Do not do:**
- AVX-512 kernels — measured *slower* than AVX2 on this class of CPU.
- Post-hoc rounding of the existing f32 net to int16 — quantize during training
  or not at all.
- **Any GPU work for the main search.**

## Cross-references

- `docs/performance-round-1-implementation.md` — the SIMD work awaiting `cargo`.
- `docs/performance-survey-2026-08-24.md` — the original survey; items 5-8 there
  (SEE, ProbCut, `gives_check`, prefetch) target the 120 ns "rest of node" term
  that becomes dominant after this work.
- `docs/research-notes-moe-2507.11181.md` — bucketed output heads.
- `docs/research-survey-arxiv-2026-08-24.md` — FP8/QAT evidence (2208.09225),
  NNUE dataset construction (2412.17948), ChessBench (2402.04494).
