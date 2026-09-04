# The policy is well calibrated as a probability, not just a ranking

Prompted by the [Lc0 AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/),
whose PUCT rule is

    PUCT = Q + c_puct · P · sqrt(N_total) / (1 + N)

`P` — the policy's prior probability — enters **multiplicatively**. Its
magnitude matters, not merely the order of the moves.

Our root hint uses the policy in the weakest possible way: it sorts by logit
and discards the magnitudes (`roots.sort_by(...policy_hint...)` in
`search.rs`). Every prior analysis here measured ranking only — top-1
accuracy, rank of the best move, first-move regret. **Whether the magnitudes
carry information had never been tested.**

## Result: it is well calibrated, and slightly under-confident

`tools/analyse_policy_prior_calibration.py`, all 600 calibration positions,
17,998 legal moves. Predicted `softmax(logit)` versus how often that move
really is the teacher's best:

| predicted | n | predicted | actual | gap |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 17361 | 0.027 | 0.025 | −0.002 |
| 0.1–0.2 | 417 | 0.137 | 0.201 | +0.064 |
| 0.2–0.3 | 116 | 0.244 | 0.319 | +0.075 |
| 0.3–0.4 | 57 | 0.343 | 0.351 | +0.008 |
| 0.4–0.5 | 24 | 0.442 | 0.542 | +0.099 |
| 0.5–0.6 | 12 | 0.539 | 0.750 | +0.211 |
| 0.7–0.8 | 4 | 0.758 | 1.000 | +0.242 |

**ECE = 0.0048.** Every gap is positive above the lowest bin, so the policy is
*under*-confident: when it says 50% it is right 75% of the time. The fitted
optimal temperature is **T = 0.70** — below 1, meaning the distribution should
be *sharpened*, not softened. Tempering halves ECE to 0.0023.

This is unusual and worth stating plainly: over-confidence is the normal
failure mode for a neural policy. Ours errs the other way.

## Confidence carries real information

| | mean top-1 confidence | n |
|---|---:|---:|
| top move correct | **0.266** | 153 |
| top move wrong | **0.148** | 447 |

A separation of **+0.118**. The obvious confound is legal-move count: fewer
moves means more softmax mass per move *and* an easier position, which would
manufacture separation for free. Normalising each confidence by the uniform
baseline `1/n` removes it:

| | confidence ÷ uniform |
|---|---:|
| correct | **6.21×** |
| wrong | **4.26×** |

Separation survives at **+1.95×**. The signal is real, not an artifact of easy
positions.

## What this changes

This is the first clearly *positive* finding about the model in several
rounds, and it is narrow but genuine: **the policy's confidence is a usable
signal, and we currently throw it away.**

It does not resurrect the root hint. The negative SPRT results stand, and
`docs/unarchitectured-v1-why-the-hint-costs-elo.md` explains why — the cost is
charged to every move while the benefit lands on the cheapest search pass.
Calibration does not change that arithmetic.

What it does is tell us which *future* uses are structurally available:

- **Confidence-weighted schemes are sound in principle.** Modulating a
  reduction or a margin by policy confidence would be resting on a calibrated
  quantity, not on noise. Any such change is tree-altering and needs its own
  SPRT — this only says the input is not garbage.
- **If a prior is ever needed, sharpen it.** T = 0.70, not the raw softmax.
  Recorded so nobody has to re-derive it.
- **Low-confidence positions are identifiable.** The model knows when it does
  not know, which is exactly the property a fail-closed design wants.

## What is not applicable

**MCTS/PUCT itself.** Adopting Lc0's search would mean replacing alpha-beta
wholesale, and round 0 already established the shape of that problem: this
engine's strength comes from deep alpha-beta at ~100–250ms per move, while
PUCT needs a network evaluation at every expanded node. Our forward pass is
9.72ms. A few hundred nodes would consume the entire move budget. The
architecture is not a near-miss here; it is off by orders of magnitude.

**The `lc3` streaming design** (GatherWorker / EvalWorker / BackpropWorker
over event queues, sharded hash-map node repository) is a well-engineered
answer to a problem we do not have — batching NN evaluations across a tree
search. Our Lazy SMP threads share a transposition table and need no such
machinery.

**The pluggable search API** (v0.32+) is sound design for a project running
several search algorithms. We have one.

Recorded as reviewed-and-rejected rather than skipped, so the question does
not resurface as an obvious gap.

## Honest limits

- Calibration is measured against **Stockfish's best move at the labelling
  node count**, not against game outcomes. "Well calibrated as P(teacher-best)"
  is the claim; a PUCT prior ideally wants P(move is played by a strong
  agent), which is related but not identical.
- The high-confidence bins are **very thin** — 4 samples in 0.7–0.8, 1 in
  0.9–1.0. The reliability curve is trustworthy at the bottom (17,361 samples
  under 0.1) and indicative at the top. ECE is dominated by the low bins by
  construction.
- Temperature was fitted on the **same** 600 positions it is evaluated on, so
  T = 0.70 is an in-sample fit. Treat it as a starting point.
- Offline analysis; no games played.
