# GAB is load-bearing, and ours is 4x under-provisioned

Findings from checking Unarchitectured Metal against
[arXiv:2605.19091](https://arxiv.org/abs/2605.19091) (*Chessformer: A Unified
Architecture for Chess Modeling*, Monroe, Eilender, Chalmers, Tang, Anderson).

This is the canonical reference for the architecture this project's model
derives from — square tokens, an attention-based source-destination policy
head, and **Geometric Attention Bias (GAB)**, a dynamic positional encoding
generated from a compressed board representation and mixed from learned
templates. Our `unarchitectured_metal_runtime.rs` implements exactly this: `gab.templates`
of shape `(32, 64, 64)`, per-layer coefficient projections, biases added to the
dot-product logits before softmax.

The paper's central ablation claim is that **GAB is the key driver** of the
architecture's advantage over absolute and relative position encodings, for
Elo, puzzle accuracy, and policy accuracy alike.

## Finding 1: our GAB is smaller than any configuration in the paper

Read directly from the exported checkpoint:

Note these are the **student's** dimensions, read from the shipped artifact.
`config/unarchitectured_metal_training.json` lists `gab_token_projection: 16,
gab_hidden: 64, gab_templates: 64`, but those sit under an `oracle` key and
describe the 58M teacher. The student that actually ships is smaller than its
own teacher on every GAB axis as well as smaller than the paper.

| dimension | ours (student) | our oracle | paper 5M | paper 23M / 79M |
|---|---:|---:|---:|---:|
| d1 (token projection) | **8** | 16 | 32 | 32 |
| d2 (compress) | **32** | 64 | 64 | 128 |
| d3 (templates) | **32** | 64 | 64 | 128 |

Our `d1` is **4x smaller than the smallest configuration the paper reports**,
and `d2`/`d3` are half. The paper's 5M model is comparable in scale to our
4.2M student, so this is not a fair-for-size tradeoff — it is a smaller GAB on
a similarly sized model.

## Finding 2: GAB is genuinely load-bearing here

`tools/analyse_gab_contribution.py` ablates GAB at inference time on the real
exported package, over all 600 calibration positions:

| variant | top-1 | mean regret (cp) | Δ top-1 |
|---|---:|---:|---:|
| baseline | 0.2683 | 146.3 | — |
| GAB zeroed | 0.2100 | 178.5 | **−0.0583** |
| GAB templates shuffled | 0.2100 | 183.0 | **−0.0583** |

Zeroing GAB costs **21.7% of top-1 accuracy** (0.2683 → 0.2100) and adds 32cp
of mean first-move regret. Shuffling the template bank — structurally intact
bias, semantically wrong — is just as damaging, and slightly worse on regret.
So the contribution is from the *learned template content*, not from the mere
presence of a bias tensor of the right shape.

For scale: the model's entire margin over the free MVV-LVA heuristic is
+0.100 top-1 (see `docs/unarchitectured-metal-theme-breakdown.md`). GAB accounts
for 0.058 of that. **Remove GAB and most of the model's advantage over a
zero-cost heuristic disappears.**

## Why this matters

Rounds 2–4 pushed the forward pass from 89ms to ~9.7ms and concluded speed had
hit diminishing returns. Rounds 7–8 established the hint never trended
positive across four SPRT batches, and
`docs/unarchitectured-metal-why-the-hint-costs-elo.md` showed the cost is
structural rather than a quality failure.

This adds a *capacity* explanation to sit beside those. The component the
paper identifies as the key driver, and which measurably carries a fifth of
our model's accuracy, is provisioned at a quarter of the paper's smallest
setting. That is a concrete, named architectural deficiency — considerably
more actionable than another round of kernel micro-optimisation.

It also sharpens the retrain guidance from
`docs/unarchitectured-metal-theme-breakdown.md`. That doc said a retrain should
target quiet positions, mates and forks by theme-balanced sampling. This adds
a second, independent lever: **widen GAB to at least the paper's 5M
configuration** (d1=32, d2=d3=64).

Both are retrain-only changes. Neither can be validated without one, and a
retrained net would still need its own SPRT gate.

## Honest limits

- **These are inference-time ablations on frozen weights.** They measure how
  much the *trained* model relies on GAB. They do **not** show that a larger
  GAB would score better — that requires a retrain, and a model trained
  without GAB might partly compensate elsewhere. The capacity comparison is a
  motivated hypothesis, not a measured gain.
- **No games.** This is offline analysis on the calibration corpus; no
  `cutechess-cli` in this sandbox.
- The paper's `d1/d2/d3` are read from its Appendix A.1, which specifies
  average pooling for the 5M configuration; our implementation flattens
  instead. The dimensional comparison is still meaningful, but the two are not
  identical designs.
- Corpus provenance is source-population-disjoint only, per
  `docs/unarchitectured-metal-calibration.md`.

## A measurement bug this caught

The first version of this tool scored a hit as `chosen == argmax_string`,
giving a baseline of 0.2550 against the 0.2683 already committed in
`benchmarks/unarchitectured-metal/ordering-risk-2026-08-24.json` — same
checkpoint, same 600 positions.

The cause: **32 of the 600 positions have two or more moves tied at the top
teacher score.** Picking any of them is equally correct, but string comparison
marks all but one arbitrary winner wrong. The earlier tool scored zero regret,
which is the correct definition. Fixed to match, and the baseline now
reconciles exactly at 0.2683 / 146.3cp.

Worth recording because the discrepancy was small enough to shrug off, and
shrugging it off would have left two committed artifacts quietly disagreeing
about the same measurement.

## The Stockfish commit

`official-stockfish/Stockfish@f4bcd40` (SFNNv16, net `nn-89cb98a217f7.nnue`)
was reviewed alongside. It adds **pawn-pair features** — NNUE inputs indexed
by pairs of pawns on the same or adjacent files ("3-wide") — and removes the
now-redundant pawn–pawn threat inputs. Passed LTC, VLTC and VVLTC; failed STC;
~3.5% slowdown.

**Not portable to this engine as-is.** It is a feature-set change to the NNUE
input layer, so it only has meaning together with a net trained on those
features. Our shipped net is `UNCHNNUE` v3 with `ft_in 22528`; adopting
pawn-pair inputs would mean designing the feature indexing, changing the
format, and retraining from scratch — the same bar as the int16 work in
`docs/performance-ceiling-and-gpu-viability.md`, and with no reason to expect
Stockfish's Elo result to carry over to a different architecture and net.

The `search.cpp` hunk in that commit is not an algorithmic change: it bundles
`DirtyPiece`/`DirtyThreats`/`DirtyPawnPairs` into one `Dirties` struct and
updates the call site. Nothing to port.

Recorded as reviewed-and-rejected rather than silently skipped. The idea is
sound and worth remembering if this project ever retrains its NNUE: pawn
structure interactions are cheap to index and Stockfish measured a real gain
from them at long time control.
