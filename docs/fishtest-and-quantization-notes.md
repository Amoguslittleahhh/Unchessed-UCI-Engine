# Fishtest statistics and Stockfish quantization: what applies here

Two independent findings from the official Stockfish documentation, each with
a concrete gap in this repository and a tool or measurement closing it.

## 1. Every result this project has recorded is trinomial

The [Fishtest mathematics page](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
states that Fishtest analyses results with the **pentanomial** model, scoring
each *game pair* on five outcomes (LL, LD, DD/WL, WD, WW) rather than treating
2N paired games as independent. Fishtest's own claim for this: it "leads to a
substantial saving of testing resources".

Round 7's batches were run with `cutechess-cli -repeat -games 2`, so they
**were** paired — the same opening played once with each colour. The pairing
information existed and was discarded at analysis time. Nothing in this
repository could compute a pentanomial interval; `grep` for
`pentanomial|ptnml|normalized elo` returned one hit, in a list of unstarted
research ideas.

`tools/pentanomial_sprt.py` closes that. It computes the pentanomial score,
Elo with a confidence interval, **normalized Elo** (Fishtest's scale-free
measure, which makes expected test duration depend only on the bounds and not
on the draw ratio or book), the SPRT log-likelihood ratio against
`elo0`/`elo1`, and the variance ratio against the trinomial view. It reads
pair counts directly, or derives them from a cutechess PGN.

**Validation against real data.** Round 7's three batches were produced
independently, on the reviewer's hardware, and reported as trinomial W-L-D
with Elo bands. The tool's trinomial path reproduces all three exactly:

| run | reported Elo | ours | reported ± | ours ± |
|---|---:|---:|---:|---:|
| MinTime=1000, tc=5+0.05 | −26.1 | **−26.1** | 22.4 | **22.4** |
| replication | −15.1 | **−15.1** | 22.5 | 22.6 |
| MinTime=30000, tc=60+0.6 | −5.8 | **−5.8** | 27.7 | 27.8 |

Matching three independently-produced figures to 0.1 Elo is good evidence the
implementation is right, not merely self-consistent.

**What it does not do.** It cannot retrospectively make round 7's results
pentanomial — the raw pair outcomes were not preserved, only the W-L-D
aggregates, and the mapping is one-way. Any *future* run can use `--pgn` to
get the pentanomial analysis directly. The tool is also a post-hoc analysis:
a real sequential test evaluates the LLR as games arrive and stops at a
boundary, so applying these bounds to a finished batch does not carry the same
error guarantees. That caveat is in the tool's own docstring.

## 2. Our weights violate the int8 range Stockfish's scheme requires

`docs/unarchitectured-v1-runtime-optimization.md` records a rejected int8
activation prototype: it failed the parity gate at `1.01e-2` against a
required `5e-3`, and was documented rather than having the tolerance loosened.
The reason it failed was never established.

The [quantization section](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#the-math-of-quantization-and-how-to-make-it-fit)
gives the scaling algebra — for a layer feeding a ClippedReLU, scale bias by
`s_A · s_W`, weights by `s_W`, divide output by `s_W` — and, more usefully,
the [trainer section](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#accounting-for-quantization-in-the-trainer)
names the failure mode directly:

> Adding (quite aggressive) quantization has reduced the possible range of
> values for the weights and biases. […] The problematic cases are the int8
> weights of the linear layer, which for example in Stockfish can only go to
> about 2 (activation range in 0..1). This is potentially a big problem, as
> the training can diverge from the quantized representation by more than just
> rounding.

Their fix is **weight clipping during training** — clamping to `±127/64`
(≈1.984) after every optimizer step, so the trained weights stay inside what
int8 can represent.

**Our trainer has gradient clipping (`clip_grad_norm_`) but no weight
clipping.** So the weights were never constrained to a quantizable range.
Measured on the real exported checkpoint against Stockfish's `±127/64`:

| tensor | max abs | over limit |
|---|---:|---:|
| history_promotion.weight | 2.45 | 7.50% |
| piece_embedding.weight | **4.01** | 5.29% |
| history_position.weight | 3.10 | 5.08% |
| history_from.weight | 3.23 | 4.83% |
| halfmove_embedding.weight | 3.67 | 4.74% |
| castling_embedding.weight | 3.68 | 4.69% |
| square_embedding.weight | **4.09** | 4.56% |

Peak magnitude is **4.09 — 2.06x the int8 limit**, and the worst tensor has
7.5% of its weights out of range. Full per-tensor data in
`benchmarks/unarchitectured-v1/int8-weight-range-2026-08-25.json`.

This is a quantitative explanation for the `1.01e-2` parity failure that was
previously unexplained: post-hoc quantization of weights that exceed the
representable range by 2x cannot be fixed by choosing a better calibration
scheme, because the information is already outside the format. The prior
conclusion — that the fix is quantization-aware training, not a smarter
post-hoc scheme — is confirmed, and now has a specific mechanism and a
specific remedy.

Note the largest offenders are **embedding** tables, not the linear layers
Stockfish's `±127/64` bound is about. The exact bound is not directly
transferable; what transfers is the principle (weights must be constrained
during training to whatever the target format represents) and the observation
that ours are unconstrained. A real int8 effort would need to derive its own
per-tensor bounds from the chosen scheme.

## What this changes

- **Any future SPRT should be analysed pentanomially.** The runner already
  produces paired games; only the analysis was missing. Use
  `tools/pentanomial_sprt.py --pgn <out.pgn> --engine Hint`.
- **The int8 backlog item should be re-scoped.** The remaining-work list in
  `docs/unarchitectured-v1-runtime-optimization.md` describes "calibrated int8
  activations […] a different calibration scheme might close that gap". This
  evidence says a calibration scheme will *not* close it on its own — the
  weights are out of range before calibration begins. The prerequisite is
  weight clipping during a retrain.
- **Nothing here justifies enabling anything.** No behaviour changed;
  `UnarchitecturedHint` remains default-off and `runtime_safety_suite` remains
  false.

## Honest limits

- The pentanomial tool has **not been run on a real match** — no
  `cutechess-cli` in this sandbox. It was validated by reproducing round 7's
  trinomial figures, by a symmetric-input sanity check (exactly 0.0 Elo, 50%
  LOS), and by a synthetic paired PGN with known pair outcomes.
- The weight-range measurement is a **static property of the exported
  checkpoint**. It shows the weights are outside int8 range; it does not by
  itself prove that clipping during training would have produced a net of
  equal strength. That needs a retrain, which needs its own SPRT.
- `±127/64` is Stockfish's bound for its own architecture and activation
  range. It is used here as a reference point, not as a claim about what our
  format would require.
