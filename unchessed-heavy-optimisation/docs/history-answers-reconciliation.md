# Reconciling the project-history answers

The review of my six questions reported that several figures I cited "don't
exist anywhere in this repository" and suggested tracing whether they came
from another repo or an uncommitted session. This records what the
discrepancy actually was, so it is not re-litigated.

## Root cause: the searches ran against `main`, and my branch was never merged

Everything reported missing is present and pushed on
`arena/01a02efe-unchessed-uci-engine` @ `e2f155f`:

| reported missing | actual location |
|---|---|
| `docs/fishtest-and-quantization-notes.md` | present, line 87 |
| "2.06x the int8 limit" | `docs/fishtest-and-quantization-notes.md:87`, and `benchmarks/unarchitectured-metal/int8-weight-range-2026-08-25.json` (`headroom_ratio: 2.062`) |
| rating-inertness sweep (0/200) | `docs/rating-conditioning-finding.md`, `benchmarks/unarchitectured-metal/rating-conditioning-2026-08-25.json`, `tools/analyse_rating_conditioning.py` |
| "21.7%" GAB ablation | `docs/gab-capacity-finding.md:45`, `benchmarks/unarchitectured-metal/gab-ablation-2026-08-25.json` |

Verified with `git cat-file -e FETCH_HEAD:<path>` for each, and confirmed
absent from `origin/main` — which is consistent with the review, not in
conflict with it. `main` is at `3995f48`; the last merge from this branch was
round 8 (`ffffc29`). Six subsequent commits have never been reviewed or
merged:

```
e2f155f  matetrack suite + engine commit feed review
2739558  policy prior calibration (ECE 0.0048)
b84a3e3  rating input does nothing (0/200)
dfb51c2  GAB load-bearing, 4x under-provisioned
19aee42  pentanomial SPRT + int8 parity explanation
c560ad8  theme tagging (5.5x accuracy spread)
9139f9b  root-hint pairing guard
```

Nothing came from another repo or an untracked session. **The queue is just
seven commits deep.**

## Two substantive corrections I accept

### Q2 — I conflated two models when phrasing the question

The review is right that the shipped **NNUE is f32 and has no int8 runtime
path**, and that int8 applies only to the Unarchitectured Metal package.

My measurement was in fact taken on the Unarchitectured Metal checkpoint — the
tensors in the artifact are `blocks.0.qkv`, `blocks.0.down`, `blocks.0.up`
and so on, plus the embedding tables. It never touched `unchessed-nnue.bin`.
So the number is real and correctly scoped, but I described it in the
question as if it concerned NNUE training, which was wrong and produced a
genuinely unanswerable question. The doc itself is scoped correctly; the
question I asked was not.

The substance stands: the failed int8 activation prototype was an
Unarchitectured Metal experiment, weights there peak at 2.06x the `±127/64`
reference bound, and the trainer has `clip_grad_norm_` but no weight
clipping. `±127/64` is Stockfish's bound for their architecture, used as a
reference point — the doc already says so.

### Q5 — I quoted the student's dimensions, the review quoted the oracle's

Both are correct, for different models. `config/unarchitectured_metal_training.json`
holds `gab_token_projection: 16, gab_hidden: 64, gab_templates: 64` — but
those live under an **`oracle`** key, alongside `board_layers: 16` and
`decoder_layers: 4`. That is the 58M teacher.

The shipped **student** artifact says otherwise. Read directly from
`artifacts/unarchitectured-metal-final.unmetal`:

```
gab.token_projection   (8, 256)     -> d1 = 8
gab.compress.weight    (32, 512)    -> d2 = 32
gab.templates          (32, 64, 64) -> d3 = 32
```

So the student is **8/32/32**, the oracle is **16/64/64**. My finding is about
the student, which is the model that actually ships and the one every
measurement in this project uses. The gap is real, and slightly larger than
I first wrote: the student's `d1` is 4x below the paper's smallest
configuration and 2x below its own teacher's.

`d1=32, d2=d3=64` was the *Chessformer paper's 5M configuration*, quoted as
the comparison target, not a claim about our config.

## Q5, second half — the paper is genuinely external

Correct that this repo does not cite Chessformer's ablation numbers.
`ChessformerWeights` is an internal Rust type name; the architecture is
paper-derived but the repo contains no external figures.

The 21.7% is **our own measurement**, not a paper figure: zeroing GAB drops
top-1 from 0.2683 to 0.2100 on the 600-position corpus. What I wanted from
the paper was their comparable ablation delta, to tell whether 21.7% is
normal or low. That remains an external lookup and is still open.

## Q4 — the shared-code inference is the useful part

The review confirms oracle and student use identical rating-conditioning
code (`values + normalized_rating * rating_weight + rating_bias`). Combined
with the measured result — the student's rating input moves the chosen move
in **0 of 200** positions across 600→3200 — the conclusion the review draws
is the right one: identical architecture plus an inert student points at
training or distillation rather than architecture.

That is a better-specified hypothesis than I had, and it narrows what a
retrain would need to check. Whether the oracle is *also* inert is still
untested; the oracle checkpoint is not in this repo.

## Q1, Q3, Q6 — accepted as confirmed gaps

Nothing to add. The provenance manifest does not exist,
`UnarchitecturedMinTime=30000` is an unmeasured round number with no stated
rationale, and no technical failure writeup exists for Apex/Hydra. The
`train_nnue.py` v3→v4 writeup being cited as the counter-example is a useful
pointer to what good looks like.

The Q3 answer changes something concrete: if an isolated retest happens,
30000 should be treated as one candidate point to test rather than a value
with prior justification to defend.

## Action

The seven-commit backlog needs review or rejection. Reviewing against `main`
will keep reporting these findings as nonexistent until then.
