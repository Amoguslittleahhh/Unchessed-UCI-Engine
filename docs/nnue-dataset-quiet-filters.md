# NNUE dataset quiet-position filters

Implements the two filters from Tan & Watkinson Medina, *Study of the Proper
NNUE Dataset* ([arXiv:2412.17948](https://arxiv.org/abs/2412.17948)), in
`unchessed-datagen`.

## Why

An NNUE evaluation is a *static* function: one forward pass, no search. So it
can only ever learn a position whose true value is visible without searching.
If the training label came from a search but the position has a tactic in it,
the label encodes something the network structurally cannot see, and the
network is forced to fit noise.

The paper reports the concrete symptoms of training on unfiltered data:

- training fails to converge, and MSE stays materially higher;
- slower training for a worse final result;
- the resulting engine "randomly sacrifices pieces for no good reason",
  miscounts material, and hesitates at crucial moments.

The paper also notes that the existing NNUE SIMD literature covers *inference*
in detail and gives essentially **no** guidance on dataset construction, which
is why this gap is easy to miss.

## What was missing here

Our datagen already had proxies for quietness, but not the two real tests.
Pre-existing filters: `NNUE_MIN_PLY=10`, `NNUE_PLY_GAP=4`,
`NNUE_MAX_PER_GAME=12`, `NNUE_ACCEPT=0.9`, an `in_check` skip, a mate /
`NNUE_MAX_ABS_SCORE=2000` reject, and a skip when the *best move* is a
capture, en passant, or promotion.

That last one is the closest thing to the paper's first filter, but it is
strictly weaker: it inspects only the single best move, not the size of the
evaluation swing. A position where the best move is quiet but the *second*
move hangs a rook passed the old filter and is exactly what the paper says to
drop. `grep` for `qsearch|quiesce|static_eval` in datagen returned nothing.

## The filters

Both compare the static evaluation against a deeper view of the same position.
All three quantities are side-to-move relative, so the subtractions are
directly comparable.

| Filter | Test | Default | Catches |
|---|---|---|---|
| M1 | `\|static − quiescence\| > margin` | 60 cp | Hanging material: a capture sequence swings the score |
| M2 | `\|static − search\| > margin` | 70 cp | Quiet tactics quiescence cannot see: forks, mating attacks |

M1 runs **before** the labelling search, because quiescence is far cheaper
than the 5000-node label search — it rejects most noisy positions for almost
nothing. M2 reuses the label search score that was already computed, so it
costs nothing extra.

M2 is not redundant with M1. Quiescence only resolves captures, so it is blind
to a quiet move that wins material next ply — the paper's example is a knight
fork on king and rook.

## Retuning

**The published margins were tuned on Xiangqi, not Western chess.** Xiangqi has
different material values and a different tactical density, so 60/70 are a
starting point, not a constant to transfer. They are therefore env-overridable
rather than baked in:

```sh
UNCHESSED_QUIET_MARGIN_QSEARCH=60 \
UNCHESSED_QUIET_MARGIN_SEARCH=70 \
  unchessed-datagen nnue out.bin 0 1 1000000 games.pgn
```

Setting either to `0` disables that filter, which is how to generate an
unfiltered baseline for an A/B comparison. Both binaries print their reject
counts per file so the selectivity is visible:

```
worker 0: done a.pgn: 12345 samples so far (900 games seen); \
  quiet-filter rejects: qsearch=4210 search=880 (margins 60/70cp)
```

Those counts are the input to retuning: if the qsearch rejection rate is
extreme, the margin is too tight for this evaluation's scale.

## Status — read this before trusting the numbers

- **The Rust does not compile in this sandbox.** There is no cargo or rustc
  available and the toolchain cannot be fetched (see the notes in the
  performance docs). The changes here were verified by symbol-level review
  only: every `Searcher` field was checked name-by-name against the struct
  declaration, every referenced symbol (`MATE`, `MAX_PLY`, `Move::NONE`,
  `Hce`, `TT`, `fen`, `AtomicBool`) confirmed in scope, and both edited files
  passed a literal-aware bracket-balance check. **That is not a substitute for
  a build.** Compile and run `cargo test --workspace --release` before using
  this.
- **No dataset has been regenerated and no network retrained**, so there is no
  measured quality delta yet. The claimed benefit is the paper's, not ours.
- A unit test (`static_and_quiescence_separates_quiet_from_hanging_positions`)
  asserts the new helper actually separates a quiet position from one with a
  hanging queen — but it has not been executed here, for the reason above.

## Next steps

1. Build and run the test suite.
2. Generate a filtered and an unfiltered dataset from the same PGN corpus,
   using the `=0` override for the baseline.
3. Compare rejection rates and retune the margins for this evaluation scale.
4. Retrain and gate the result through SPRT as usual — a dataset change is a
   playing-strength change and does not bypass the gate.
