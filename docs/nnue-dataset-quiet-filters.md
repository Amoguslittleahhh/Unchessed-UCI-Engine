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
unfiltered baseline for an A/B comparison.

### Measured on this engine (2026-08-28)

Full-file run, 1 thread, this 2-vCPU host:
`data/training/lichess-2022-10-05/elo-1700-2000.pgn` (4,465 games seen,
4,092 contributed, 121 s, ~358 samples/s).

| Margin (cp) | 10 | 20 | 30 | 40 | 50 | **60** | **70** | 80 | 100 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 rejects \|static−qsearch\| > m | 23541 | 21982 | 20561 | 19336 | 18137 | **17021** | 15917 | 14811 | 12478 | 9638 |
| M2 rejects \|static−search\| > m | 41202 | 31926 | 24311 | 18508 | 14186 | 10867 | **8543** | 6868 | 4671 | 2282 |

At the published 60/70 defaults, with the pipeline counts:

- M1: **17,021 rejected of 79,267** positions tested against quiescence
  (**21.5%**). M1 rejects are cheap — they happen before the labelling search.
- 62,246 survive M1 and get the 5000-node labelling search; 10,428 of those
  are dropped by the pre-existing post-label skips (depth fallback, mate /
  >2000cp score, tactical best move) before M2 ever sees them.
- M2: **8,543 rejected of 51,818** positions with a label score (**16.5%**).
  M2 rejects are expensive — each one already paid for the full label search.
- **43,275 accepted** (54.6% of M1 candidates).

Both reject curves are long-tailed (roughly hyperbolic out to at least 150
cp) — there is no elbow in the curve that argues for a different default, and
this engine's evaluation swings are large enough that even the loose end of
the curve is informative. **Decision: keep the published 60/70.** An unfiltered
A/B on the same file showed the filter costs ~3% throughput (308 vs 300
samples/s for the same 20,000-record cap).

### Base-seconds gate vs the committed corpus

The dataset also carries a fail-closed `NNUE_MIN_BASE_SECS=180` gate (games
must record a TimeControl with ≥180 s base time, to exclude time-pressure
play). **None of the committed corpora carry TimeControl headers**, so the
gate rejected 100% of candidates — 0 samples out of the whole file. The gate
now honours the `UNCHESSED_NNUE_MIN_BASE_SECS` env var (default 180; `0`
disables it). Runs against the committed corpus must set it to `0`; this is a
data-provenance property of the corpus, not a quality property of the games.
The numbers above were measured with `UNCHESSED_NNUE_MIN_BASE_SECS=0`.

## Status

- **Built and tested here**: cargo 1.88.0, `cargo test --workspace --release`
  green; the unit test
  `static_and_quiescence_separates_quiet_from_hanging_positions`
  (`unchessed-core/src/search.rs`) was executed and passes.
- **The retuning table above was measured on this host** (2 vCPU, 1 thread,
  ~358 samples/s). The full committed corpus (~95k games across
  `data/training`, `data/training-elo`, `data/selfplay`) is ~1 hour single-
  threaded here and a few minutes on the 180-vCPU cloud host — regenerating
  training data is no longer an infrastructure decision.
- **No dataset has been regenerated for training and no net retrained**, so
  there is no measured quality delta yet. The claimed benefit is the
  paper's, not ours.
- The 8 piece-count output buckets proposed in
  `docs/research-notes-moe-2507.11181.md` are now implemented in the runtime
  (file format version 4, `unchessed-core/src/nnue.rs`) and the trainer
  (`tools/train_nnue.py`), with a cross-checked trainer→runtime ABI
  (trainer-exported net loads and matches the Python manual forward exactly).
  Retraining on the filtered corpus with the 8-bucket head is the next gated
  step — owner's call, and it goes through SPRT.

## Next steps

1. ~~Build and run the test suite.~~ Done (2026-08-28, cargo 1.88.0, green).
2. Generate the filtered and unfiltered datasets from the full committed
   corpus with `UNCHESSED_NNUE_MIN_BASE_SECS=0` (no TimeControl headers).
3. ~~Compare rejection rates and retune the margins.~~ Done above — 60/70
   kept as the default.
4. Retrain (filtered corpus, 8-bucket head) and gate the result through SPRT
   as usual — a dataset change is a playing-strength change and does not
   bypass the gate.
