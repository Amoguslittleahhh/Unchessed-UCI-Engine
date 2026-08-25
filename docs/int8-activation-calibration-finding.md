# Calibrated int8 activations: measured, and it doesn't work

`docs/unarchitectured-v1-runtime-optimization.md` lists calibrated int8
activations as the first item of "Remaining performance work", with the
honest caveat that a better calibration scheme "might close that gap, but
it's unproven". This settles it: **five schemes measured, none passes, and
the mixed-precision fallback doesn't survive contact with unseen positions
either.** The item should be closed as tried-and-rejected rather than left
open as a promising lead.

Tool: `tools/analyse_int8_activation_calibration.py`.
Artifact: `benchmarks/unarchitectured-v1/int8-activation-calibration.json`.
Tests: `tools/test_int8_activation_calibration.py` (17 tests, 19 subtests).

## Why this was worth measuring before implementing

The runtime already retains int8 *weights* and quantizes activations to
int16. Moving activations to int8 would roughly double the lanes per AVX2
instruction on the dominant matmuls — the largest arithmetic win left that
doesn't require retraining. The previous attempt (per-token symmetric)
missed the parity gate at `1.01e-2` versus the required `5e-3` and was
correctly reverted.

The important point is that **this question is decidable without writing a
single AVX2 kernel**. Quantization error is a property of the weights and
activations, not of instruction selection. Simulating the quantize/dequantize
round trip in float reproduces exactly what an integer kernel would see, so
the parity outcome can be known in advance. Writing the kernels first and
discovering the drift afterwards is the expensive way to get the same answer.

## Finding 1: no whole-model scheme passes, and it isn't close

Max absolute logit drift against the real exported checkpoint, on the two
positions the Rust parity tests freeze. Gate is `5e-3`.

| scheme | start position | midgame (1.e4 e5) |
|---|---:|---:|
| `per_token_symmetric` (the rejected baseline) | 2.62e-2 | 4.09e-2 |
| `per_tensor_static` | 4.60e-2 | 4.84e-2 |
| `per_channel_symmetric` | 2.52e-2 | 3.18e-2 |
| `per_group_symmetric` (group 32) | 2.41e-2 | 2.76e-2 |
| `percentile` (99.9) | 7.24e-2 | 6.54e-2 |

Every scheme fails by 5-14x. The best of them, per-group with 32-wide
groups, is still 4.8x over the gate — and per-group scaling is already the
expensive end of what a fast kernel can afford, since each group needs its
own scale applied during accumulation.

Two things worth noting rather than glossing over:

- **Percentile clipping made things worse, not better**, despite being the
  most promising idea going in. Clipping outliers helps when a few loud
  channels are crushing everything else; here it discards real signal. The
  unit tests still verify the mechanism works as intended on a synthetic
  outlier, so this is a genuine property of the weights, not a bug.
- **The best move survived every scheme on both fixtures.** That is not a
  pass. The gate exists because the port's numerical correctness is the only
  thing proving it matches the Python reference, and "the argmax happened to
  hold on two positions" is exactly the kind of weak evidence the gate is
  there to reject.

## Finding 2: the error is diffuse, which kills the obvious workaround

If the drift were concentrated in a few sensitive matmuls, the fix would be
easy: run those in int16 and everything else in int8. Measuring all 50
matmul sites individually says otherwise.

- Only **14 of 50** sites are individually harmless (drift `<= 5e-4`).
- The worst single site contributes `1.24e-2` on its own.
- The individual drifts sum to `~1.04e-1` — no single site dominates.

This is a diffuse accumulation through the residual stream, not a few bad
actors. Errors compound across 8 layers rather than staying local.

## Finding 3: the mixed split works on the fixtures and overfits them

Admitting sites to int8 cheapest-error-first, against the two frozen
fixtures, finds **28 of 50 sites (44.4% of MACs)** that jointly pass at
`4.2e-3`. That looks like a real result — roughly half the arithmetic in
int8 while holding parity.

It isn't. Re-measuring that same frozen assignment on 150 unseen corpus
positions:

**80 of 150 positions exceed the gate**, worst case `1.10e-2`.

The assignment was tuned on its own test set. Two positions cannot
characterise the activation ranges the model meets in real play.

Note also that intersecting the two per-fixture winners — the obvious
repair — does not work either: the 30-site overlap still fails the midgame
fixture at `5.74e-3`, because each set was selected against its own fixture.
The search has to optimise the worst case across fixtures jointly, which is
what the tool now does.

## Finding 4: making it generalise destroys the speed case

Adding real corpus positions to the calibration set and re-running the joint
search:

| calibration positions | int8 sites | MACs in int8 | holdout worst | over gate |
|---:|---:|---:|---:|---:|
| 0 (fixtures only) | 28 | 44.4% | 1.10e-2 | 80 / 150 |
| 24 | 20 | 17.3% | 7.47e-3 | 13 / 150 |
| 80 | 18 | 10.6% | 5.82e-3 | 2 / 150 |
| 160 | 18 | 14.0% | 6.56e-3 | 3 / 150 |

The trend is unambiguous and it is the whole finding: **every position added
to the calibration set shrinks the admissible int8 set, and it still never
reaches zero holdout failures.** Coverage collapses from 44% of MACs to
~11-14% while 2-3 positions per 150 remain over the gate.

Even taking the most favourable row at face value, ~11% of MACs in int8 buys
a small fraction of one forward pass — against per-group scale handling,
two accumulator paths, and a second set of kernels to maintain. That is not
a trade worth making, and it is not even available, because the gate still
fails.

The 80 → 160 row going slightly *worse* is expected: the greedy search
optimises the worst case over its own calibration set, and a different
calibration sample selects a different set. It confirms the search is at a
noise-dominated plateau, not converging on a safe assignment.

## Conclusion

Close "calibrated int8 activations" as measured and rejected. The blocker is
not the calibration scheme — five were tried, spanning per-tensor to
per-channel to per-group to percentile clipping — it is that this
checkpoint's activations carry more dynamic range than int8 resolves, and
the resulting error accumulates diffusely across 8 layers.

**This is a weights problem, and it has the same root cause as the finding
in `docs/fishtest-and-quantization-notes.md`:** the trainer applies
`clip_grad_norm_` but never clips weights, so nothing during training ever
pressured the activation distributions into a quantization-friendly range.
Stockfish's trainer clamps weights every step for exactly this reason.

That makes int8 activations a **retrain-gated** item, not a kernel-
engineering item — and it lands on the existing retrain backlog next to GAB
capacity, rating conditioning, and weight clipping, rather than sitting on
the performance list implying someone could pick it up and implement it.

## Honest limits

- Simulated in float, not executed as integer kernels. This measures
  quantization error faithfully; it does not measure achieved speedup, and
  the MAC fractions above are arithmetic counts, not benchmarks.
- The holdout is 150 corpus positions at `rating=2700, policy_kind=1`. Other
  personas were not swept.
- Not compiled or run in Rust — no Rust toolchain is available in this
  sandbox. No Rust code changed, so nothing here needs recompiling.
- The greedy search is greedy: it finds *a* good assignment, not provably the
  best one. A better assignment might exist at some coverage level. Given
  the holdout failures persist across every calibration size tried, this is
  unlikely to change the conclusion.
