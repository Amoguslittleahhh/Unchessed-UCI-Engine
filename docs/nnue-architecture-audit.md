# Auditing our NNUE against the Stockfish reference

A section-by-section check of `unchessed-core/src/nnue.rs` against the
[official NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html):
the quantization scheme, the int16→int8 / int32→int8 conversions, the forward
pass, HalfKP/HalfKAv2_hm feature sets, and the architectures section.

Most of it confirms existing decisions. One real gap was closed.

## Verified: our feature set is byte-identical to Stockfish's

Our v3 scheme is HalfKAv2_hm — 32 king buckets with horizontal mirroring, 11
piece planes, 64 squares, `22528 = 32 × 11 × 64`. The doc explains why the
mirroring is sound:

> The idea behind this feature set is to, for each perspective, transform the
> board such that our king is on the e..h files […] Knowing that only half of
> the king squares will be used allows us to cut the number of input features
> in half.

`KING_BUCKETS` decides which of the 32 buckets every feature index lands in.
**It had no test.** One wrong entry silently routes a king square to the wrong
704-feature block: the net still loads, still evaluates, and still passes the
existing colour-mirror tests — it just quietly reads the wrong weights for
some king positions. Strength loss with no visible symptom.

Checked against Stockfish's actual `half_ka_v2_hm.h`, fetched from source
rather than reasoned about: **all 32 e–h entries match exactly.** Ours stores
`-1` for files a–d because the board is mirrored before lookup, where
Stockfish stores a full horizontally symmetric table; comparing the e–h half
is therefore the correct comparison, and the a–d half is pinned separately.

Now covered by three Rust tests (`king_buckets_match_stockfish_half_ka_v2_hm`,
`king_buckets_cover_only_the_mirrored_half`, `v3_feature_dimensions_are_consistent`)
and a compiler-free Python equivalent, `tools/test_king_buckets.py`, which
parses both tables straight out of the Rust source so the invariant holds even
when cargo is unavailable. The Rust test validates its own reference first, by
asserting Stockfish's table is horizontally symmetric before comparing
against it.

Verified the tests bite: corrupting a single entry fails two of them
independently (the Stockfish comparison and the bijection check).

## Confirmed: our f32 inference is a deliberate divergence, not an oversight

Stockfish's scheme is int16 accumulator → int8 activations → int32
accumulation in the linear layers, with ClippedReLU clamping to 0..127 instead
of 0..1. Ours is f32 throughout.

This is already the subject of two prior findings and this audit changes
neither:

- `docs/performance-ceiling-and-gpu-viability.md` measured the int16 upside
  (output layer 67.6 → 13.5 ns, accumulator 13.91 → 7.02 ns) and the cache
  argument (f32 FT is 23.1 MB against a 24 MB L3; int16 is 11.5 MB).
- `docs/fishtest-and-quantization-notes.md` found our weights peak at **2.06x
  the int8 range** with no weight clipping in the trainer, which is why the
  earlier post-hoc int8 attempt failed parity at 1.01e-2.

The doc's own framing supports treating this as retrain-only work. Its
quantization section is explicit that the scheme has to be designed up front
and the trainer made aware of it — the range constraint is a *training*
concern, not a conversion step. Nothing here suggests a post-hoc path exists.

## Reviewed and not adopted

**Full_Threats feature set.** Threat features model pairs where one piece
attacks another, as an add-on to HalfKAv2_hm with a separately accumulated
second accumulator. Real gains in Stockfish, but it is a feature-set change:
new indexing, new format, full retrain — the same bar as the pawn-pair commit
reviewed last round, and for the same reason not portable to a net we cannot
retrain here.

**Feature factorization.** Trains virtual "K"/"HalfRelativeKP" factors that
are coalesced into real weights before export. This is a *training-time*
technique with no inference cost, so it is genuinely applicable if this
project ever retrains — worth remembering, but nothing to implement now.

**Sparse-input linear layers with nnz index extraction.** A large inference
win for Stockfish, but it depends on int8 activations to find non-zero lanes
with `_mm256_cmpgt_epi8` + `movemask`. On f32 activations the trick does not
apply. Blocked behind the same quantization work.

## Status

No behaviour change. The only code change is three new tests plus their Python
mirror; `KING_BUCKETS` itself was already correct and is unmodified.
`UnarchitecturedHint` stays default-off and `runtime_safety_suite` stays
false.

**Not compiled** — no cargo or rustc in this sandbox. The Rust tests were
checked by symbol review (`KING_BUCKETS`, `N_BUCKETS`, `FT_IN_V3`,
`N_PIECE_SQ_V3` all resolve with the right types), by bracket balance, and by
executing the identical logic in Python against tables parsed out of the real
source. They have not been run by `cargo test`.
