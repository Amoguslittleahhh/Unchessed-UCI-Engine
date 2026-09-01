# Unarchitectured v1 runtime: default-path GEMM tiling and scratch reuse

This commit does **not** flip `UnarchitecturedHint` (still default-off) and
does not change search. It only rearranges work already in
`aegis_v4_runtime.rs` so the sequential forward pass does less allocation
and less cache thrash. Numerics: same i16×i8 products and the same
attention math, written into the existing `attended` buffer instead of a
fresh `Vec` per head.

## What changed

1. **Token-outer quantized `linear_full`.** The old loop streamed all
   tokens for two output channels, then the next pair of channels — so
   the 4-token activation tile was reloaded `out_width/2` times. The new
   loop keeps those four i16 rows hot and streams weight pairs. Products
   and scales are unchanged.
2. **Attention writes through.** Sequential heads (the default
   `UNCHESSED_INFERENCE_THREADS=1` path) accumulate into
   `BlockScratch.attended` in `(token, width)` layout. No per-head
   allocation and no copy-back.
3. **Thread-local `BlockScratch`.** One ~0.9 MB workspace per thread,
   reused across forwards of the same Matryoshka width. First call still
   allocates.

## What this is not

- Not a measured host speedup in this sandbox (no `rustc` here). Re-run
  `cargo test -p unchessed-core --release benchmark_forward_pass -- --ignored --nocapture`
  on real hardware and treat the number as host-specific.
- Not a reason to turn the hint on. Four SPRT batches were never
  positive; the remaining Elo hole is structural (hint only sorts the
  first ID pass) as documented previously. This commit does not re-open
  that wiring.

Parity gates that must still hold after this change:
`start_position_matches_python_reference`,
`midgame_position_matches_python_reference`,
`position_to_input_matches_hand_built_start_position`.
