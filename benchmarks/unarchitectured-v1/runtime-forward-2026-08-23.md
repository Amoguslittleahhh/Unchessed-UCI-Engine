# Unarchitectured v1 runtime optimization — round 3

Measured on the sandbox's two-visible-CPU Intel Xeon 2.60GHz host with Rust
1.97.1. The real package SHA-256 is
`5fd9fc3fbf47bd2620c2e832e24c98525b59feeea791abf1c7ae32b9d311b16d`;
the measured runtime source SHA-256 is
`dc0babf878f2884ef8f7c366baf4a9bd52c99c27d1e3f3208b76146c88ccddf0`.

## Alternating comparison against merged round 2

A detached worktree at `main` commit `8991045` and the round-three working tree
were benchmarked in alternating order. Each entry below is the median of three
release runs of 200 full 8/256 forwards. This controls host drift better than
comparing a new minimum with a result from another session.

| Threads | Merged round 2 | Round 3 | Speedup |
|---:|---:|---:|---:|
| 1 | 15.831563 ms | 14.697528 ms | **1.0772x** |
| 2 | 14.158920 ms | 13.163678 ms | **1.0756x** |

Raw one-thread baseline runs were 15.831563, 15.847183, and 15.815639ms;
round-three runs were 15.137596, 14.573288, and 14.697528ms. Raw two-thread
baseline runs were 14.156111, 14.158920, and 14.206816ms; round-three runs were
13.239590, 13.163678, and 12.940551ms.

A separate two-thread exit ladder measured:

| Exit | Latency |
|---|---:|
| Shallow 2/128 | 2.570135 ms |
| Middle 4/192 | 5.613094 ms |
| Full 8/256 | 13.102574 ms |

These absolute numbers are specific to this shared host. They are not promised
to transfer to the reviewer's or deployment CPU.

## Retained changes

- reduce AVX2 integer and f32 accumulators in registers instead of storing lanes
  and folding scalars;
- retain each ordinary linear's int8 weight-row pair across all token blocks;
- resolve runtime-selected dot/AXPY kernels once per operation rather than in
  every inner call;
- replace per-forward formatted layer tensor names with static names; and
- run regret source/target projections through the retained-int8 backend.

The complete integer-versus-dequantized output gate remains `5e-4`. The maximum
observed component in this round is `9.4e-5`; Python parity and best-move gates
remain unchanged.

## Rejected experiments

- AVX-512 VNNI was exact but much slower (roughly 23.4/19.1ms at one/two
  threads in the trial), consistent with frequency throttling.
- A degree-six range-reduced exponential passed parity but regressed latency.
- Output-major QKV/FFN-up traversal regressed from strided output writes.
- Pairwise pooled-token summation did not materially tighten the narrow-exit
  Python difference, so the reviewed tolerance was not changed.

## Scope

The model remains completely unwired from search. No integrated NPS, clock,
tactical-safety, Elo, or SPRT claim is made. Any search use still requires the
existing deployment calibration, deadline accounting, mate/only-move, and
paired-game gates.
