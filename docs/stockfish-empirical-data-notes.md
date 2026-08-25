# What Stockfish's published measurements say for this engine

Notes from the [Useful Data](https://official-stockfish.github.io/docs/stockfish-wiki/Useful-data.html)
and [Regression Tests](https://official-stockfish.github.io/docs/stockfish-wiki/Regression-Tests.html)
pages, plus the chessprogramming.org entries on Stockfish, NNUE, Leela and
Syzygy bases.

Two outcomes: one small feature shipped, and one larger feature **argued
against on their own data**.

## Shipped: `info hashfull`

Stockfish measured the Elo cost of hash pressure directly (SF15.1, UHO book):

| Hash | hashfull (permille) | Elo |
|---:|---:|---:|
| 256 MB | 131 | 0.0 |
| 64 MB | 397 | −0.8 |
| 32 MB | 591 | **−12.1** |
| 16 MB | 766 | **−21.4** |
| 4 MB | 931 | **−52.4** |

Their conclusion: keep average hashfull below roughly 300 permille.

**Our engine never reported `hashfull`.** It is a standard UCI field, every
GUI displays it, and without it an operator has no way to tell whether their
`Hash` setting is costing them 0 Elo or 50. Our default is 128 MB with a
2048 MB maximum, which is reasonable, but "reasonable default" and
"observable" are different things — long analysis sessions and high thread
counts are exactly where the table saturates, and exactly where nobody could
see it.

Implemented as `TT::hashfull()`, plumbed through `InfoEvent`, printed between
`nps` and `time`. Sampled over the first 1000 slots rather than scanned in
full: UCI defines the field as approximate, and a full scan of a 2 GB table
during search would cost more than the information is worth.

The occupancy test is `data != 0`, which is **exact rather than heuristic**
here: `pack` stores `depth + 1`, so a live entry can never be all-zero, and
`new`/`clear` leave exactly zero. That invariant is now asserted by a test,
because a change to the packing would otherwise leave the engine reporting
`hashfull 0` forever — a failure that looks plausible rather than broken.

## Argued against: Syzygy tablebases

We have **no tablebase support at all** (`grep` for `syzygy|tablebase|tbprobe`
finds nothing). The obvious reading of the chessprogramming.org Syzygy page
is that this is a missing feature. Stockfish's own measurements say
otherwise.

Their consistent 6-man measurement across versions, TB in RAM, STC:

| Stockfish | Elo gain from 6-man Syzygy |
|---|---:|
| SF 8 | +15.8 |
| SF 11 | +15.8 |
| SF 12 (NNUE introduced) | **+7.2** |
| SF 15 | **+2.7** |

And on a fast M2 SSD with SF 17.1, 50,000 games:

> Elo: **−1.26 +/− 1.46**, nElo: −2.64 +/− 3.05, LOS 4.46%

The trend is the point: **tablebase value collapsed as NNUE evaluation
improved.** A neural evaluator already plays simple endings well, so the
tablebase's marginal contribution shrinks toward nothing — and off RAM it is
measurably *negative*, because probe latency costs more than the knowledge
gains.

For this engine that argument is stronger still. Syzygy is a large piece of
work: probe code, WDL/DTZ file format, root-move filtering, `SyzygyPath`
option, 50-move-rule interaction, plus a multi-gigabyte download users must
obtain separately. The measured ceiling on a far stronger engine is +2.7 Elo
with the files in RAM, and negative from SSD.

**Recommendation: do not implement Syzygy.** Recorded here so the question
does not get reopened as an obvious gap. If it is ever revisited, the case
would have to come from analysis-quality goals, not playing strength.

## Also noted, no action

- **MultiPV costs Elo at fixed time** — −97 at MultiPV 2, −157 at 3 (SF15.1,
  60+0.6). Our default is 1 and our calibration work used MultiPV only for
  offline teacher labelling, which is the correct use. Nothing to change,
  but worth knowing the search-time cost is that steep.
- **Threading efficiency**: at 64 threads, equivalent nodestime is ~200%,
  and efficiency improves with larger node budgets. Consistent with the
  earlier `default_threads()` fix; no new action.
- **Regression-test methodology**: Stockfish tracks strength across releases
  with fixed books and long time controls rather than only per-patch SPRT.
  Not applicable yet — this project has no release cadence — but it is the
  right model if one ever exists.

## Status

The only behaviour change is one additional field in the `info` line, which
is additive and cannot alter search. `UnarchitecturedHint` stays default-off
and `runtime_safety_suite` stays false.

**Not compiled** — no cargo or rustc in this sandbox. Verified by symbol
review (`store`'s signature, `Move::NONE`, `BOUND_EXACT`, and `Move` already
imported in `tt.rs`), bracket balance across all 21 tracked `.rs` files, and
by working the slot arithmetic through by hand: `TT::new(1)` allocates 65536
slots, so hashes `0..1000` occupy distinct slots and the permille arithmetic
gives exactly 0, 1000 and 250 for the three filled cases the tests assert.
The Rust tests have **not** been run by `cargo test`.
