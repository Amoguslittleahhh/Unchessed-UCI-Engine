# Implementation status for the empirical audit

This document maps the supplied IEEE research audit to the implementation in
this repository. Claims from the paper are not silently promoted beyond their
evidence: fixed-depth node reductions remain proxies, and experimental search
features remain default-off pending game-match gates.

## Implemented

### NNUE inference

- Verified v4 dimensions: 22,528 inputs, accumulator 256, 5,767,937 parameters,
  exact f32 file size 23,071,768 bytes.
- Added a standalone finite-value/header/size inspector:
  `python tools/inspect_nnue.py unchessed-nnue.bin`.
- Quantize the feature transformer and bias at load time to int16 with scale
  511.
- Reject a network when a conservative 32-active-feature bound could overflow
  an int16 accumulator.
- Enumerate active features into a 32-entry stack array and accumulate into a
  stack-resident `[i16; 256]`; the previous two per-eval heap allocations are
  gone.
- Select AVX-512BW, AVX2, or scalar accumulation once per loaded network.
- Keep the f32 output reduction scalar, avoiding reassociation beyond the
  intentional quantization drift.
- Test selected SIMD against scalar accumulation exactly.

A local depth-10 comparison against commit `fde7fce` over the committed
12-position harness measured a **1.537x geometric-mean NPS gain** and 11/12
best-move agreement. This is a single-machine measurement, not an Elo claim.
Incremental parent/child accumulator updates are now implemented with a ply-indexed evaluator-state stack threaded
through root search, negamax, ProbCut, null move, and quiescence. Quiet moves
update only changed feature rows; a perspective is fully refreshed when its own
king changes HalfKA bucket/orientation. En-passant, promotion, captures, and
castling are covered by exact incremental-vs-refresh tests over special lines
and a full depth-3 move tree.

An isolated same-tree depth-10 comparison against an otherwise identical build
with incremental updates disabled measured a **1.215x geometric-mean NPS
speedup**, exact node/score identity, and 12/12 best-move agreement. This is
below the paper's projection and is reported as measured rather than adjusted.

### Search audit features

Implemented behind default-off UCI checks:

- `IIR`: depth-1 internal iterative reduction above depth 5 without a TT move;
- `HistGravity`: bounded gravity update plus malus to previously searched
  quiets;
- `CounterMoves`: side/previous-from/previous-to reply table below killers;
- `Razoring`: quadratic shallow fail-low verification through quiescence;
- `LMP`: non-PV shallow late-quiet pruning with capture/promotion/check/mate
  guards and schedule `3 + depth^2`.

All five complete legal searches under regression tests. They are default-off.
A local depth-10 run of the committed 12-position harness reproduced LMP's tree
effect at a **0.651 geometric-mean node ratio** with 11/12 best-move agreement.
That is a pruning diagnostic, not a strength result. The paper's corrected
experiment supports an SPRT for LMP, not automatic shipping;
IIR/history/countermove/razoring likewise require match evidence.

### Repetition correctness

Real game history and search-local cycles are now distinguished by a
`root_path_len` boundary. Search-local recurrence can still cut immediately;
pre-root history requires two prior occurrences, so the current node is the
third. Scanning is bounded by the halfmove clock and also applies in
quiescence. A regression test proves one prior forced-child occurrence remains
losing while two prior occurrences score a draw.

### Persona calibration and timing

- Retain 32 informative per-move strength samples and use their 20th percentile
  for the point estimate, while preserving the 100–3650 one-Elo posterior for
  uncertainty.
- Use the 75th-percentile recent ceiling for engine-strength evidence.
- Replace a permanent `estimate + 60` target with a zero-mean,
  confidence-scaled deterministic oscillation.
- Replace the often-unsatisfiable blunder band with a lognormal intended-error
  draw (sigma 1.1), mean `300 * exp(-Elo/900)`, and a Gaussian kernel over
  common-depth root candidate losses.
- Targets at/above the actual engine ceiling (2600) select the best move rather
  than retaining a 12cp artificial handicap. The opponent detector still spans
  100–3650, but the fixed-strength UCI control honestly advertises 100–2600.
- Remove mixed-depth MATCH side probes; root alternatives come from one shared
  target-dependent MultiPV search.
- Replace the inverted instant-move detector with lag-1 autocorrelation of log
  clock fraction. Timing may lower the evidence threshold for an opponent
  already playing at the measurement ceiling, but can never classify weak play
  by itself.
- Add tests for stable weak regular timing, strong regular timing, and strong
  irregular/premove timing.

### Book and anti-fingerprinting

- Effective historical book depth scales from 6 plies for sub-800 estimates to
  40 for 2400+/engines, bounded by `BookDepth`.
- Default `BookDepth` is 40 so the full named corpus is reachable at strong
  levels.
- CLINCH suppresses troll-tier book choices even if `Troll=On`.
- Existing identity/type safety gates and the 3,810-line, 500-ECO CC0 corpus
  remain in force.

### Incremental evaluation and production correctness

- Added ply-indexed `EvalState` plumbing through root search, negamax,
  quiescence, ProbCut, and null move. HCE remains stateless; NNUE updates
  changed feature rows and refreshes only the perspective whose own king
  changes a HalfKA bucket/orientation.
- Isolated same-tree measurement: 1.215x geometric-mean NPS, exact node/score
  identity, 12/12 best-move agreement at depth 10.
- Strict FEN validation now rejects malformed ranks, duplicate rights,
  impossible castling metadata, invalid en-passant state, adjacent/duplicate
  kings, back-rank pawns, and invalid counters. Castling generation also
  independently verifies the home king and rook.
- Canonical internal EP hashing omits uncapturable targets, while FEN
  serialization retains the supplied legal target.
- Checkmate takes precedence over rule-fifty/repetition draws; search TT keys
  include halfmove context; null moves no longer advance the game clock.
- `go nodes` checks exact global nodes shared by Lazy SMP helpers;
  `searchmoves`, `mate`, `ponder`, and `ponderhit` are parsed/handled;
  `go infinite` never returns merely because mate was found.
- Model/book preprocessing time is charged against the main move clock.
- Added adversarial `tools/uci_edge_smoke.py` and a checksum-producing release
  packager with optional hard requirement for the policy sidecar.

### Measurement and infrastructure

- Added `tools/bench_research.py`, which refuses the adaptive binary, pins
  MultiPV=1/Threads=1/Hash=128, reads through `bestmove`, and compares NPS when
  trees differ.
- Added CI for Rust check/tests, deep perft, Python/data checks, NNUE inspection,
  the fixed-depth reviewer harness, and HCE/NNUE UCI/persona smoke tests.
- Corrected README claims about v1 vs shipped v4 and about the absent policy
  sidecar.

## Deliberately not claimed or not yet implemented

- **Incremental NNUE Elo gain:** implementation and exact-score tests are done,
  but the measured speedup still requires an SPRT before an Elo claim.
- **LMP Elo gain:** requires SPRT; fixed-depth node savings do not establish
  strength.
- **Trained policy network:** inference/training exist, but no
  `unchessed-maia.bin` is present, so the engine still falls back to heuristic
  priors.
- **Correction and continuation history and singular extensions:** supported
  by the audit as future candidates, not implemented by its measured
  intervention. TT generation aging and UCI `hashfull` are now implemented.
- **General human/engine classifier from timing:** explicitly rejected by the
  paper's leave-one-account-out result. An independent public-data replication
  now also fails the configured standalone gates: matched account AUC 0.413
  (account-bootstrap 95% CI 0.260–0.575), with only 20 matched BOT accounts.
  Timing remains only a modulator; see `docs/timing-classifier-validation.md`.
- **One-Elo accuracy:** the posterior has one-Elo granularity, but credible
  intervals—not bucket width—represent statistical precision.
- **Independent SPRT replication:** no Elo claim in the paper was replicated in
  this implementation pass.
