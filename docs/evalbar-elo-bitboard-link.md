# Eval-bar, Elo detector, and bitboard connection

The stable connection point is the public Rust API in `unchessed_core::eval_bar`:

```rust
use unchessed_core::eval_bar::{EvalBar, EvalBarLink, EvalBarSource};

let mut bar = EvalBar::new();
let sample = bar.sample_from_score(&position, raw_stm_score, EvalBarSource::LoadedNnueProxy);
let detector = opponent_model.telemetry_snapshot();
let link = EvalBarLink::from_sample(
    &sample,
    detector.estimate_elo,
    detector.confidence_cp,
    detector.suspect,
    detector.suspect_reason.name(),
);
```

The source path is [`unchessed-core/src/eval_bar.rs`](../unchessed-core/src/eval_bar.rs). `EvalBarSample::bitboard` carries the position hash, total occupancy, White occupancy, Black occupancy, and the Stockfish-style material index. `EvalBarLink` carries those bitboard values together with the White-centric display score, bar fraction, Elo estimate, confidence interval, suspect flag, and suspect reason.

The existing UCI connection is available through:

```text
position startpos
evalbar
```

The engine emits two lines. The first contains the score, WDL, bar fraction, smoothing state, and evaluator source. The second is the machine-readable connection link:

```text
info string [Unchessed] evalbar-link hash <position_hash> occ <occupancy> material <material_index> elo <estimate> +/-<confidence_cp> suspect=<bool> reason=<reason>
```

This interface is deliberately read-only. It does not alter search, move selection, Elo observations, or NNUE accumulators. The `sample_with_state` method accepts an existing `EvalState`, allowing a caller that already owns an incremental NNUE accumulator to project the bar without rescanning the board or rebuilding the network state.

For performance-sensitive integration, call `sample_from_score` or `sample_with_state` at an existing evaluator boundary. Do not call the HCE convenience method from every search node. The current UCI command is on-demand and therefore cannot bottleneck normal search.
