# Unchessed Eval Bar Experiment

This folder contains the experimental harness for the engine-integrated evaluation bar in `unchessed-core/src/eval_bar.rs`. The implementation is intentionally **Stockfish-19-inspired but not an exact Stockfish clone**: it reuses Unchessed’s current static HCE score, converts it to a White-centric score, applies a presentation-only rule-50 dampener, and maps it through the pinned Stockfish 19 material-dependent WDL calibration.

The core engine exposes the same projection through the diagnostic UCI command:

```text
position startpos
evalbar
```

The output includes the White-centric display score, WDL per mille tuple, expected score, smoothed score, and `provenance=proxy`. The smoothing layer is bounded and presentation-only; it never feeds back into search or move selection.

The standalone harness accepts either no argument, `startpos`, or a FEN string:

```bash
cargo run -p unchessed-eval-bar -- startpos
cargo run -p unchessed-eval-bar -- '4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1'
```

Exact Stockfish 19 parity would require the SFNNv16 network, its feature transformer, quantization constants, and matching source/build configuration. This experiment therefore reports its provenance rather than presenting a proxy as an official Stockfish score.
