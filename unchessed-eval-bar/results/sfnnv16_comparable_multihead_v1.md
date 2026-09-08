# SFNNv16-Comparable Multi-Head Research Result

## Scope

This experiment implements an original, opt-in tactical and terminal evaluator that follows public architectural principles associated with modern sparse neural chess evaluators: a reusable existing score, sparse board-derived signals, bounded integer-friendly inference, confidence gating, and separate score, terminal, and mate-distance outputs. It does not copy Stockfish source, network weights, feature rows, constants, or implementation fragments.

The implementation is `unchessed-core/src/tactical_eval.rs`. The `evalbar tactical` UCI command exposes it for presentation and research diagnostics only. The default search evaluator is unchanged.

## Baseline and post-change validation

| Gate | Frozen baseline | Post-change result | Status |
|---|---:|---:|---|
| Workspace library/binary tests | 152 passed, 6 ignored | 155 passed, 6 ignored | Passed |
| Focused tactical tests | Not applicable | 3 passed | Passed |
| Release build | Existing baseline | `unchessed-eval-bar` release build passed | Passed |
| Terminal classification | Existing bounded probe | Checkmate and stalemate separated by unit test and UCI smoke | Passed |
| Search overhead <=5% | Approximately 0.8896 nonlinear/HCE ratio in handoff | Not measured for promotion | Open |
| All-position MAE <=100 cp | 155.049 cp prior broader-corpus result | No new corpus/retrain in this iteration | Open |
| Powered fixed-control match | Not established | Not run | Open |
| Stable 100/100 parity | Not achieved | Not achieved and not claimed | Failed/open |

The baseline environment and test log are preserved under `results/baseline/`. The baseline commit is `918fccc69097e1a1f6873540ed83a14ef24425d4`.

## Runtime smoke result

On the legal checkmate position `7k/6Q1/6K1/8/8/8/8/8 b - - 0 1`, the UCI diagnostic returned `legal=0`, `terminal=1`, `mate_distance=0`, and `terminal_prob=1000`. The output included `provenance=original_clean_room_research_only`. This verifies the separate terminal head without enabling it in search.

## Interpretation

The branch now contains a stable, test-covered architectural extension comparable in *design principles* to the supplied SFNNv16 diagrams, but it is not numerically equivalent to SFNNv16. Exact parity requires a provenance-matched corpus, independently trained weights, untouched tactical and terminal test strata, a speed benchmark, and a powered match. The implementation must remain opt-in until those gates pass.

## Reproducibility commands

```text
cargo test --workspace --lib --bins
cargo test -p unchessed-core tactical_eval --lib
cargo build -p unchessed-adapter
printf 'uci\nisready\nposition fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1\nevalbar tactical\nquit\n' | target/debug/unchessed-adapter
```
