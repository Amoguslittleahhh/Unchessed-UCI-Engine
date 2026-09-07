# All-position parity progress

## Broader teacher corpus

A second 5,000-position manifest was generated from real repository PGNs without the quiet-position filter. The split was game-level and leakage-safe: 3,965 train, 493 validation, and 542 test positions across 265/34/33 games. Stockfish 19 labels were collected through a persistent UCI process at depth 8 and were used only as black-box targets.

## Original v4 retraining

An original Unchessed v4 network was trained on 4,995 labels after excluding five mate-like targets above 4,000 cp from static-score regression. The best deterministic validation MAE reported by the trainer was 107.0 cp.

Through the real engine EvalFile path, the model measured:

| Population | Held-out MAE |
|---|---:|
| All 5,000 positions | 155.049 cp |
| 4,995 non-mate positions (absolute target <= 2,000 cp) | **99.523 cp** |

The ordinary-position strict 100 cp gate is therefore crossed on this sampled held-out stratum. It does not establish all-position parity: five of 5,000 labels were mate-like, and one direct 120-position comparison contained a 29,887 cp mate outlier.

## Mate-aware fallback

The calibrated evalbar now includes an original bounded mate-in-one detector based on Unchessed legal move generation and check detection. It is opt-in through `evalbar calibrated`; the search evaluator and default modes remain unchanged. Focused evalbar tests pass, the full core suite remains green, and a tactical UCI smoke search returns a legal bounded result.

## Strict interpretation

Stable 100/100 is still not reached. The ordinary score gate has improved to 99.523 cp, but all-position score fidelity, deeper forced-mate recognition, nonlinear hot-path speed, and a powered match gate remain open. The correct next step is a mate-distance target/head and a shallow tactical proof path, not declaring success from the non-mate subset.
