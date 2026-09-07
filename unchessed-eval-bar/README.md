# Unchessed Homemade Eval-Bar Stack

This folder contains an independently authored Unchessed evaluator and bar projection. Stockfish/SFNNv16 is used only as a public architectural reference and an external black-box comparison target. No Stockfish source files, network weights, feature rows, constants, or implementation code are imported into this stack.

The current runtime label is `homemade-stack-v3-selected`. It combines an existing evaluator score with an original, bounded bitboard residual and an original rational WDL projection. The broader research stack contains eleven independently implemented signals:

| Breakthrough family | Original Unchessed signal |
|---|---|
| Center-pressure lattice | Occupancy in a four-square central lattice |
| Pawn-geometry analyzer | Doubled, isolated, and connected-file structure |
| King-shelter ring | Own pawn cover around the king |
| Activity-space field | Extended-center occupation |
| Coordination mesh | Own pawn and knight attack overlap |
| Passed-pawn race | Enemy-ahead occupancy test on adjacent files |
| Outpost detector | Central knight squares outside enemy pawn reach |
| Bishop-pair topology | Same-side two-bishop indicator |
| Rook-file pressure | Rooks on pawn-free files |
| Tempo polarity | Side-to-move sign, used only in the experimental residual |
| Phase-aware gain | Occupancy-derived phase multiplier |

The complete stack is intentionally not forced into production. A held-out ablation selected pawn geometry, king shelter, passed pawns, bishop-pair topology, and rook-file pressure because that subset generalized better than the full noisy combination on the direct reference corpus. The rejected signals remain available for future independent training rather than being silently discarded.

The runtime calibration uses Unchessed-owned coefficients learned from a held-out split of black-box UCI score observations: a score scale of `0.680292396` and a bias of `24.603188915` cp. These values are not Stockfish coefficients and are not used by the baseline or official engine.

The comparison harness is `tools/compare_sfnnv16_homemade.py`. It launches official Stockfish 19 and the Unchessed reviewer as separate processes, sends identical FEN positions through UCI, and records only returned scores. The 120-position corpus was extracted from real PGN games by `tools/extract_pgn_positions.py`; it contains no engine scores or weights.

The direct result on 120 legal real-game positions at Stockfish depth 10 was:

| Metric | Selected homemade stack |
|---|---:|
| Mean absolute error | 178.925 cp |
| Median absolute error | 103 cp |
| Maximum absolute error | 1204 cp |

This is a score-proxy comparison, not an Elo match and not evidence that the homemade evaluator is stronger than SFNNv16. Establishing parity or superiority requires a trained independent network, fixed test suites, and statistically significant engine matches.

The stable UCI connection remains:

```text
position startpos
evalbar homemade
```

The output label `source=homemade-stack-v3-selected` makes the provenance explicit. The link line carries the shared bitboard snapshot and Elo-detector telemetry without changing search or detector state.

A second command, `evalbar parity`, exposes an experimental compact linear model trained from black-box Stockfish score observations and original Unchessed features. It is deliberately not promoted: on the same 120-position depth-10 corpus it measured 262.792 cp MAE versus 178.925 cp for the selected handcrafted stack. The failed experiment is retained as a reproducible negative result and a warning against fitting a small corpus too aggressively.


## Nonlinear search experiment

The repository now exposes `setoption name HomemadeNonlinear value true`. This enables the original `NonlinearHce` wrapper through the search evaluator seam; it is separate from the read-only `evalbar homemade` and `evalbar parity` display modes. The model is a bounded residual over the existing Unchessed HCE, using material imbalance, pawn phase, and king geometry. It has no Stockfish/SFNNv16 source or weight dependency.

The reproducible Stockfish-19 calibration command is:

```text
python3 tools/tournament_stockfish.py \
  --candidate target/release/unchessed-reviewer \
  --stockfish /tmp/stockfish19-reference/source/src/stockfish \
  --out docs/tournament-v1.jsonl \
  --movetimes 30,100,300 --games-per-pair 2
```

The first 24-game tournament did not justify changing the default: default HCE and the nonlinear mode tied at 0.167 average game points per game, while the nonlinear mode measured 0.8622 of baseline NPS on the seven-position depth-eight overhead benchmark. The nonlinear mode therefore remains experimental. The primary hot-path advice is to replace repeated context extraction with incremental scalar state and a precomputed king-distance table before increasing model complexity.
