# Stockfish 19 research findings (working notes)

## Sources reviewed

1. Official Stockfish 19 release announcement: https://stockfishchess.org/blog/2026/stockfish-19/
2. Official Stockfish source, `src/evaluate.cpp`: https://github.com/official-stockfish/Stockfish/blob/master/src/evaluate.cpp

## Evidence captured

The Stockfish 19 announcement dated 2026-09-05 reports the SFNNv16 NNUE architecture, removal of redundant threat features, introduction of pawn-pair features, retirement of the secondary network introduced in 16.1, quantization-aware training, and rescoring of hundreds of billions of positions with a strong Leela net. It also reports an advertised gain of up to 44 Elo against Stockfish 18, but this is a release-level strength claim rather than an evaluation-bar calibration claim.

The current official evaluator source exposes a transparent post-NNUE score pipeline. The network returns separate `psqt` and `positional` components. Their sum is used as `nnue`; complexity is `abs(psqt - positional)`. The code adjusts optimism by complexity, applies a complexity-based shrinkage to `nnue`, adds a material/optimism interaction, applies a rule-50 reduction, clamps the result to tablebase-safe bounds, and returns the side-to-move score. The source’s `trace()` path separately prints raw NNUE evaluation and final scaled evaluation, converting the side-to-move internal value to white-centric centipawns with `UCIEngine::to_cp`.

The implementation implication is that a homemade eval bar should not treat raw NNUE output as the final displayed score. It should reproduce the side convention, post-NNUE scaling, rule-50 adjustment, score clamping, and the calibrated conversion from centipawns to win/draw/loss or bar percentage. Exact numeric equivalence requires the matching Stockfish 19 network, source revision, and UCI conversion code; a lightweight independent bar should be documented as a proxy unless these are integrated.

## Additional evidence

The official NNUE documentation describes the accumulator as an incrementally maintained first linear layer: added feature columns are summed and removed feature columns subtracted, while king moves trigger a refresh because king-relative features change broadly. It documents HalfKP features as `(our_king_square, piece_square, piece_type, piece_color)`, with two side perspectives and a side-to-move-aware combination. The documented feature-index example has 64*64*5*2 base combinations, and the design is explicitly optimized for low-precision integer inference. This supports using incremental state and bounded integer arithmetic in a homemade evaluator, but also shows why a faithful Stockfish replica requires the exact network feature transformer and weights.

The official WDL model documentation states that Stockfish’s displayed “centipawn” is not a literal pawn value; it is calibrated so that +100 corresponds to a 50% self-play win probability under the relevant calibration conditions. The model uses a symmetric logistic win-rate function with material-dependent cubic parameters: `win_rate(x,mom)=1/(1+exp(-(x-p_a(mom))/p_b(mom)))`, `loss_rate(x,mom)=win_rate(-x,mom)`, and draw is the remainder. The repository notes that since SF17 the internal evaluation is scaled to displayed evaluation 1.0 for each material level before WDL conversion. Therefore a stable eval bar should use material-aware calibration rather than a fixed linear mapping from centipawns to percentage.

## Release tag and protocol evidence

The official repository exposes a dedicated `sf_19` tag at commit `edb0d9db6731067ec50ce619ff372b463bc4dd5d`, dated 2026-09-05. The tag contains `src/evaluate.cpp`, `src/score.cpp`, `src/uci.cpp`, and the NNUE implementation, so exact reverse engineering must pin source and network artifacts to this tag rather than silently using moving `master`.

The official UCI documentation states that `score cp` is positive for White and negative for Black, while the engine’s internal score is side-to-move oriented and must be normalized for GUI display. `score mate` is a separate forced-mate representation. `UCI_ShowWDL` reports approximate WDL statistics based on engine evaluation and game ply/material under fishtest LTC self-play conditions (60+0.6 seconds). A homemade bar should preserve mate precedence, explicitly convert side-to-move scores to a fixed board perspective, and label WDL percentages as calibration-dependent rather than universal probabilities.

## Comparative model evidence

Leela Chess Zero’s official technical explanation describes a different evaluator/search contract: a neural network emits both a position value and a move policy; PUCT/MCTS expands a tree, backs up evaluated values, and uses visit statistics to improve the root estimate. This is not directly interchangeable with Stockfish’s static NNUE score. The comparison supports a design choice for Unchessed: keep the homemade eval bar tied to a deterministic static score plus calibrated WDL, while treating search depth, MultiPV, and principal-variation stability as confidence metadata rather than mixing MCTS semantics into the bar.
