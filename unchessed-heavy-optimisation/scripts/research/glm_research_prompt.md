# Research request: Unchessed AI engine — open questions

## Context

Unchessed AI is a from-scratch Rust UCI chess engine with two binaries (an
opponent-adaptive engine and a full-strength analysis engine) sharing one
core crate. Current state:

- **Search**: iterative deepening alpha-beta, quiescence, TT, null-move,
  LMR, killers/history, MultiPV, time management.
- **Eval**: NNUE v4 — HalfKAv2_hm-style features (32-bucket horizontal king
  mirroring, own-king included as an active feature, feature factorization
  via a shared virtual embedding table coalesced at export time), 256-wide
  accumulator, plain SCReLU output head, 22528 inputs. SPRT-validated
  +26.1 ± 12.4 Elo over the prior version (v1, a plain 768-input net), 2251
  games, LOS 100%. A prior architecture (v3, using a Stockfish-SFNNv5-style
  concat output head) lost SPRT badly (-70.3 ± 22.1 Elo, LOS 0%) — the
  fix that produced v4 was replacing the feature set and output head, not
  a data or hyperparameter change.
- **Human policy net**: Maia-style, one net per rating bucket
  (<1300, 1300-1599, 1600-1899, 1900+), trained on 19.9M positions from
  781k human Lichess games, castling-rights + en-passant aware inputs.
  Top-1 accuracy predicting real human moves ranges 29-34% overall; notably
  en-passant recognition accuracy is only 18.8% in the <1300 bucket vs
  67.4% at 1900+.
- **Adapter persona system**: live Bayesian opponent-Elo estimate from
  centipawn loss, converging in ~8-12 moves; four personas (MATCH/PUNISH/
  CLINCH/DEFEND) with hysteresis-gated transitions; engine-tell detection
  from move-timing + quality; opening book with a troll tier gated by the
  live Elo estimate.
- **Reviewer tool** (PGN move classification, accuracy %): not yet built,
  listed on the roadmap only.

## What I want researched

For each numbered item below, I want: (1) how strong engines/projects in
the public literature or open-source (Stockfish, Leela/lc0, Maia,
nnue-pytorch, etc.) have approached it, (2) whether it's a well-established
win or still an open/contested question, (3) a concrete recommendation for
whether it's worth prioritizing for a small (~200K-23M param) NNUE-based
engine at this project's scale, and (4) any pitfalls specific to combining
it with the features already in place above (e.g. interactions with the
king-mirroring bucket scheme, or with the persona/policy-net system).

### NNUE / eval architecture
1. Ablating v4's three simultaneous changes (king mirroring, own-king
   feature, factorized training) to find which one(s) actually drove the
   Elo gain.
2. King-bucket count sensitivity (32 vs other counts) for a net this size.
3. Accumulator width tradeoffs (256 vs 512/1024) at ~20-25M total params.
4. Dual-net (small+big) setups — is this worth it before quantization work?
5. Quantization-aware training (int8/int16) — practical recipe for a
   from-scratch (non-Stockfish-codebase) engine.
6. Incremental accumulator updates on move make/unmake.
7. WDL loss exponent sensitivity (currently 2.5, copied from Stockfish's
   nnue-pytorch recipe without re-tuning for this dataset).
8. Feature sets besides HalfKAv2_hm worth considering at this scale
   (HalfKP, non-mirrored HalfKA, piece-square-relative).
9. Output head alternatives beyond plain SCReLU (concat/SFNNv5 already
   failed here — what else is known to work well at small scale?).

### Training data / labeling
10. Using Syzygy tablebases to correct/replace endgame labels.
11. Curriculum learning ordering (easy/quiet positions first) for NNUE.
12. Mining adversarial positions (largest eval disagreement between
    successive NNUE versions) as an oversampled training subset.
13. Label noise from a fixed-depth HCE search labeler — how much does
    label quality typically matter vs quantity at this data scale
    (~100M positions)?

### Search
14. Static exchange evaluation for capture ordering — is it already
    implicitly covered by history/killers at this depth range, or a
    clear independent win?
15. Correction history / continuation history — expected Elo impact at
    a from-scratch engine's current maturity level (has null-move, LMR,
    killers/history already) before adding these.
16. NNUE-eval-volatility-driven time allocation — known implementations
    and whether it's proven or still experimental.

### Testing discipline
17. The concrete gap that let v3's -70 Elo regression run for ~756 games
    before SPRT caught it: what's the standard practice (in Stockfish's
    fishtest or similar) for a cheap pre-SPRT smoke test that would catch
    an architecture regression this severe in far fewer games?

### Policy net / persona
18. Adding time-pressure (clock fraction remaining) as an input feature
    to a Maia-style policy net — has this been tried, what was the effect?
19. Calibrating "human move preference" against real per-rating blunder
    *rates*, not just move *frequency* — is there prior art on making a
    policy net's practical playing strength match a target Elo rather
    than just its move-prediction accuracy?

## Output format

For each item, 3-6 sentences. Flag clearly if something is speculative vs.
backed by a known paper/project. No need for exhaustive citations — named
sources (e.g. "Stockfish's fishtest wiki", "the nnue-pytorch repo docs",
"the Maia chess paper") are enough.
