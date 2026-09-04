# Cross-Engine Integration and Strength Plan

## Objective and limits

The goal is **Stockfish-class potential with the unique Unchessed behaviors active**, not a claim that a neural policy or another engine's centipawn score can be pasted into this engine. The existing alpha-beta search remains authoritative. AlphaZero and Leela Chess Zero ideas contribute policy/value learning and prior-guided ordering; Maia contributes Elo-conditioned human move distributions; Stockfish and RubiChess contribute CPU-first NNUE/search engineering. Every imported component is optional, versioned, and fail-closed.

AlphaZero's public record is limited. The published work establishes tabula-rasa self-play reinforcement learning, a policy/value network, MCTS/PUCT, legal-action masking, and visit-count targets. It does **not** establish DeepMind's production code, weights, replay system, distributed implementation, or undisclosed tuning. Therefore this project must not market a reconstruction as the original AlphaZero.

## Proposed linking system

Introduce a typed provider boundary rather than linking engines directly into search internals:

```text
UCI position firewall
  -> BudgetPlanner (Normal / Fast / Critical / Emergency)
  -> PersonaRouter (Full / Match / Clinch / Punish / Defend)
  -> Candidate providers
       ObjectiveAnalyzer: pinned Stockfish or RubiChess UCI child
       NeuralSearch: pinned Lc0 UCI child and network
       HumanPolicy: pinned Maia-3 UCI child/checkpoint
       RootPrior: optional AlphaZero-inspired policy provider
  -> TrollGuard + objective verification
  -> existing alpha-beta authority / adapt::select_move
  -> legal MoveDecision
```

The core interface should expose `capabilities()`, `configure()`, `new_game()`, `analyze(position, budget)`, `stop_and_drain()`, `health()`, and `shutdown()`. Each child process gets one owner, serialized commands, continuously drained stdout/stderr, bounded line buffers, a single active-search token, and a state machine from `Spawned` through `Ready`, `Searching`, `Draining`, and `Failed`. A stopped search is not reusable until its matching `bestmove` has been drained.

A `PositionRequest` must carry the complete move history, six-field FEN state, legal move list, side, Elo context, mode, and deadline. A `Candidate` must carry provider/version/model hash, move, native score type, latency, and confidence. Never average Stockfish cp, Lc0 visits/value, and Maia WDL as if they were the same unit. Convert only through a calibrated, explicitly versioned verifier.

## Safe transfer of modern techniques

| Source | Transferable capability | Safe boundary in Unchessed |
|---|---|---|
| Stockfish | Clustered/aged TT, capture/continuation history, singular extensions, modern NNUE quantisation, PVS/selective search | Introduce one feature at a time; fixed-depth equivalence first, then SPRT. Quantised NNUE requires a new ABI and golden parity tests. |
| Lc0 | Batched neural inference, policy/value/moves-left heads, PUCT-inspired priors, accelerator backends | Root ordering only; all legal moves still searched by alpha-beta. No neural value in bounds, pruning, contempt, or final selection. |
| Maia / Maia-3 | Elo-conditioned human policy, historical context, legal masking, temperature/top-p controls | Human-policy candidate source for Match/Clinch experiments; Maia score is not objective cp. Keep tactical verifier and persona selector authoritative. |
| RubiChess | Portable NNUE builds, PGO/LTO, Syzygy/Polyglot integration, defensive UCI lifecycle | External UCI adapter or selectively ported, license-reviewed algorithms. Pin source, binary, net, and CPU target together. |
| AlphaZero | Self-play, policy/value training, MCTS visit targets, draw-aware outcomes | Offline `az-selfplay-lab` only. Runtime gets a validated root prior, never a replacement search. |
| NNUE research | Better data curation, quiet-position filtering, group-disjoint validation, deeper labels | New dataset manifest and model version; never promote on validation MAE alone. |

## Stability fixes already applied in this copy

The isolated copy now filters opening-book entries against `go searchmoves`, gates auxiliary opponent/troll probes for strict fixed-node, shallow-depth, and sub-second requests, avoids forced adaptive MultiPV expansion once a known engine is detected for FULL play, corrects batched observation timestamps, and binds Aegis hint identity to en-passant and halfmove inputs. These fixes preserve the five personas while removing protocol and stale-cache failure modes.

## Promotion sequence

1. Freeze golden behavior for all five personas, opponent detection, contempt, troll vetoes, low-time gates, and legal output.
2. Add the UCI provider/session substrate and pinned Stockfish shadow analyzer.
3. Add a position firewall and `MoveDecision` schema; reject malformed or illegal external results.
4. Add Maia-3 with complete UCI history and `SelfElo`/`OppoElo`, initially shadow-only and temperature zero.
5. Add Lc0 only when a warmed accelerator and fixed network meet the latency budget.
6. Add an AlphaZero-inspired root prior behind `UseAZPrior=false` by default; validate position hash, legal-action fingerprint, model/schema hash, finite scores, and exact supplied-move coverage.
7. Add Stockfish-style TT/move-ordering/NNUE changes individually, never as a bundle.
8. Run fixed-hardware, fixed-opening, color-balanced SPRTs and stratified human-policy tests. Canary and rollback before default activation.

## Hard gates

Legality and UCI protocol gates are always active. Validate FEN/history, returned moves, `uciok`, `readyok`, `stop -> bestmove`, EOF, stale output, and process restart. Troll safety remains a veto layer. Low-time gating happens before spawning work and accounts for queue, inference, IPC, stop/drain, and move overhead. No cold model download, queue wait, unbounded MultiPV, or second engine may begin if it cannot fit the hard deadline.

Persona gates must show no change to the five stable IDs, mode-specific contempt, opponent-Elo transitions, candidate safety limits, or low-time behavior. AlphaZero-style priors may not change telemetry, target Elo, draw scoring, troll eligibility, or the final choice directly. A hostile prior must still produce the same completed alpha-beta best move as the baseline when root scores are complete.

Strength promotion requires tactical/perft correctness, no increased clock forfeits or crashes, no persona safety regression, and a pre-registered SPRT. A reasonable initial objective-strength test is H0 <= 0 Elo versus H1 >= +5 Elo, alpha=beta=0.05, one change per run. Human-policy changes need move-match/log-likelihood and verified-blunder non-inferiority by rating/time bucket, not engine Elo. Report p50/p95/p99 latency, NPS, completed depth, fallback rate, and confidence intervals.

## Sources

- [Silver et al., AlphaZero, arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- [Official Stockfish repository](https://github.com/official-stockfish/Stockfish)
- [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- [Stockfish UCI documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html)
- [Lc0 developer overview](https://lczero.org/dev/overview/)
- [Lc0 technical PUCT explanation](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/)
- [Maia original paper, arXiv:2006.01855](https://arxiv.org/abs/2006.01855)
- [Maia-2 paper, arXiv:2409.20553](https://arxiv.org/abs/2409.20553)
- [Maia-3 / Chessformer preprint, arXiv:2605.19091](https://arxiv.org/abs/2605.19091)
- [Official Maia-3 repository](https://github.com/CSSLab/maia3)
- [Official RubiChess repository](https://github.com/Matthies/RubiChess)
- [Study of Proper NNUE Dataset, arXiv:2412.17948](https://arxiv.org/abs/2412.17948)
- [ChessBench, arXiv:2402.04494](https://arxiv.org/abs/2402.04494)
- [UCI protocol specification](https://backscattering.de/chess/uci/)
