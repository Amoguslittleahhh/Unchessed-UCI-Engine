# Hardware Portability and Neural Verification Design

## Hardware policy

The optimisation copy targets three deployment classes rather than one `target-cpu=native` binary:

| Tier | Intended CPUs | Build policy | Runtime policy |
|---|---|---|---|
| Portable | Legacy x86-64 Intel/AMD with SSE2 | No host-specific Rust flags | Scalar/portable code paths and conservative threads/hash |
| Modern x86-64 | AVX2/FMA/BMI2-capable Core/Ryzen/Xeon/EPYC | Optional separately built `-C target-cpu=x86-64-v3` artifact | Runtime feature detection for specialised kernels |
| Host-tuned | A known deployment CPU | Optional `-C target-cpu=native` artifact only | Benchmark AVX2/VNNI/other paths on that exact machine |

The default repository build is now portable: it no longer forces `target-cpu=native`. A host-specific binary must be deliberately built and separately labelled. This prevents an AVX2/VNNI build from being deployed accidentally on legacy systems and keeps consumer Intel, consumer AMD, hybrid Core Ultra, EPYC/Xeon, and virtualised CPUs in scope.

AVX-512 is not assumed. Core Ultra 9 285H systems should prefer AVX2/AVX-VNNI where measured; Ryzen 7 7730U should use AVX2 if available and otherwise fall back cleanly. Runtime feature detection belongs inside optional kernels, with a scalar oracle and parity tests. Quantised NNUE/VNNI is a separate model/runtime ABI project, not a compiler flag.

## Cache-aware transposition-table sizing

The default `Hash` is now derived from `/sys/devices/system/cpu/cpu0/cache/index3/size` when available. It targets half of the detected shared L3 and clamps to 4–128 MiB. If the operating system does not expose cache metadata, it uses a conservative 32 MiB default. An explicit UCI `Hash` option always wins.

This is deliberately conservative. The Ryzen 7 7730U commonly has 16 MiB L3, so automatic hash is approximately 8 MiB. The Core Ultra 9 285H configuration documented in this repository has 24 MiB L3, so automatic hash is approximately 12 MiB. The evaluator, code, thread stacks, and OS still need cache; placing an enormous TT in RAM can reduce nodes/second even when it increases theoretical hit capacity. Servers with large L3 can use a larger explicit hash, but this must be benchmarked under the real thread count.

A future `CachePolicy` option can expose `auto`, `small`, `medium`, and `large` profiles. The current implementation keeps the UCI surface compatible and treats a user-specified `Hash` as an explicit override. Benchmark each profile with fixed nodes and fixed time, reporting NPS, completed depth, hashfull, memory bandwidth, and p95 latency.

## Neural root-prior contract

The AlphaZero-inspired component is a **root prior**, not a replacement for alpha-beta. A provider receives a fully reconstructed position, the exact legal root move list, persona/mode context, and a disposable deadline. It returns:

```text
PriorReply {
  position_hash,
  legal_action_fingerprint,
  model_hash,
  schema_version,
  finite score per supplied legal move,
  optional normalized probabilities,
  latency
}
```

The firewall rejects wrong hashes, wrong action counts, duplicate/missing moves, illegal moves, NaN/infinite values, stale model/schema versions, and late responses. The validated scores become `RootHint` ordering signals for the first iterative-deepening pass only. Alpha-beta still searches every legal root move, retains TT/bound semantics, and chooses from completed scores/PVs. The prior cannot affect leaf evaluation, aspiration bounds, null move, LMR, pruning, contempt, troll eligibility, opponent telemetry, or the final persona selector.

Training is offline. A true AlphaZero-style target requires MCTS visit counts; alpha-beta MultiPV scores are distillation targets and must not be mislabeled as AlphaZero targets. Legal-action masking, draw-aware outcomes, position history, castling, en-passant, promotion, halfmove/repetition state, and codec version are part of the dataset manifest. UCI runtime never learns online from the opponent.

Promotion sequence:

1. Shadow mode: compute valid priors, record coverage/rank correlation/latency, never change ordering.
2. Root-order-only A/B: enable first-pass ordering with every legal move still authoritative.
3. Tactical and deadline gate: hostile priors, stale results, slow workers, NaN values, and queue saturation must all fail closed.
4. Persona matrix: prove identical Full/Match/Clinch/Punish/Defend safety rules, contempt, troll vetoes, and low-time behavior.
5. Fixed-hardware paired games and SPRT before any default change.

## Lc0 verification pipeline

Lc0 is integrated as a pinned external UCI provider, not linked into Rust search internals:

1. Pin the Lc0 release, network file and SHA-256, backend, driver/runtime, `Threads`, `MinibatchSize`, cache, and search options.
2. Spawn one owner process with piped stdin/stdout/stderr. Serialize commands, continuously drain both output streams, cap line sizes, and allow only one active search token.
3. Handshake with `uci -> uciok`, discover options, configure while idle, then `isready -> readyok`. Send `ucinewgame` and synchronize before independent games.
4. Send complete `position startpos moves ...` or a validated six-field FEN plus history. Do not send only a final board when history-conditioned models are being tested.
5. Request a bounded `go` budget. Parse `info` defensively; only `bestmove` completes the request. On cancellation, send `stop` and drain until the matching `bestmove`.
6. Validate the returned move against Unchessed legal moves. Convert Lc0's value/policy/visits into a provider-specific evidence record; never compare visits directly to Stockfish cp or Maia WDL.
7. Use Lc0 as an objective verification candidate only when the BudgetPlanner predicts completion before the hard deadline. Under Critical/Emergency tiers, skip cold starts, GPU waits, and second-engine calls.
8. For disagreement, do not average incompatible scores. Ask the existing safety gate whether a move is legal, mate-adjacent, materially losing, tablebase-safe, or unknown. Choose the safest verified candidate according to the active persona.
9. Run baseline vs Lc0 shadow reports for tactical-suite pass rate, mate finding, WDL calibration, p50/p95/p99 latency, GPU memory, NPS-equivalent throughput, and persona distribution drift.
10. Promote only with color-balanced, opening-stratified, fixed-version SPRT plus hard non-regression gates for legality, crashes, clock forfeits, troll safety, and persona behavior.

Lc0's normal sampling/noise mechanisms are disabled for deterministic verification. Maia is a separate human-policy provider; its Elo-conditioned WDL and move probabilities are never treated as objective evaluations.

## Validation commands

```bash
source "$HOME/.cargo/env"
cd unchessed-heavy-optimisation
bash scripts/build-and-test.sh
bash scripts/build-and-test.sh --release

# Optional host-specific build, never the portable default:
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --workspace --release
# Exact-host build, only for a pinned deployment machine:
RUSTFLAGS="-C target-cpu=native" cargo build --workspace --release
```

The portable build must pass the full unit/perft/UCI suite. Host-specific builds must additionally pass the same suite and a fixed-node benchmark on each target CPU.

## References

- [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- [Stockfish UCI documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html)
- [Lc0 technical overview](https://lczero.org/dev/overview/)
- [Lc0 PUCT explanation](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [Maia-3 repository](https://github.com/CSSLab/maia3)
- [RubiChess repository](https://github.com/Matthies/RubiChess)
