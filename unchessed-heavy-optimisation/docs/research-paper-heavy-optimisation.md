# Hardware-Portable, Persona-Preserving Optimisation of the Unchessed UCI Engine

**Author:** Manus AI  
**Project:** `unchessed-heavy-optimisation`  
**Repository branch:** `manus/research-facilities`  
**Date:** 4 September 2026

## Abstract

This paper reports the design, implementation, and validation of an isolated optimisation sub-project for the Unchessed UCI chess engine. The work has two constraints. First, the engine must preserve its distinctive adaptive behaviors: Full, Match, Clinch, Punish, and Defend personas; live opponent modelling; risk-tiered troll opening behavior; mode-specific contempt; and low-time gating. Second, the implementation should have a credible path toward strong-engine performance across legacy x86-64 systems, consumer Intel and AMD processors, hybrid laptop CPUs, and server-class processors.

The implemented work focuses on correctness-preserving infrastructure rather than unvalidated search-tree changes. It fixes opening-book interaction with `go searchmoves`, prevents auxiliary persona probes from consuming strict fixed budgets, recovers depth in known-engine Full mode, corrects batched opponent-model timestamps, and binds neural hint-cache identity to en-passant and halfmove inputs. It also replaces an unconditional host-specific compiler target with portable defaults and introduces cache-aware transposition-table sizing based on detected shared L3 capacity.

The project installed Rust 1.98.1 and completed the existing validation suite. The result was 124 passing workspace tests, six ignored tests, successful deep perft, successful UCI smoke tests, and successful portable and `x86-64-v3` release builds. A comparative one-thread benchmark across four positions and five transposition-table sizes measured a modest `x86-64-v3` advantage ranging from 0.52% to 6.68% in mean nodes per second on the current virtual machine. Memory usage scaled with the configured transposition table, while the two build variants had nearly identical resident-memory footprints.

The paper also specifies, but does not claim to have fully implemented, a neural root-prior firewall and an Lc0 verification pipeline. The proposed design keeps alpha-beta search authoritative. Neural priors may reorder the first root iteration, but they cannot remove legal moves, alter pruning or contempt, bypass troll safety, or directly select the final persona move. External Lc0 integration is treated as a pinned, supervised UCI provider with strict deadline, protocol, legal-move, and model-version validation.

## 1. Research question and scope

The central research question is:

> How can an adaptive, humanised UCI engine gain portable performance and safely incorporate modern neural-engine ideas without sacrificing its unique persona contract?

The project does not claim that compiler flags alone can make a small engine equal to Stockfish. Contemporary engine strength depends on search selectivity, transposition-table design, move ordering, NNUE architecture and training data, time management, and extensive statistical testing. Stockfish uses NNUE as a static evaluator inside a selective alpha-beta/PVS search rather than as a standalone move selector [1] [2]. Leela Chess Zero uses neural evaluation and policy priors inside MCTS-family search [3] [4]. Maia optimises human-move prediction rather than objective best play [5] [6]. These objectives are related but not interchangeable.

The sub-project therefore separates three classes of work:

| Class | Examples | Release policy |
|---|---|---|
| Correctness-preserving infrastructure | UCI restriction fixes, stale-cache prevention, strict budget gates | Implement after focused tests |
| Portable performance | LTO, release code generation, cache-aware TT defaults, separate portable/v3 builds | Benchmark on target hardware |
| Tree-changing strength research | Clustered TT, capture history, singular extensions, quantised NNUE, neural pruning | Candidate branch, fixed-depth regression, then SPRT |

The working copy is nested at `unchessed-heavy-optimisation/`. The original `main` branch was not edited.

## 2. Protected Unchessed behavior

The engine's distinctive functions are treated as an explicit compatibility contract rather than incidental implementation details.

| Function | Required behavior |
|---|---|
| Full | Select the strongest completed alpha-beta line. |
| Match | Condition play toward a target Elo and, below 2200, include all legal moves in the shallow candidate pool so characteristic human errors are representable. |
| Clinch | Probe opponent reply gaps and favor narrow-path traps while retaining both queens in suitable drawish positions. |
| Punish | Force discovered mates and otherwise prefer forcing captures or checks when materially ahead. |
| Defend | Select maximum resistance. |
| Opponent modelling | Estimate opponent Elo from choice-weighted centipawn loss and time-usage engine signals. |
| Troll book | Select risk-tiered lines and abort a refuted troll continuation when the shallow safety check falls below the configured threshold. |
| Mode contempt | Score draws according to the active persona instead of one flat constant. |
| Low-time gating | Skip luxury observation, Clinch probes, and troll refutation work when the clock or current move budget is tight. |

The central design rule is that a neural or external-engine component may provide evidence, but it may not silently bypass this contract.

## 3. Implemented changes

### 3.1 UCI `searchmoves` and opening book

The UCI protocol permits a GUI to restrict a search to a list of root moves. The earlier book path could return a book move before the later root-search restriction was applied. The isolated copy now filters book entries against the requested UCI move strings before selecting a book move. If no permitted book entry exists, the engine falls through to restricted alpha-beta search.

This is a protocol correctness fix. It does not weaken the book or alter unrestricted games.

### 3.2 Strict budget protection

Opponent observation and troll-line safety checks are themselves searches. A strict `go nodes`, shallow fixed-depth, or sub-second `go movetime` request must not spend most of its budget on side searches before the requested move search begins.

The implementation extends the low-time gate to cover:

- Any request with an explicit node budget.
- Depth requests of two plies or less.
- Movetime requests below one second.
- Existing wall-clock and computed hard-budget thresholds.

The main search continues to receive the existing pre-processing charge mechanism for work that is allowed to run. This preserves adaptive behavior during ordinary games while preventing hidden overrun during protocol-constrained analysis.

### 3.3 Known-engine Full-mode depth recovery

When the opponent model has already reached the engine-suspect state, Full mode returns the best completed line and does not need adaptive candidate expansion. The implementation avoids automatically forcing `MultiPV >= 5` in that case when strength limiting is disabled. This recovers search depth without changing the Full selector or the opponent model.

### 3.4 Opponent-model evidence hygiene

Book observations now use the actual observed move ply rather than the final game ply. When several pending moves are processed together, the current clock-use signal is applied only to the newest observation. This prevents earlier history from being incorrectly treated as a late-game observation and prevents one clock measurement from being counted repeatedly.

### 3.5 Aegis hint-cache identity

The Aegis background hint key previously included the position hash, rating, time class, policy kind, exit, and legal action list. The position hash intentionally ignores dead en-passant flags and clock counters, while the neural input includes en-passant file and halfmove information. The key now includes the en-passant file and a halfmove bucket.

A regression test constructs three otherwise identical inputs and verifies that changing en-passant state or the halfmove bucket prevents equality. This is a stale-result prevention measure, not a playing-strength claim.

## 4. Portable hardware design

### 4.1 Build tiers

The earlier optimisation copy forced `target-cpu=native`, which is fast only when the binary remains on the same compatible CPU family. The current design removes that unconditional setting.

| Tier | Compiler policy | Intended deployment |
|---|---|---|
| Portable | Rust defaults with no host-specific flags | Legacy x86-64, broad Intel/AMD compatibility, virtual machines |
| Modern | Separately built `RUSTFLAGS='-C target-cpu=x86-64-v3'` | AVX2/FMA/BMI2-class consumer and server CPUs |
| Host-tuned | Separately labelled `RUSTFLAGS='-C target-cpu=native'` | A pinned machine whose instruction set and thermal behavior are known |

Runtime-specialised kernels should use feature detection and retain a scalar oracle. AVX-512 is not assumed. The Core Ultra 9 285H configuration documented in the repository favors AVX2 and AVX-VNNI rather than assuming AVX-512. Quantised NNUE and VNNI are separate model/runtime projects because the network format and numerical parity must be validated independently.

### 4.2 Cache-aware transposition tables

A transposition table is a random-access cache. Making it much larger than the shared last-level cache can increase memory latency on every probe. The implementation reads the Linux shared-L3 size from `/sys/devices/system/cpu/cpu0/cache/index3/size` and sets the default UCI `Hash` to approximately half that value, clamped to 4–128 MiB. An explicit UCI `Hash` setting overrides the automatic value.

The half-L3 target is a policy, not a universal optimum. It leaves cache space for executable code, thread stacks, board/search state, and evaluator data. Server users may benefit from larger tables, but the correct choice must be measured with the actual thread count and evaluator.

On the current VM, Linux reports 36,608 KiB of shared L3. The engine advertised an automatic Hash default of 17 MiB. The requested target examples are approximately 8 MiB for a 16 MiB-L3 Ryzen 7 7730U and 12 MiB for a 24 MiB-L3 Core Ultra 9 285H.

## 5. Comparative benchmark

### 5.1 Method

The benchmark built two independent release artifacts from the same source:

- Portable Rust release build.
- `x86-64-v3` release build using `-C target-cpu=x86-64-v3`.

Each build was tested with one search thread, adaptive mode disabled, and the opening book disabled. Four positions were used: the starting position, a castling-rich middlegame, a tactical middlegame, and a simplified endgame. Five Hash values were tested: 4, 8, 16, 32, and 64 MiB.

Each case received a 500,000-node maximum and a two-second graceful UCI window. Because the engine emits iterative-deepening information and may stop on the wall-clock boundary, observed nodes are not identical in every row. The reported `nps` is the engine's own last `info` value. Maximum resident set size was recorded with `/usr/bin/time`.

### 5.2 Results

| Build | Hash | Mean NPS | Median NPS | Mean RSS | Mean observed nodes |
|---|---:|---:|---:|---:|---:|
| Portable | 4 MiB | 1,823,284 | 1,517,852 | 23,677 KiB | 310,843 |
| Portable | 8 MiB | 1,933,064 | 1,625,244 | 27,681 KiB | 343,017 |
| Portable | 16 MiB | 1,871,675 | 1,568,616 | 35,746 KiB | 371,722 |
| Portable | 32 MiB | 1,908,441 | 1,620,951 | 51,628 KiB | 411,948 |
| Portable | 64 MiB | 1,820,084 | 1,549,680 | 83,673 KiB | 394,286 |
| x86-64-v3 | 4 MiB | 1,832,842 | 1,729,398 | 23,706 KiB | 310,843 |
| x86-64-v3 | 8 MiB | 1,986,948 | 1,740,700 | 27,703 KiB | 343,017 |
| x86-64-v3 | 16 MiB | 1,971,977 | 1,704,448 | 35,714 KiB | 371,722 |
| x86-64-v3 | 32 MiB | 1,924,170 | 1,634,655 | 51,644 KiB | 411,948 |
| x86-64-v3 | 64 MiB | 1,941,687 | 1,655,477 | 83,673 KiB | 394,286 |

The `x86-64-v3` mean-NPS advantage over portable was:

| Hash | v3 advantage |
|---:|---:|
| 4 MiB | +0.52% |
| 8 MiB | +2.79% |
| 16 MiB | +5.36% |
| 32 MiB | +0.82% |
| 64 MiB | +6.68% |

The benchmark suggests that the instruction-set tier provides a small-to-moderate throughput improvement on this VM. It does not establish that the same percentages apply to the Core Ultra 9 285H, Ryzen 7 7730U, legacy Intel CPUs, or server CPUs. The result also shows that larger Hash did not monotonically increase NPS. The 8–32 MiB range was generally more favorable than 64 MiB for this workload, which supports cache-aware defaults.

Resident memory increased with Hash size, while portable and v3 builds remained effectively equal in RSS. This is expected because both binaries allocate the same TT structure; the v3 difference is instruction selection, not table layout.

### 5.3 Reproducibility

The raw benchmark table is stored at `benchmarks/results/portable-v3-20260904-113843.tsv`. The harness is `scripts/benchmark-portable-v3.sh`, and the summarizer is `scripts/summarize-benchmark.py`. Results should be regenerated on each target CPU with fixed power mode, thread count, thermal state, operating-system scheduler, and model/evaluator.

## 6. Persona integration with an Lc0 verification pipeline

The persona system and Lc0 provider have separate responsibilities. The persona system decides what behavior the engine is trying to express. Lc0, when available, supplies an additional neural-search opinion or verification signal. The existing alpha-beta search remains the objective authority.

The integration is budget-tiered:

| Budget tier | Persona behavior | Lc0 behavior |
|---|---|---|
| Normal | Full observation, book logic, Clinch/Match probes when allowed, normal alpha-beta search | Optional warmed Lc0 verification if predicted latency fits |
| Fast | Reduced auxiliary work and bounded persona selection | Optional single bounded verification; no cold start or reconfiguration |
| Critical | Skip observation and luxury probes; preserve legal persona fallback | Disabled unless an already-completed result is available |
| Emergency | Immediate legal and safety-checked fallback | Disabled |

A normal move flows as follows. The engine reconstructs the position and legal moves. The opponent model and persona state are read. The alpha-beta search produces complete root lines. A candidate provider may propose a policy-like move or Lc0 may provide verification evidence. The active persona then applies its existing selection rules to the alpha-beta lines. Finally, the troll and tactical safety gates run before the move is emitted.

The Lc0 signal cannot directly convert Match into Full, cannot change Clinch's reply-gap calculation, cannot alter Punish's mate priority, and cannot replace Defend's maximum resistance. It can only affect a separately defined candidate-ranking or safety-verification stage that has sufficient remaining budget.

In low time, the gate occurs **before** spawning or waiting for Lc0. The planner subtracts measured inference, process, queue, stdout-drain, stop, and move-overhead percentiles from the hard deadline. If the predicted completion does not fit, Lc0 is not called. A late result is discarded by request token and position/model key. The engine returns to its locally generated alpha-beta/persona path.

The troll system is stricter. A neural provider may not introduce a new troll line, override the curated book, or disable the shallow refutation check. The safest initial policy is to disable Lc0 after a troll-book decision and use it only in future experiments as a verifier for already-approved candidates.

## 7. Neural root-prior firewall

The currently implemented internal root-hint path already contains several firewall properties. In `search::go_with_root_hints`, the engine first generates legal root moves. It optionally restricts them to `searchmoves`. For each legal root move, it searches the supplied hints for a matching move and accepts only finite policy scores. A missing or non-finite hint becomes negative infinity for ordering purposes. The root move list is never replaced by the hint list.

The essential behavior is equivalent to:

```rust
let root_moves = legal(pos);
let roots = root_moves.iter().map(|&mv| RootMove {
    mv,
    score: -MATE,
    policy_hint: root_hints
        .iter()
        .find(|hint| hint.mv == mv && hint.policy_score.is_finite())
        .map(|hint| hint.policy_score)
        .unwrap_or(f32::NEG_INFINITY),
    ..
}).collect();
```

The search then uses the policy score only to order the first iterative pass. The completed alpha-beta scores remain authoritative. Existing tests demonstrate that hostile or stale root hints cannot suppress legal moves or override forced mates.

The proposed external firewall adds validation before converting a provider response into `RootHint` values:

```text
1. Check request token and exact position hash.
2. Check model hash and schema version.
3. Check exact legal-action fingerprint.
4. Reject wrong vector length.
5. Reject duplicate or missing legal moves.
6. Reject moves not present in the supplied legal list.
7. Reject NaN, infinity, or non-finite probabilities.
8. Reject stale deadlines or responses from a previous position.
9. Normalize only after validation.
10. Convert to RootHint only for the supplied legal moves.
11. Fall back to baseline alpha-beta on any failure.
```

This design filters malformed UCI responses at two layers. The child-process adapter treats arbitrary `info` lines as non-terminal and completes only on `bestmove`. It validates `bestmove` against Unchessed's legal move list. The neural-provider layer independently validates the structured prior response. A malformed or illegal response is therefore a provider failure, not a search input.

## 8. AlphaZero, Lc0, Maia, and Stockfish relationship

AlphaZero established the public policy/value and self-play MCTS framework, including legal-action masking and visit-count training targets [7]. Lc0 is an open-source practical engine in that family, with neural policy/value inference and MCTS/PUCT search [3] [4]. Maia adapts related neural modelling ideas to predict human moves conditioned on skill and history [5] [6]. Stockfish demonstrates the strength of CPU-first selective alpha-beta combined with quantised NNUE evaluation [1] [2]. RubiChess provides another open-source UCI/NNUE implementation and portability reference [8].

These systems should be linked by typed evidence interfaces rather than by mixing their native scores. Stockfish centipawns, Lc0 visit counts, Lc0 value estimates, and Maia WDL are not calibrated to one common unit. A provider can report legal move, source, score type, latency, and confidence. A separate verifier can classify a candidate as safe, materially losing, mate-adjacent, tablebase-supported, or unknown.

## 9. Limitations and future work

The benchmark was performed on one virtual machine with one search thread and the current available evaluator configuration. The engine reported that no NNUE file was available in the smoke environment, so the measured NPS should not be interpreted as a production NNUE benchmark. The workload was short and included only four positions. It did not measure thermal throttling, NUMA effects, hybrid-core scheduling, memory bandwidth, AVX-VNNI, or real Core Ultra/Ryzen hardware.

The automatic Hash heuristic currently uses Linux sysfs and therefore falls back on non-Linux platforms. Future work should add a portable cache-detection layer, expose explicit cache profiles, and benchmark evaluator-plus-TT working sets rather than TT RSS alone.

The external Lc0 process adapter, typed provider trait, budget planner, and structured neural-prior response schema are specified but not fully implemented in this sub-project. The current implementation has an internal root-hint firewall and Aegis worker infrastructure. The external UCI supervision layer remains a separate development stage.

The most important strength work remains empirical. Clustered and aged transposition tables, capture and continuation history, direct legal generation, quantised NNUE inference, improved data curation, and learned priors should be introduced one at a time. Every tree-changing change requires correctness tests and a pre-registered sequential probability ratio test (SPRT). No claim of Stockfish parity is made without that evidence.

## 10. Conclusion

The sub-project established a validated and portable foundation for stronger adaptive play. It corrected several protocol and timing hazards that could misfire persona behavior, removed an unsafe universal host-specific compiler setting, and made transposition-table defaults responsive to the CPU's shared L3 cache. On the current VM, `x86-64-v3` improved one-thread NPS by 0.52–6.68% across tested Hash values, while memory scaling followed the transposition-table footprint.

The main architectural conclusion is that neural and external-engine capabilities should be **linked through a fail-closed evidence layer**. Alpha-beta remains the authority. Persona logic remains explicit. Lc0 and AlphaZero-inspired priors can provide useful policy or verification evidence only when legality, model identity, deadlines, and safety invariants are satisfied. This arrangement allows future strength improvements without sacrificing the humanisation, opponent modelling, troll safeguards, contempt semantics, or low-time stability that define Unchessed.

## References

[1]: https://github.com/official-stockfish/Stockfish "Official Stockfish repository"
[2]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html "Stockfish NNUE technical documentation"
[3]: https://lczero.org/dev/overview/ "Leela Chess Zero developer overview"
[4]: https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/ "Leela Chess Zero technical explanation of neural MCTS"
[5]: https://arxiv.org/abs/2006.01855 "Aligning Superhuman AI with Human Behavior: Chess as a Model System"
[6]: https://arxiv.org/abs/2409.20553 "Maia-2: A Unified Model for Human-Aware Chess"
[7]: https://arxiv.org/abs/1712.01815 "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
[8]: https://github.com/Matthies/RubiChess "Official RubiChess repository"
[9]: https://arxiv.org/abs/2412.17948 "Study of the Proper NNUE Dataset"
[10]: https://arxiv.org/abs/2402.04494 "ChessBench: A Universal Chess-Playing Model Grounded in Human Games"
[11]: https://backscattering.de/chess/uci/ "Universal Chess Interface protocol specification"


## Addendum: Persona stability and firewall audit

The persona state machine was strengthened after review. Persona smoothing is now enabled by default, while the explicit `PersonaSmooth` UCI option remains available for ablation. Smoothed mode uses the existing EMA and dwell logic, preserves immediate Full/Defend/Punish emergency transitions, and adds a two-update cooldown after a deliberate non-emergency transition. A conflicting proposal during cooldown is recorded for telemetry but cannot immediately reverse the active persona. The cooldown is exposed in opt-in persona telemetry, making transition stability auditable without allowing telemetry to influence selection.

A new regression test verifies that a Clinch transition cannot be immediately reversed by a subsequent conflicting evaluation. The post-change workspace run passed 125 tests, with six ignored, deep perft, release validation, and UCI smoke checks.

The root-prior audit confirms that the internal firewall is legal-set anchored. `go_with_root_hints` starts from generated legal moves, intersects them with `searchmoves` when requested, accepts only matching finite policy scores, and retains every legal root even when no hint is supplied. Existing tests cover non-finite and stale hints, check positions, and forced mates. The Aegis hint key now binds en-passant and halfmove state. These measures prevent malformed scores, stale rule-state reuse, illegal candidates, and neural ordering from overriding alpha-beta authority. The external Lc0 UCI process firewall remains a planned supervised provider layer and is not represented as completed implementation.
