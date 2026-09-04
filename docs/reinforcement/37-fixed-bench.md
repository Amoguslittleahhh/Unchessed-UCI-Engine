# 37 — Fixed bench and NPS regression

**Date:** 2026-09-03
**Branch:** `manus/research-facilities`
**Scope:** Tier 1 research and measurement only. No implementation, default change, training, match, commit, or push was performed.

## Executive summary

The repository has **no standard UCI `bench` command**, no fixed general-search position corpus, and no committed NPS regression runner. The search engine does support the UCI `go nodes N` limit, and an existing release binary was exercised against real chess positions. The binary returned legal `bestmove` responses and depth/NPS information, but its current output does not expose a final aggregate node/time record suitable for a stable benchmark signature. The tested node-limited searches also stopped at an observed completed iterative-deepening depth rather than reporting an exact requested-node total in the UCI output.

There is an existing ignored Rust benchmark suite in `unchessed-core/src/aegis_v4_runtime.rs`, including `benchmark_forward_pass`, but it measures the **Unarchitectured neural-network forward path**, not integrated alpha-beta search throughput. Its comments and test code explicitly describe warmups and timed forward calls. Existing repository benchmark artifacts likewise focus on Unarchitectured inference and worker/NPU/runtime experiments; they are not a fixed search bench.

A lightweight fixed bench is worth placing in **Tier 1 as default-preserving measurement infrastructure**, but only as a separately approved implementation. It should be deliberately small, deterministic, and explicit about what it measures: one thread, fixed hash, fixed FEN list committed in the repository, `OwnBook=false`, a fixed node limit per position, and aggregate `nodes`, wall-clock time, and NPS. It should also emit a configuration/provenance line and distinguish a **node-count signature** from a speed measurement. This is not a strength claim and must not be used to promote search changes. Real-world testing was completed within Tier 1 budget; the Rust test route was additionally blocked by the local Cargo 1.75 / lockfile mismatch.

## Questions and verified answers

| Question | Result | Evidence/status |
|---|---|---|
| Does a standard UCI `bench` command exist? | **No evidence found; source scan negative.** | Targeted `rg` over `unchessed-core/src`, tools, scripts, README, and docs (excluding reinforcement reports) found no engine `bench` or `speedtest` dispatch. The UCI command loop handles `uci`, `isready`, `setoption`, `ucinewgame`, `position`, `go`, etc., but no `bench` case. |
| Does a fixed general-search position set exist? | **No.** | `find` found `benchmarks/matetrack.epd`, but no general engine bench runner or fixed search corpus. The existing benchmark files are Unarchitectured-v1 runtime/data artifacts. |
| Is fixed NPS regression already present? | **No.** | No aggregate search benchmark or regression comparison script was found. UCI `info` computes per-iteration NPS only. |
| Is node-limited UCI search present? | **Yes.** | `unchessed-core/src/uci.rs:980` parses `nodes`; the UCI protocol route passes it into search. |
| Was a real engine search run? | **Yes.** | Existing `target/release/unchessed-adapter` was run with `go nodes 10,000` and two real FENs at 20,000 nodes, plus three repetitions at 50,000 nodes. |
| Could the source/Cargo benchmark test be run? | **No, due to a concrete toolchain blocker.** | Cargo 1.75.0 rejects repository `Cargo.lock` version 4: `lock file version 4 requires -Znext-lockfile-bump`. No lockfile edit or workaround was made. |

## Repository inspection

The supplied update identifies this topic as “No `bench` command / fixed NPS regression” and says the only ad-hoc microbenchmarks test network/hint code rather than general search throughput. A fresh targeted scan agrees with that premise. The relevant existing runtime test is:

```text
unchessed-core/src/aegis_v4_runtime.rs:3246  #[test]
unchessed-core/src/aegis_v4_runtime.rs:3247  #[ignore]
unchessed-core/src/aegis_v4_runtime.rs:3248  fn benchmark_forward_pass() {
unchessed-core/src/aegis_v4_runtime.rs:3252      for _ in 0..5 { forward(&weights, &input); }
unchessed-core/src/aegis_v4_runtime.rs:3255      let n = 200;
unchessed-core/src/aegis_v4_runtime.rs:3256      let started = std::time::Instant::now();
unchessed-core/src/aegis_v4_runtime.rs:3257      for _ in 0..n { black_box(forward(&weights, &input)); }
unchessed-core/src/aegis_v4_runtime.rs:3261      println!("{} calls in {:?} -> {:?}/call", n, elapsed, elapsed / n);
```

The same module contains ignored `benchmark_integer_matrix_speedup`, `benchmark_exit_ladder`, and other runtime-specific tests. Its module comment records an earlier real-hardware forward-pass measurement (Core Ultra 9 285H) of 7.92 ms at one internal thread, rising monotonically to 16.42 ms at eight threads. That is useful evidence for the Unarchitectured runtime decision, but it is **not** an integrated search NPS baseline.

The UCI parser contains `"nodes" => num(&mut l.nodes)` at line 980. The command loop also accepts `go` and launches the search worker. No command dispatch for `bench` or `speedtest` was found. `print_info` calculates `nps` from each `InfoEvent` as `nodes * 1000 / time_ms` (or `nodes * 1000` when time is zero), then prints it in an ordinary UCI `info` line. This is per-iteration telemetry, not a final aggregate benchmark result.

The repository state during inspection was:

```text
HEAD 7348e53 (manus/research-facilities, origin/manus/research-facilities) Expand Tier 1 engine research facilities
untracked existing files: docs/reinforcement/33-pondering.md, docs/reinforcement/35-quantized-nnue.md
```

No existing file was modified except the requested new report.

## Real-world runs

### 1. Existing release binary, start position, node-limited smoke

Exact command:

```sh
cd /home/ubuntu/Unchessed-UCI-Engine
printf 'uci\nisready\nucinewgame\nposition startpos\ngo nodes 10000\nquit\n' \
  | timeout 20s ./target/release/unchessed-adapter 2>&1
```

Relevant output:

```text
id name Unchessed Game Adapter 0.2.3
option name Threads type spin default 6 min 1 max 64
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove e2e4
```

This verifies a real UCI node-limited invocation completed with a legal-looking best move and did not hang. It emitted no `info depth` lines before completion, so this particular 10,000-node run provides no usable NPS sample. The startup output also verifies that this binary used hand-crafted evaluation because no NNUE file was found; it must not be compared as though it were the shipped NNUE configuration.

### 2. Two real FENs, fixed 20,000-node requests

Exact command:

```sh
cd /home/ubuntu/Unchessed-UCI-Engine
{
  printf 'uci\nisready\nsetoption name Hash value 16\nucinewgame\nposition fen r1bq1rk1/pppp1ppp/2n2n2/8/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1\ngo nodes 20000\n'
  printf 'position fen 8/8/8/8/8/2k5/5K2/7R w - - 0 1\ngo nodes 20000\n'
  printf 'quit\n'
} | timeout 30s ./target/release/unchessed-adapter 2>&1
```

Relevant search output:

```text
info depth 1 multipv 1 score cp 691 nodes 38 nps 38000 hashfull 0 time 0 pv d2d4
info depth 2 multipv 1 score cp 630 nodes 237 nps 237000 hashfull 0 time 0 pv d2d4 d7d6
info depth 3 multipv 1 score cp 646 nodes 1364 nps 1364000 hashfull 0 time 0 pv d2d4 d7d6 c1e3
bestmove d2d4
info depth 1 multipv 1 score cp 620 nodes 30 nps 30000 hashfull 0 time 0 pv h1c1 c3d4
info depth 2 multipv 1 score cp 623 nodes 174 nps 174000 hashfull 0 time 0 pv h1h8 c3d4
info depth 3 multipv 1 score cp 630 nodes 474 nps 474000 hashfull 0 time 0 pv h1h7 c3d4 h7d7
info depth 4 multipv 1 score cp 639 nodes 1965 nps 1965000 hashfull 0 time 0 pv f2e3 c3c2 h1h8 c2c3
bestmove h1h7
```

The two FENs completed successfully. Output is host-clock coarse: several iterations report `time 0`, so their displayed NPS is the fallback `nodes * 1000`, not a measured per-second rate. The first position reached depth 3 with 1,364 reported nodes; the second reached depth 4 with 1,965 reported nodes. Neither output provides a final aggregate node total or confirms that exactly 20,000 nodes were consumed. This is a negative result for using current UCI output as a regression harness without an additional wrapper or engine-side aggregate output.

### 3. Repeated fixed position, one thread, 50,000-node request

Exact command:

```sh
cd /home/ubuntu/Unchessed-UCI-Engine
for i in 1 2 3; do
  printf 'setoption name OwnBook value false\nsetoption name Threads value 1\nucinewgame\nposition fen r1bq1rk1/pppp1ppp/2n2n2/8/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1\ngo nodes 50000\nquit\n'
done | timeout 30s ./target/release/unchessed-adapter 2>&1 \
  | grep -E 'info depth|bestmove|eval:'
```

Observed output:

```text
info depth 1 multipv 1 score cp 691 nodes 38 nps 38000 hashfull 0 time 0 pv d2d4
info depth 2 multipv 1 score cp 630 nodes 295 nps 295000 hashfull 0 time 0 pv d2d4 d7d6
info depth 3 multipv 1 score cp 646 nodes 1596 nps 1596000 hashfull 0 time 1 pv d2d4 d7d6 c1e3
bestmove d2d4
```

Because the adapter command loop exits on the first `quit`, the piped three-run sequence did **not** produce three independent runs; only the first run was consumed. This is an important harness caveat and a negative result: a benchmark wrapper must keep one process per run or wait for `bestmove` and then issue the next `position/go`, rather than queueing `quit` after each trial.

The successful first run reached depth 3 and reported `time 1` ms at 1,596 nodes, yielding a minimally timed displayed NPS of 1,596,000. It is not a reliable regression baseline: the process was a pre-existing release binary, the NNUE file was absent, the requested node limit was not visibly reached, and the search was too short for stable clock resolution.

## Build/test blocker

The documented runtime benchmark command was attempted exactly:

```sh
cd /home/ubuntu/Unchessed-UCI-Engine
cargo test -p unchessed-core --release benchmark_forward_pass -- --ignored --nocapture
cargo test -p unchessed-core --lib
```

Both failed before compilation:

```text
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`
```

The local toolchain is `rustc 1.75.0` / `cargo 1.75.0`; the repository lockfile is version 4. An existing `target/release/unchessed-adapter` (ELF x86-64, timestamp 2026-09-02) allowed the independent UCI smoke above. I did not edit `Cargo.lock`, install or switch toolchains, or fabricate a source-test result. The blocker applies to the Rust benchmark/test route only, not to the completed binary smoke.

## Lightweight design research

The authoritative [Stockfish UCI and command documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html) describes `bench` as a search over a pre-selected assortment of positions that prints total nodes and elapsed time. It explains two distinct uses: a combined node-count signature (“fingerprint”) for verifying that binaries implement the same search algorithm, and a basic NPS speed benchmark. The documentation recommends the newer `speedtest` command for speed measurement. It specifies parameters for hash, threads, limit, FEN source, and limit type, including fixed node limits.

The [official Stockfish `benchmark.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/benchmark.cpp) implements this model: a built-in default FEN list, optional current position or file, explicit hash/thread settings, and `go depth`, `go nodes`, `go movetime`, or perft limits. Its source comments explicitly call out `bench 64 1 100000 default nodes` as a fixed-node mode. This is authoritative precedent for separating a reproducible node signature from wall-clock NPS.

The [UCI protocol specification](https://backscattering.de/chess/uci/) defines `go nodes x` as “search x nodes only” and requires a `bestmove` when search stops. It also advises sending `isready` after `ucinewgame`. The proposed harness should follow these protocol requirements and should not infer a stable NPS from zero-millisecond UCI iterations.

## Proposed Tier 1 design (not implemented)

A minimal implementation could be a developer-only CLI subcommand or standalone script, rather than a new default UCI protocol command. It should:

1. Use a committed, reviewed FEN file containing roughly 12–32 positions spanning opening, middlegame tactics, quiet positional play, endgame, promotion, check, and edge legality cases. Keep the set stable; changing it requires an explicit benchmark-version identifier.
2. Force `Threads=1`, a fixed small `Hash` (for example 16 or  Hash chosen and documented), `OwnBook=false`, no adaptive/persona/hint options, and an explicitly named evaluation/model file. A missing model must be a hard error or be clearly recorded; the hand-crafted fallback must not silently mix with NNUE baselines.
3. Run each FEN after `ucinewgame`, then `position fen ...`, then `go nodes N`, and wait for `bestmove`. Use a sufficiently large but Tier-1-cheap N (for example 50k–200k per position) so wall-clock timing is above the host timer resolution; calibrate once on the local machine rather than prescribing a strength budget.
4. Record per-position requested limit, observed final node count if the engine exposes it, elapsed monotonic milliseconds, bestmove, and depth. Print aggregate total nodes, total milliseconds, and `total_nodes * 1000 / total_ms`. Also print git revision, binary hash, compiler/build profile, CPU information, thread/hash/evaluation options, benchmark-set hash, and benchmark schema version.
5. Keep two gates separate: **node signature** (algorithm/tree/protocol regression, expected to change when search behavior changes) and **speed/NPS** (performance regression, noisy across hosts). Do not fail a cross-machine NPS gate on an absolute number. For local CI, use repeated runs and a generous noise-aware threshold; report median and spread rather than one run.
6. Avoid making `bench` a normal UCI command unless there is a concrete GUI/automation need. A standalone `tools/bench.py` or a developer CLI reduces protocol surface and can orchestrate one process per position safely. If a UCI command is preferred, it must be documented as non-standard, reject malformed parameters, and aggregate internally rather than relying on coarse `info` lines.

### Why it is worth Tier 1

The cost is low and the benefit is operational: it would catch accidental search-speed regressions, missing SIMD/build flags, changed node accounting, and option/default drift before an expensive match. It complements (rather than replaces) tactical safety tests and SPRT. A fixed bench cannot establish Elo, evaluate a new pruning rule, or justify a default. The node signature will intentionally change for legitimate search changes, so the report must classify changes rather than treating every delta as a failure.

### Limits and risks

NPS is hardware-, compiler-, thermal-, model-, thread-, hash-, and OS-sensitive. A tiny position set can overfit to one search phase; a larger set costs more but remains cheap at node limits in this project. The current engine’s per-iteration `time 0` behavior demonstrates why timing must be measured by an external monotonic clock or an aggregate engine timer. `go nodes` exactness also needs an explicit acceptance test: the existing synthesis records that the prior polling cadence could overshoot, while current output here did not expose enough information to validate exact enforcement. A bench must therefore record actual visited nodes from a trusted final event or separately test the node-budget contract; it must not silently call the requested limit the measured total.

## Verified versus assumed

**Verified:** no `bench`/`speedtest` engine dispatch was found in the targeted source/repository scan; no fixed integrated-search FEN suite or NPS regression runner was found; ad-hoc Unarchitectured runtime benchmarks exist; UCI parses `go nodes`; the existing release binary completed real node-limited searches and returned best moves; output can have `time 0`; the local Cargo test path is blocked by lockfile/toolchain incompatibility; Stockfish provides authoritative precedent for fixed positions, fixed node limits, aggregate totals, and separate signature/speed concepts.

**Not verified:** exact node-limit compliance for the current binary; stable NPS on the current host; NNUE-mode integrated throughput; reproducibility across repeated independent processes; tactical correctness or Elo impact; whether a specific FEN count or node budget is optimal. Those require a future approved harness and, for strength/default claims, real paired-game testing.

## Recommendation

**Recommend Tier 1 implementation approval for a lightweight, default-preserving fixed search bench/NPS report, but do not start it under this research item.** Prefer a standalone developer harness first. Make the first version a provenance-rich measurement tool with a stable FEN file, one-thread/fixed-hash controls, `OwnBook=false`, explicit model/fallback status, aggregate monotonic timing, and separate node-signature and NPS outputs. Add an exact `go nodes` acceptance test before using node totals as a signature. Do not add a CI hard failure based on one host’s raw NPS, do not change engine defaults, and do not treat the result as strength evidence. The current evidence supports the infrastructure as worthwhile Tier 1 hygiene, not Tier 2 search optimization or a Tier 3 campaign.

## References

1. [Stockfish UCI Protocol and Stockfish Commands — official documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html), sections `bench` and `speedtest`.
2. [Stockfish `src/benchmark.cpp` — official source](https://github.com/official-stockfish/Stockfish/blob/master/src/benchmark.cpp), `setup_bench` and fixed default FEN benchmark construction.
3. [Universal Chess Interface specification](https://backscattering.de/chess/uci/), `go nodes x`, `ucinewgame`, `isready`, and `bestmove` requirements.

**Report status:** research complete; no benchmark implementation was begun.
