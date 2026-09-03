# Lazy SMP depth staggering

**Investigation:** 39
**Branch:** `manus/research-facilities`
**Commit inspected:** `7348e5324daf9a6ef963f1c0a2ce7af581b86151`
**Scope:** source verification, bounded real multithread UCI smoke/depth comparison, and research only. No implementation, default change, commit, push, match campaign, or Tier 2/3 work was started.

## Executive summary

The repository does implement the claimed Lazy SMP depth staggering. In `unchessed-core/src/uci.rs:1651–1701`, the main search starts at depth `1`, while each helper starts at `offset = 1 + (i % 3)`, producing the repeating helper sequence **1, 2, 3, 1, 2, 3, ...**. With six configured threads (five helpers), the starting depths are therefore `main=1`, helpers `1,2,3,1,2`; the next helper would be `3` when there are seven total threads. The helper searches share the TT and run single-PV searches under the same limits; only the main thread's full MultiPV result is printed and selected.

A real fixed-FEN UCI run on the available six-logical-CPU host completed successfully for 1, 2, 4, and 6 threads at depth 10. The observed main-thread reported node counts were respectively **321,477**, **236,853**, **198,458**, and **241,130**; reported NPS was **1,806,050**, **1,728,854**, **1,653,816**, and **1,826,742**. Depth 10 completed in **178, 137, 120, and 132 ms**, respectively. Best moves differed between thread counts (`c1d2` at 1 and 4 threads; `f1e2` at 2 and 6 threads), demonstrating observable multi-thread search diversity/TT interaction, but this is not an Elo or scaling result. The experiment is too small and virtualized to establish a preferred staggering scheme.

The fixed modulo-3 cycle does not mathematically “saturate” diversity after three helpers: the cycle repeats starting depths, but concurrent scheduling, shared TT state, search timing, and each thread's local search state can still produce different trees. Conversely, the cycle guarantees no increasing depth offsets as thread count rises, so the marginal diversity signal is bounded and increasingly redundant once many helpers are present. Higher offsets also risk spending too much of a finite time budget on searches that cannot contribute a useful completed iteration. A scaling scheme is therefore a plausible diagnostic candidate, not a justified change from this evidence.

**Recommendation:** preserve the current default and do not change the helper-depth formula now. If this item is approved later, first add opt-in observability (per-thread start depth, completed depth, nodes, stop reason, root move/PV and TT hit/cutoff counters), then compare the current modulo-3 schedule against predeclared alternatives on a fixed multi-position suite and real paired games. Do not infer strength from the smoke data.

## Repository verification

The relevant implementation is:

```text
unchessed-core/src/uci.rs:1651–1658
    // Lazy SMP: helper threads share this TT ... each
    // run a single-PV search ... staggered to a different starting
    // depth ...
unchessed-core/src/uci.rs:1659
    let n_helpers = job.opt.threads.saturating_sub(1);
unchessed-core/src/uci.rs:1666–1669
    for i in 0..n_helpers {
        let offset = 1 + (i % 3) as i32; // stagger: 1,2,3,1,2,3,...
unchessed-core/src/uci.rs:1687–1699
    search::go(..., offset, ...)
unchessed-core/src/uci.rs:1703–1732
    search::go(..., 1, ... ) // main thread
```

`unchessed-core/src/search.rs:842–852` documents `start_depth`: passing `1` is normal, while helpers skip cheap shallow iterations to diverge from the main search sooner. `search.rs:846–850` also explicitly states that the TT is safe to share concurrently and that this is the Lazy SMP use case.

For `Threads=N`, the implementation creates `N-1` helpers. The exact start-depth schedule is therefore:

| Configured threads | Main start depth | Helper start depths | Full schedule including main |
|---:|---:|---|---|
| 1 | 1 | none | 1 |
| 2 | 1 | 1 | 1, 1 |
| 4 | 1 | 1, 2, 3 | 1, 1, 2, 3 |
| 6 | 1 | 1, 2, 3, 1, 2 | 1, 1, 2, 3, 1, 2 |
| 7 | 1 | 1, 2, 3, 1, 2, 3 | 1, 1, 2, 3, 1, 2, 3 |

The source does not contain a depth-offset scheme that grows with thread count, nor does it contain a per-depth participation counter, adaptive skip threshold, root-move randomization, or thread-voting result combiner in this path. Helpers are explicitly warm-up searches for the shared TT; the main result controls output.

## Real UCI experiment

### Environment and exact command

The host exposed six logical CPUs (`nproc` output `6`; Intel Xeon 2.50 GHz under KVM). The existing release executable was used:

```text
target/release/unchessed-adapter: ELF 64-bit LSB pie executable, x86-64
```

The same FEN, hash size, book setting, depth, and binary were used for every run. Exact command shape (executed once for each `t` in `1 2 4 6`) was:

```bash
for t in 1 2 4 6; do
  { printf 'uci\n';
    printf 'setoption name OwnBook value false\n';
    printf 'setoption name Threads value %s\n' "$t";
    printf 'setoption name Hash value 64\n';
    printf 'isready\n';
    printf 'position fen r1bq1rk1/ppp1nppp/2n5/3p4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQ - 3 6\n';
    printf 'go depth 10\n'; sleep 8;
    printf 'quit\n';
  } | timeout 15s target/release/unchessed-adapter 2>&1 |
    grep -E '^(info depth|bestmove|readyok|uciok|id )' | tail -30
done
```

### Captured result summary

| Threads | Depth-10 nodes | Depth-10 NPS | Time | Best move |
|---:|---:|---:|---:|---|
| 1 | 321,477 | 1,806,050 | 178 ms | `c1d2` |
| 2 | 236,853 | 1,728,854 | 137 ms | `f1e2` |
| 4 | 198,458 | 1,653,816 | 120 ms | `c1d2` |
| 6 | 241,130 | 1,826,742 | 132 ms | `f1e2` |

All four runs returned `uciok`, `readyok`, depth-10 `info`, and `bestmove`. The depth-10 lines reported scores of approximately +507, +507, +508, and +510 cp. Intermediate depth lines were also emitted, confirming that this was a real iterative UCI search rather than a source-only or mocked test. The single-thread run's depth progression included nodes `40, 275, 2,261, 7,118, ... 321,477`; the six-thread run included `40, 249, 1,399, 4,569, ... 241,130`.

The different best moves at equal depth are evidence that the multi-thread configuration changes search behavior in this engine. It is **not** evidence that either move is stronger, because no reference engine, tactical suite, repeated-position aggregate, or game result was used. The measurements are wall-clock-sensitive and were collected in a shared virtualized environment; NPS is not monotonic with thread count and should not be generalized to physical high-core machines.

A separate three-run six-thread attempt produced only `bestmove` lines because the shell fed commands without a delay before `quit`, so it is treated as a protocol-timing artifact and not as a depth comparison. It is reported here as a negative/quality caveat rather than silently discarded.

### Test/build caveat

The requested workspace test command could not run with the installed toolchain. Both commands failed before compilation:

```text
cargo test --workspace --locked
error: failed to parse lock file ... Cargo.lock
Caused by: lock file version 4 requires `-Znext-lockfile-bump`

cargo test --workspace
error: failed to parse lock file ... Cargo.lock
Caused by: lock file version 4 requires `-Znext-lockfile-bump`
```

This blocks a fresh source rebuild/test under the available Cargo, but not the real UCI experiment against the existing release binary. No lockfile, source, or default was modified to work around it.

## Research findings: scaling and diversity

The [Chess Programming Wiki Lazy SMP reference](https://chessprogramming.org/Lazy_SMP) defines Lazy SMP as multiple threads searching the same root with different depths and/or root move ordering, sharing a hash table so timing nondeterminism and cross-thread entries can improve the aggregate search. Its pseudocode describes helper depth offsets (including an even-helper increment) and notes that modern engines may use thread voting. It characterizes Lazy SMP as scaling surprisingly well up to eight cores and beyond, while also noting worse time-to-depth speedup than Young Brothers Wait. These are broad family-level observations, not measurements of this Rust engine.

The [official Stockfish terminology documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html) describes Lazy SMP as N threads sharing the TT, allowing faster TT filling and a faster/wider search, particularly for longer searches. It does not prescribe modulo-3 offsets or claim that one schedule is universally optimal.

The [TalkChess Lazy SMP discussion](https://talkchess.com/viewtopic.php?t=68278) is useful engineering evidence but not a controlled peer-reviewed benchmark. It reports that alternating depth starts can still converge, that more extreme odd/even iteration schemes may fail to help each other efficiently, and that a 50% depth-participation skip policy was reported as useful in one engine. Its example measurements show non-linear, hardware- and implementation-dependent scaling and a modulo-free half-at-depth/depth+1 arrangement. These observations support measuring completed-depth overlap and useful work rather than assuming that a larger numeric offset is automatically better.

The practical implications for this repository are:

1. **Diversity is multidimensional.** Starting depths are one source of decorrelation; shared TT entries, timing, local history/killers, aspiration outcomes, and move ordering also matter. Repeating `1,2,3` does not prove identical searches after the first three helpers.
2. **The current cycle bounds explicit depth diversity.** At seven or more threads, additional helpers repeat the same offsets. This may be sufficient if scheduling and TT races provide diversity, but it may become redundant on high-core hardware.
3. **Larger offsets have a cost.** A helper started too deeply can spend its budget on an incomplete iteration and contribute little useful TT information. A helper started too shallowly retraces work. The optimum depends on time control, branching factor, TT size, and synchronization/stop semantics.
4. **Scaling must distinguish NPS from time-to-depth and playing strength.** Official documentation and the wiki explicitly distinguish depth/rootDepth from selective depth and warn that Lazy SMP's speedup characteristics differ by metric. This experiment measured only the main UCI line's nodes/NPS/time and cannot establish any of those stronger claims.
5. **High-core validation is absent here.** The host has six logical CPUs, so it cannot test the requested “past a handful” behavior on 8, 16, or more physical cores. Oversubscribing this container would measure contention, not a useful high-core scaling curve.

## Verified versus assumed

| Claim | Status |
|---|---|
| Branch and inspected commit | **Verified:** `manus/research-facilities`, `7348e5324daf9a6ef963f1c0a2ce7af581b86151`. |
| Modulo-3 helper formula exists | **Verified:** `uci.rs:1667–1669`. |
| Helper offsets are `1,2,3,1,2,3,...` | **Verified:** direct source formula and comment. |
| Main starts at depth 1; helpers share TT and use single PV | **Verified:** `uci.rs:1651–1658`, `1669–1699`, `1703–1732`. |
| Six-thread UCI search reaches depth 10 | **Verified:** real release-binary UCI transcript. |
| Thread counts change nodes/time/best move | **Verified observationally** for this one FEN/run; not a strength claim. |
| Modulo-3 saturates useful diversity on high-core hardware | **Not verified:** no high-core hardware or diversity telemetry. |
| A scaling formula improves NPS, depth, or Elo | **Assumed/speculative:** requires controlled fixed-suite and paired-game evidence. |
| Workspace tests pass | **Not verified/blocked:** installed Cargo rejects lockfile version 4. |
| No implementation/default change occurred | **Verified:** only report file was written; no source/config change was made. |

## Recommendation and next gate

**Do not implement or alter the default staggering in this investigation.** Retain modulo-3 as a reasonable low-complexity, default-preserving heuristic, while recording that its explicit offset diversity is bounded. The next evidence gate, if separately approved, should be an opt-in instrumentation patch—not a behavior change—with a fixed suite containing quiet, tactical, endgame, and high-branching positions. For each position and thread count, record per-thread start/completed depth, nodes, elapsed time, stop reason, root move/PV, TT probes/hits/cutoffs, and whether a helper completed an iteration before the main search stopped. Compare the incumbent against at least one predeclared schedule (for example, bounded offsets or a participation-aware schedule) under cold and warm TT conditions.

Only after those diagnostics show a repeatable net benefit should an isolated candidate proceed to real paired-game testing with immutable binary/options/hardware manifests. Any change to search behavior or defaults remains outside this report and requires the repository's established SPRT gate.

## Sources

1. [Chess Programming Wiki: Lazy SMP](https://chessprogramming.org/Lazy_SMP).
2. [Official Stockfish Docs: Terminology — Lazy Symmetric Multiprocessing](https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html).
3. [TalkChess: Lazy SMP ideas](https://talkchess.com/viewtopic.php?t=68278).
4. [Chess Programming Wiki: Parallel Search](https://chessprogramming.org/Parallel_Search).

**Final scope statement:** research and verification only; no Tier 2/Tier 3 implementation, expensive training, cloud spend, default change, commit, or push was performed.
