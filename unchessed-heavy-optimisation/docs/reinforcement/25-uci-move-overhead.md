# 25 — Tier 1: UCI `MoveOverhead`

**Investigation ID:** `tier1-uci-move-overhead`
**Tier:** 1 — protocol completeness / infrastructure
**Status:** Research and design only. No Rust implementation, UCI default change, match, benchmark, or SPRT was performed.

## Executive conclusion

`MoveOverhead` is a small but genuine UCI interoperability gap. The engine already has unusually explicit time management—clock/increment allocation, movestogo handling, low-clock urgency tiers, a hard reserve, situation-based scaling, and a fixed `movetime` reserve—but there is currently no user-configurable allowance for GUI, process-scheduling, or network delay. A GUI that uses a loaded host or remote transport cannot communicate that delay, so the engine can consume the entire reported budget and lose on time or make near-instant moves after the GUI/network path is accounted for.

The safe design is to add the standard UCI spin option **`Move Overhead`**, with **default `0 ms`**, minimum `0`, and maximum `5000` (or a deliberately documented lower maximum). Default zero is the important safety choice: it preserves all current search timing and avoids silently changing existing engine-engine results. When nonzero, the value should be reserved from the per-move wall-clock budget, without changing `go infinite`, fixed-depth, node-limited, or analysis behavior. The option should be applied at search-budget construction, not by sleeping after a move and not by modifying the GUI-reported clocks.

Recommendation: **pursue implementation as a low-risk protocol fix, but defer shipping the code until the focused parser and budget tests below pass.** This is not a search-strength proposal and does not require a playing-strength SPRT merely to expose a default-zero compatibility option. Any nonzero default or policy that changes game-search allocation materially remains subject to the master brief’s paired-game gate.

## What the protocol means

The canonical UCI description makes `Move Overhead` an engine option whose value is a number of milliseconds. Current official Stockfish documentation advertises:

> `Move Overhead` — `type spin default 10 min 0 max 5000`: Assume a time delay of x ms due to network and GUI overheads.

The same documentation says that a value above the default is useful to avoid time losses or near-instant moves, especially without increment, and that the suitable value depends on local versus loaded or networked operation [1]. The UCI protocol itself defines option advertisement as the mechanism by which the engine tells the GUI which parameters it supports; a `spin` option has an integer range and default [2]. These sources establish the **interface semantics**, not a mandatory internal formula: UCI does not prescribe whether an engine subtracts the value from soft time, hard time, both, or a separate deadline.

The repository’s requested safety posture is therefore preferable to copying Stockfish’s current default. A default of 10 ms would be a behavior change for every existing game. A default of zero is equivalent to “feature available, no reservation unless explicitly requested,” and is compatible with existing timing evidence.

## Repository inspection

### Existing UCI surface

`unchessed-core/src/uci.rs` owns an `Options` value, advertises options during `uci`, parses `setoption name ... value ...`, clones the options into each `GoJob`, and joins the prior worker before handling a new `go`, `position`, or `ucinewgame`. The parser lowercases option names, so both the standard spelling `Move Overhead` and case variants can be accepted by the existing path. `parse_go` currently recognizes `depth`, `movetime`, `wtime`, `btime`, `winc`, `binc`, `movestogo`, `nodes`, and `infinite`; overhead is correctly an option, not a `go` token.

The current UCI advertisement has no `Move Overhead` line. Existing option handling clamps numeric settings rather than rejecting out-of-range values. The implementation should follow that established behavior: parse a signed/unsigned integer safely, clamp to `[0, 5000]`, and leave the previous value unchanged on malformed input (or explicitly choose the project’s existing invalid-value convention and test it).

### Existing budget calculation

`unchessed-core/src/search.rs` defines `Limits` and its private `budget(side)` method. Relevant current behavior is:

* `go infinite` returns `(None, None)` and therefore has no wall-clock deadline.
* `movetime` returns equal soft and hard budgets after `25 ms` is subtracted, with a minimum of `5 ms`.
* Clock searches allocate `t / movestogo + 3/4 increment` soft time and approximately `t / 5 + 1/2 increment` hard time.
* Below 20 seconds and 6 seconds, increasingly urgent caps apply; below 2 seconds, panic mode keeps a small reserve and favors the increment.
* A final ceiling is `t - 60 ms` (with a minimum of `5 ms`), then soft is bounded by hard.
* `go()` records preprocessing time by moving its effective start backwards, obtains `(base_soft, hard_ms) = limits.budget(pos.side)`, and applies a situation multiplier based on legal-move count and check status. Search polling checks the hard limit periodically.

The existing unit tests verify game-mode detection, shrinking budgets, panic behavior, and a hard reserve. The code already passes time information into one centralized calculation, which is the right seam for this feature.

## Proposed design (no implementation in this investigation)

### State and advertisement

Add a field to the UCI options state, conceptually:

```text
move_overhead_ms: u64 = 0
```

Advertise exactly one standard-compatible line during `uci`:

```text
option name Move Overhead type spin default 0 min 0 max 5000
```

Handle `setoption name Move Overhead value N` in the existing option dispatcher. Clamp valid numeric input to `0..=5000`; reject malformed input without changing the prior value. The option persists across `position` and `ucinewgame`, as other UCI options do. It must be copied into the `GoJob` snapshot, so a running worker observes a stable value and a subsequent command cannot race a mutable global option.

### Budget application

Keep `Limits` as the representation of GUI-reported controls and pass the option value as a separate budget parameter, or construct an equivalent immutable “effective limits” value at the worker boundary. Do not rewrite the reported `wtime`/`btime` values: those values are also used for opponent-clock telemetry and should remain the GUI’s facts.

A suitable conceptual helper is:

```text
budget(side, move_overhead_ms) -> (soft_ms, hard_ms)
```

with this ordering:

1. If `infinite`, return unlimited budgets exactly as today; overhead must not turn analysis into a timed search.
2. For `movetime`, reserve overhead before applying the existing 25 ms safety reserve: `effective = movetime - overhead - 25`, saturating and retaining the existing 5 ms floor. This preserves current behavior at overhead zero.
3. For clock control, compute an `effective_clock = reported_clock - overhead` with saturating subtraction. Use that effective clock in the existing soft/hard allocation and urgency tiers. Do **not** subtract overhead from the increment; the increment is additional thinking time supplied by the time control, whereas overhead is a per-move transport/scheduling reserve.
4. Preserve the existing final reserve and minimums. If overhead is greater than the reported clock, the engine cannot guarantee avoiding a flag; fail closed to the existing minimum budget rather than underflowing or creating a large unsigned budget.
5. Apply the existing situation multiplier only after the overhead-aware base budgets are made, and clamp against the overhead-aware hard limit as today.
6. Leave depth-only, node-only, and unbounded searches unchanged. The option is about wall-clock move timing, not a hidden node multiplier.

There are two defensible variants for `movetime`: subtract overhead from the exact movetime limit, as above, or treat explicit `movetime` as already a caller-provided engine budget and apply overhead only to clock controls. The first is more faithful to the standard phrase “delay due to network and GUI overheads,” and is recommended, but it must be documented because users often expect `go movetime N` to cap engine thinking at exactly N. In either variant, default zero is behavior-preserving.

### Why not add overhead after search?

Sleeping after selecting a move cannot protect the clock and would make time losses worse. It also makes the engine’s reported `info time` and stop behavior misleading. The allowance belongs before allocation/deadline construction, where it can reduce both soft and hard wall-clock budgets while retaining the engine’s existing urgency and reserve logic.

### Why not simply increase the existing fixed reserves?

The current 25 ms `movetime` reserve and 60 ms clock ceiling are internal safety margins, not configurable transport delay. Folding the option into those constants would make the relationship opaque and could double-count overhead. The option should be a distinct input, with the existing reserves still applied so default-zero is exactly the current algorithm.

## Test plan

These are design requirements for implementation; they were not run because this task explicitly forbids implementation and the available host has an older Cargo toolchain unable to parse the repository’s lockfile v4 (as recorded by neighboring reinforcement documentation).

| Area | Test | Expected result |
|---|---|---|
| Advertisement | Capture `uci` output | Exactly one `option name Move Overhead type spin default 0 min 0 max 5000` line; `uciok` still follows options. |
| Parsing | Set `0`, `37`, `5000`, a value above max, a negative value, and malformed text | `0`, `37`, and `5000` retained; above-max clamps to 5000; invalid input does not create a negative/overflow value or silently enable a large reserve. |
| Case/spacing | `setoption name Move Overhead value 37` plus case variants | Existing case-insensitive option path stores 37. |
| Lifecycle | Set overhead, then `position` and `ucinewgame` | Value persists; game/TT reset behavior is unchanged. |
| Default equivalence | Compare `budget(side, 0)` with the pre-feature budget for movetime and multiple clock/increment/movestogo inputs | Identical soft/hard values, including low-clock and panic tiers. |
| Monotonicity | For the same control, compare overhead 0, 10, 100, and 5000 | Effective soft and hard budgets never increase as overhead increases. |
| Saturation | Overhead greater than movetime or reported clock | No unsigned wrap; result remains within existing minimums and hard is never below soft. |
| Clock semantics | Same reported clock and increment with overhead added | Only the clock-derived reserve is reduced; increment remains additive; no change to `wtime`/`btime` telemetry fields. |
| `movetime` semantics | `go movetime 1000` with overhead 0 and 100 | Zero matches old 975 ms budget; nonzero is lower by the requested reserve subject to floors. |
| Analysis | `go infinite` with overhead 5000 | Still unlimited; no deadline is created. |
| Fixed limits | `go depth N` and `go nodes N` with overhead 5000 | Node/depth semantics and exact node enforcement are unchanged. |
| Worker snapshot | Set option between completed searches and issue a new `go` | New job uses new value; command joining prevents a concurrent mutable-option race. |
| Regression | Existing time-management tests and the full workspace suite | Existing tests remain green; no parity gate or evaluator path is touched. |

A minimal implementation should add pure helper tests first, then UCI advertisement/parser tests. A live elapsed-time test is useful only as a coarse guard because scheduler jitter makes exact wall-clock assertions flaky; pure budget algebra should carry the correctness burden.

## Risk assessment and recommendation

The main correctness risks are integer underflow, double-counting the existing reserves, accidentally applying overhead to analysis or node-limited searches, and changing opponent-clock telemetry by mutating `Limits`. All are avoidable with a separate immutable option parameter and saturating arithmetic. The feature does not alter evaluation, move ordering, pruning, or move choice except indirectly when a user explicitly requests a nonzero reserve and search consequently receives less time.

The practical benefit is strongest for network play, loaded machines, and GUIs that impose command/response latency. On a dedicated local engine-engine host, zero is safe and should preserve current measurements. Stockfish’s documented default of 10 ms is evidence that the option is conventional, not evidence that 10 ms is right for this repository’s host or harness. There is no basis here for changing the default or claiming Elo gain.

**Recommendation: pursue the default-zero implementation as a protocol-completeness fix; defer any nonzero default.** After implementation, run the focused tests and workspace regression suite. A paired-game SPRT is unnecessary for the default-zero parser/compatibility change, but would be required before selecting a nonzero shipped default or presenting timing changes as a strength improvement.

## Verification and limitations

| Item | Status |
|---|---|
| Branch | Verified from repository context as `manus/research-facilities`; no source changes made. |
| Master brief | Read `/home/ubuntu/upload/pasted_content_6.txt`, including Tier 1 item 17 and standing rules. |
| Existing reinforcement docs | Read the available `docs/reinforcement` material, including 00–12 context and neighboring UCI analysis-mode report; this report follows the project’s design-only/default-preserving convention. |
| Repository code | Inspected `unchessed-core/src/uci.rs` option dispatch, `parse_go`, worker snapshot/lifecycle, and `unchessed-core/src/search.rs` `Limits::budget`, `go`, and timing tests. |
| External research | Read the canonical UCI protocol text and current official Stockfish UCI documentation; claims about option type, GUI/network delay, and documented range/default are linked below. |
| Implementation | Not performed, by request. |
| Tests | Not run for this report. No passing test result is claimed. Neighboring documentation records a Cargo lockfile/toolchain compatibility blocker on this host. |
| Strength evidence | No benchmark, game, SPRT, or Elo claim was made. |

## References

[1] [Stockfish official documentation, “UCI Protocol and Stockfish Commands,” Move Overhead](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html). Documents `Move Overhead` as a spin option, default 10, range 0–5000, and describes reserving time for network/GUI delay.

[2] [Universal Chess Interface protocol description](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf). Defines `option`, `spin`, `default`, `min`, and `max`, and the GUI/engine option exchange.

[3] Repository source: [`unchessed-core/src/uci.rs`](../../unchessed-core/src/uci.rs), especially option advertisement/dispatch, `parse_go`, and `GoJob` construction.

[4] Repository source: [`unchessed-core/src/search.rs`](../../unchessed-core/src/search.rs), especially `Limits::is_game_mode`, `Limits::budget`, `go`, deadline polling, and existing time-management tests.

[5] Repository context: [`docs/reinforcement/11-tier1-synthesis.md`](11-tier1-synthesis.md), [`docs/reinforcement/12-tier2-calibration.md`](12-tier2-calibration.md), and [`docs/reinforcement/26-uci-analyse-mode.md`](26-uci-analyse-mode.md).

**Report status:** complete; design and research only, no implementation or default change.
