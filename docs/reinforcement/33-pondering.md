# 33 — Pondering and `ponderhit`

**Investigation ID:** `tier1-pondering`

**Tier:** 1 — UCI protocol and time-management research

**Status:** Research/design only. No Rust implementation, default change, match, benchmark campaign, commit, push, Tier 2/3 work, or cloud spend was started.

## Executive conclusion

The repository has **no implemented pondering path**. There is no `ponder` or `ponderhit` handling in the UCI loop or search code, and the existing release binary does not advertise a `Ponder` option. A live UCI probe nevertheless showed an important negative result: `go ponder ...` is accepted only accidentally because the parser ignores unknown `go` tokens, while `ponderhit` is ignored as an unknown top-level command. The engine then performs an ordinary search and emits an ordinary `bestmove`; this is not partial pondering support.

Pondering is a potentially useful protocol feature, but it is not a safe parser-only addition. It changes worker lifecycle, expected-position state, stop/race handling, and the boundary between opponent time and the engine's own clock. The safe recommendation is **defer implementation until a narrowly scoped protocol design is approved**. If pursued, implement it as an opt-in `Ponder` check option, with a separate immutable ponder state and an explicit transition on `ponderhit`; do not reinterpret ignored tokens as support and do not let speculative time consume the engine's post-hit budget. Preserve all current behavior when `Ponder` is false.

No Elo or speed benefit is verified. The only local runtime evidence is protocol behavior and ordinary search output. A future implementation needs parser/state tests and a real GUI-like hit/miss smoke harness before any playing-strength claim.

## Protocol semantics from external sources

The UCI command model is a **command pair**: a GUI sends `go`, and the engine eventually answers with `bestmove`. Official Stockfish documentation describes `go ponder` as the search mode used while thinking during the opponent's time, and documents `ponderhit` as the signal that the expected opponent move was played. After `ponderhit`, the engine should continue searching but switch from pondering to normal search [1]. The Python-chess UCI API states the same contract and treats `ponderhit` as valid only while the engine is currently searching in ponder mode [2].

The expected move is carried in the position/search context: official Stockfish's example starts from `position startpos moves e2e4 e7e5 g1f3`, then sends `go ponder`, so the engine searches the position after the expected move (`g1f3`) and continues if that move really occurred [1]. If the opponent plays a different move, the GUI must stop the speculative search and send a new `position` followed by a normal `go`; the engine must not apply a stale speculative tree to an unrelated position.

Pondering is commonly described as searching likely opponent continuations during the opponent's move time. Its possible benefit is transposition-table and iterative-deepening preparation, or immediately reusing a sufficiently deep result after a hit [3]. The same source notes that a miss requires unmaking/restarting the predicted line, and gives roughly 50% prediction accuracy as historical context—not a guarantee for this engine or a basis for an Elo claim.

UCI does not prescribe the internal time-allocation formula after a hit. The protocol establishes the state transition; the engine must define how much of its own clock is available, how much speculative work is retained, and when to move after a hit. That policy must be deterministic, documented, and tested against the existing urgency tiers and reserves.

## Repository verification

### Source scan

The requested source scan was run with:

```text
cd /home/ubuntu/Unchessed-UCI-Engine
rg -n -i 'ponder|ponderhit' unchessed-core/src Cargo.toml README.md docs || true
```

Result: **no pondering/ponderhit matches** in the searched engine source, manifest, README, or documentation. The supplied project context independently identifies this gap, specifically stating that no `ponder`/`go ponder`/`ponderhit` handling exists in `uci.rs` or `search.rs` [4].

The current parser is visible at [`unchessed-core/src/uci.rs`](../../unchessed-core/src/uci.rs), around lines 958–985. `parse_go` recognizes `depth`, `movetime`, `wtime`, `btime`, `winc`, `binc`, `movestogo`, `nodes`, and `infinite`; its wildcard branch ignores everything else. Therefore `ponder` is not represented in `Limits` and cannot affect search mode. The top-level UCI loop around lines 204–480 handles `uci`, `isready`, `position`, `go`, `stop`, `quit`, and options, but has no `ponderhit` branch. Existing `stop` joins the worker; `ponderhit` is not a transition signal.

The current search timing seam is [`unchessed-core/src/search.rs`](../../unchessed-core/src/search.rs), around lines 86–175. `Limits::budget` returns unlimited time for `infinite`; for `movetime` it subtracts a fixed 25 ms reserve with a 5 ms floor; for clock searches it allocates soft/hard budgets from remaining time, increment, moves-to-go, low-clock tiers, panic mode, and a final 60 ms ceiling. The worker path passes one immutable `Limits` snapshot to the search and polls an atomic stop flag. This is a sound seam for a future normal-search deadline, but not by itself for a mid-search ponder-to-normal transition.

The Lazy SMP code around `uci.rs` lines 1651–1690 starts helper threads that share the same `job.limits` and stop flag. A future ponder transition must therefore update a shared, synchronized deadline/mode or stop and restart the complete search; mutating an ordinary copied `Limits` value would not affect already-running helpers.

### Actual UCI smoke checks

A real existing release binary was used; no source was changed. Binary metadata:

```text
ls -lh target/release/unchessed-adapter
-rwxr-xr-x ... 1.1M ... target/release/unchessed-adapter
file target/release/unchessed-adapter
ELF 64-bit LSB pie executable, x86-64, ... not stripped
```

Baseline protocol/search command:

```text
(printf 'uci\nisready\nposition startpos\ngo depth 2\n' ; sleep 2) | timeout 10s ./target/release/unchessed-adapter 2>&1
```

Observed relevant output (full run also advertised all ordinary options and ended successfully):

```text
id name Unchessed Game Adapter 0.2.3
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove d2d4
```

Ponder parser/hit probe:

```text
{ printf 'uci\n'; sleep 0.2; printf 'isready\n'; sleep 0.2; printf 'position startpos moves e2e4\n'; sleep 0.2; printf 'go ponder depth 2\n'; sleep 0.8; printf 'ponderhit\n'; sleep 0.8; printf 'stop\n'; sleep 0.5; } | timeout 8s ./target/release/unchessed-adapter 2>&1
```

Observed result:

```text
... ordinary option advertisement; no option name Ponder ...
uciok
... readyok
info string [Unchessed] book: Scandinavian, Main Line (B01) [main] — opponent ~1500, playing the popular stuff
bestmove d7d5
```

There was **no ponder state, no `info`/state transition on `ponderhit`, no `ponder` move in the output, and no error**. Because the request used `depth 2`, the ignored `ponder` token did not prevent the ordinary depth-limited search. This verifies the negative behavior, not support.

An active probe was also run:

```text
{ printf 'position startpos\n'; sleep 0.1; printf 'go ponder infinite\n'; sleep 0.7; printf 'ponderhit\n'; sleep 0.2; printf 'stop\n'; sleep 0.5; } | timeout 6s ./target/release/unchessed-adapter 2>&1
```

It produced normal iterative `info depth ...` lines followed by:

```text
bestmove e2e4
```

The search continued until its normal command processing/stop path; there was no observable ponder-to-normal transition. This is a real-world smoke check of the existing binary and a verified negative result.

### Build/test limitation

The mandatory feasible local check was performed using the existing release binary. A fresh workspace build was also attempted:

```text
cd /home/ubuntu/Unchessed-UCI-Engine && cargo build --workspace
```

Exact failure:

```text
cargo 1.75.0
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`
```

Thus no fresh Cargo test result is claimed. The pre-existing release executable was sufficient for the actual protocol smoke check, but it cannot validate hypothetical implementation behavior.

## Safe design, if implementation is later approved

### State and option

Add the standard-compatible option only as an opt-in control:

```text
option name Ponder type check default false
```

The option means “the GUI permits the engine to think during the opponent's time”; it does not itself start a search. Parse it case-insensitively and snapshot it into each job. With the default `false`, `go` and all existing timing behavior remain byte-for-byte compatible except for the new advertisement line. Do not use a nonstandard `go` option as a substitute for the stateful check option.

### Ponder start

When `Ponder` is true and the GUI sends `go ponder`, parse and retain a `pondering` mode in the worker request. Require a usable position and search request; preserve all ordinary depth/node/movetime/clock limits as explicit metadata, but do not spend the engine's own move budget as if the engine were already choosing its move. The job should record the expected root position and the expected opponent move/continuation represented by the GUI's `position` command. The engine should eventually provide `bestmove <move> ponder <expected-reply>` only after a normal stop/hit transition; while still pondering, it must not prematurely terminate with `bestmove` merely because a normal soft limit was reached.

A conservative first version should **reuse TT and completed iterative results but not claim correctness from arbitrary stale PV data**. A hit is valid only when the GUI's current position exactly corresponds to the expected ponder continuation. Root move legality, side to move, castling rights, en-passant square, halfmove/fullmove state where relevant, and the expected move must be checked. A miss should stop the job and require the normal `position`/`go` cycle.

### `ponderhit` transition and clocks

`ponderhit` must be accepted only while a ponder job is active; otherwise return a harmless protocol diagnostic or ignore it according to the project's chosen convention, but never create a search or alter unrelated state. On a valid hit:

1. Atomically mark the job normal rather than pondering.
2. Establish the **normal-search start/deadline at the hit**, using the current side's reported clock and increment from the most recent valid game state. Ponder wall time must not be charged against the side's own thinking budget; the opponent was moving during that interval.
3. Continue the same search only if its root matches the expected line. Otherwise stop and restart from the actual position.
4. Preserve the best completed line and TT work, but re-evaluate soft/hard stopping under the normal budget. Never let a ponder `infinite` or depth-only request bypass a normal post-hit clock deadline.
5. Ensure all Lazy SMP helpers observe the transition through an atomic mode/deadline object, or stop/join and restart them. Do not mutate a copied `Limits` behind their backs.

The recommended first implementation policy is **restart-on-miss, continue-on-hit**, with a monotonic deadline computed at hit time and a minimum safety reserve. Do not subtract ponder elapsed time from the engine clock, and do not simply reuse the original `go ponder movetime N` as N milliseconds after hit unless that is explicitly documented; doing so can overthink and flag in a real clock game. For explicit `movetime`, document whether N is total normal thinking time after hit or a maximum search budget; retain current semantics when `Ponder` is false.

### Interaction with current time management

The existing `Limits::budget` has low-clock and panic tiers, situation scaling, and fixed reserves. Pondering must not add an independent multiplier or silently bypass them. The hit path should feed the current clock/increment through the same centralized budget calculation, then apply only a narrowly defined “already searched” policy. In particular:

* `go infinite` remains unlimited in ordinary analysis; `Ponder` should not convert analysis into a clocked search.
* `go ponder` with clocks must transition to the same low-clock/panic rules as a normal `go` at hit time.
* `movetime`, node, and depth limits must have explicit precedence when combined, following the engine's existing “any limit can stop” behavior.
* Existing preprocessing and root-hint elapsed-time charging must not accidentally charge speculative opponent-time work to the post-hit budget unless the design explicitly chooses that conservative policy.
* `Move Overhead` is not yet implemented in this branch. If it is later added, it should reserve transport/scheduling overhead from the **post-hit normal budget**, not from opponent pondering time, and must not create an unsigned underflow at low clocks.
* `stop`, `quit`, a new `position`, and a new `go` must synchronously invalidate/join the ponder worker before replacing state, as existing command lifecycle code does for ordinary searches.

## Real test plan before implementation or promotion

| Stage | Concrete test | Acceptance criterion |
|---|---|---|
| Parser | `uci`, `setoption name Ponder true/false`, case variants, `isready` | Exactly one check option; state changes are acknowledged without racing a worker. |
| Ordinary regression | `Ponder=false`; run existing depth, node, movetime, clock, infinite, and `stop` commands | Output, limits, and timing behavior match the pre-feature binary; no default change. |
| Ponder start | `setoption name Ponder true`; `position ...`; `go ponder` | No premature `bestmove`; iterative info may continue; worker remains cancellable. |
| Hit | Start ponder on a position with a known expected reply, send `ponderhit` while active | Search continues, changes to normal deadline policy, and emits exactly one legal `bestmove`/optional ponder move. |
| Miss | Start ponder, then send `stop`, a different `position`, and normal `go` | No stale PV/root is used; exactly one result for the new position. |
| Invalid hit | Send `ponderhit` with no active ponder, or after a miss/stop | No panic, deadlock, duplicate bestmove, or accidental new search. |
| Clock algebra | Repeat hit tests at ample time, below 20 s, below 6 s, below 2 s, with and without increment | Post-hit soft/hard deadlines follow existing urgency/panic rules; ponder elapsed time is not charged to own time. |
| Limit precedence | Combine ponder with `movetime`, depth, nodes, and clocks | Documented earliest-limit behavior; no indefinite search after hit. |
| SMP lifecycle | Run with `Threads=1`, 2, and default; hit/miss/stop repeatedly | No helper-thread leak, duplicate output, data race, or missed stop. |
| Position identity | Vary side-to-move, castling, en-passant, and expected move | Only exact expected continuation gets a hit; all mismatches restart safely. |
| GUI-like integration | Use a small local script or python-chess asynchronous driver against the built binary, recording timestamps and all stdout/stderr | Reproduce several hit and miss cycles with raw logs retained. |
| Strength gate | Only after correctness passes, compare Ponder off/on in a frozen, paired real-game harness | Separate match evidence; no default or Elo claim from smoke tests. |

The first implementation should add deterministic unit tests around a pure state/deadline transition helper, then the live harness. Wall-clock assertions should use generous bounds and verify ordering/state rather than exact milliseconds; exact budget algebra belongs in pure tests. Because the current Cargo toolchain cannot parse the lockfile, install/use a compatible toolchain only if available within Tier 1 and record its version; otherwise the blocker must remain explicit rather than silently treating the old release binary as a fresh test.

## Verified versus assumed

| Claim | Status |
|---|---|
| No `ponder`/`ponderhit` implementation or source matches in the searched repository paths | **Verified** by `rg` and source inspection. |
| `parse_go` ignores `ponder` | **Verified** at `uci.rs` lines 958–985. |
| `ponderhit` is not dispatched | **Verified** from the UCI command loop and live probe. |
| Release binary has no advertised `Ponder` option | **Verified** from captured `uci` output. |
| Existing release binary returns ordinary `bestmove` after ignored ponder commands | **Verified** by two real smoke probes. |
| Current budget has soft/hard, urgency, panic, and reserve logic | **Verified** from `search.rs` lines 133–175. |
| Pondering can improve practical strength by pre-search/TT reuse | **Protocol/background rationale**, not measured for Unchessed. |
| Around 50% prediction accuracy | **Historical external context**, not an engine-specific measurement. |
| Continue-on-hit is faster/stronger than restart | **Assumed design hypothesis** requiring a later controlled test. |
| Pondering's net Elo value for this engine | **Unknown; no game evidence collected.** |
| Fresh workspace build/tests pass | **Not verified; blocked by Cargo 1.75.0 versus lockfile v4.** |

## Recommendation

**Defer implementation, but retain as a well-scoped protocol candidate.** The feature is worthwhile only for a concrete GUI/game-host interoperability need or after a bounded implementation can be tested with hit/miss and clock-boundary coverage. If approved later, prioritize correctness and lifecycle over speculative search optimization: default-off `Ponder`, explicit worker state, exact expected-position validation, atomic hit transition, restart-on-miss, and post-hit reuse subject to the existing budget function. Do not change defaults, claim Elo, or begin a match until the real GUI-like smoke harness and clock tests pass. The present branch should remain unchanged.

## References

[1] [Stockfish official documentation — UCI Protocol and Stockfish Commands](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html), sections on `go ponder`, `ponderhit`, `bestmove`, and examples.

[2] [python-chess UCI documentation](https://python-chess.readthedocs.io/en/v0.17.0/uci.html), `ponderhit`: expected move has been played; continue searching and switch to normal search.

[3] [Chess Programming Wiki — Pondering](https://www.chessprogramming.org/Pondering), purpose, predicted-move/hit/miss approaches, TT preparation, and historical context.

[4] Repository context: [`/home/ubuntu/upload/pasted_content_7.txt`](file:///home/ubuntu/upload/pasted_content_7.txt), item 25 and mandatory real-world testing rule.

[5] Repository source: [`unchessed-core/src/uci.rs`](../../unchessed-core/src/uci.rs) and [`unchessed-core/src/search.rs`](../../unchessed-core/src/search.rs).

**Report status:** complete; no implementation or default change.
