# 40 — TT memory auto-sizing

**Investigation ID:** `tier1-tt-memory-sizing`
**Tier:** 1 (research and source/runtime verification only)
**Repository:** `Unchessed-UCI-Engine`
**Branch checked:** `manus/research-facilities`
**Scope:** Verify the fixed Hash default and TT lifecycle, run a real UCI smoke, and assess safe memory detection and sizing.
**Status:** Research only. No implementation, default change, match, cloud spend, commit, or push was performed.

## Executive conclusion

The item is correctly described: the engine has a fixed **128 MB** default and requires the GUI/user to send `setoption name Hash value N` to choose another size. The option is advertised as `spin default 128 min 1 max 2048`. At startup, `Options::default()` sets `hash_mb: 128`, and `run()` constructs one shared `TT` from that value. A `Hash` change synchronously replaces the table; `Clear Hash` clears it; `ucinewgame` joins the worker and clears the complete table before resetting game state. The table is therefore intentionally process-persistent between searches in a game, but not across a correctly signalled game boundary.

A real local UCI run confirmed the option exchange, `Hash=1` acceptance, `isready`/`readyok`, a legal search producing `bestmove f2f4`, `Clear Hash`, and `ucinewgame` readiness. The source-level and smoke evidence show no automatic memory detection. The workspace test command was attempted but was blocked before compilation by the checked-in Cargo.lock v4 requiring a newer/nightly Cargo feature (`-Znext-lockfile-bump`); this is a negative result, not a test pass.

**Recommendation:** keep the shipped 128 MB default unchanged for now, but consider a separate, default-preserving Tier 2 implementation plan for conservative startup sizing only after defining platform and container semantics. If pursued, detect an effective memory ceiling (cgroup/job/container limit where available, otherwise platform available memory), reserve a conservative fraction, clamp to the existing 1–2048 MB range, and retain 128 MB as the fallback. Do not size from `MemFree`, swap, or unconstrained host `MemTotal`; do not silently allocate several gigabytes on a constrained container or shared machine. Any change to the default must have allocation-failure/latency tests and a real paired-game evaluation before promotion.

## Verified source findings

### Fixed default and UCI declaration

The following commands were run:

```text
cd /home/ubuntu/Unchessed-UCI-Engine
nl -ba unchessed-core/src/uci.rs | sed -n '95,110p;204,216p;253,266p;337,365p;570,587p'
```

Relevant output:

```text
98  impl Default for Options {
99      fn default() -> Self {
100         Options {
101             hash_mb: 128,
...
204 pub fn run(ident: EngineIdent) {
205     let stdin = std::io::stdin();
206     let mut opt = Options::default();
...
212     let tt = Arc::new(Mutex::new(TT::new(opt.hash_mb)));
...
259                 println!("option name Hash type spin default 128 min 1 max 2048");
...
337             "isready" => println!("readyok"),
...
353             "ucinewgame" => {
354                 join_worker(&mut worker, &stop);
355                 tt.lock().unwrap().clear();
...
578         "hash" => {
579             if let Ok(mb) = value.parse::<usize>() {
580                 opt.hash_mb = mb.clamp(1, 2048);
581                 tt.lock().unwrap().resize(opt.hash_mb);
...
584         "clear hash" => tt.lock().unwrap().clear(),
```

**Verified:** the default is fixed at 128 MB, the advertised range is 1–2048 MB, and no startup memory query is present in this path. `Hash` is user/GUI-controlled after startup.

### Allocation and lifecycle

The following command was run:

```text
cd /home/ubuntu/Unchessed-UCI-Engine
nl -ba unchessed-core/src/tt.rs | sed -n '45,90p'
```

Output:

```text
48  struct Slot {
49      xor_key: AtomicU64,
50      data: AtomicU64,
51  }
53  pub struct TT {
54      table: Vec<Slot>,
55      mask: usize,
58  impl TT {
59      pub fn new(mb: usize) -> TT {
60          let bytes = mb.max(1) * 1024 * 1024;
61          let mut n = bytes / std::mem::size_of::<Slot>();
62          if !n.is_power_of_two() {
63              n = n.next_power_of_two() >> 1;
64          }
65          n = n.max(1024);
66          let mut table = Vec::with_capacity(n);
67          table.resize_with(n, || Slot {
68              xor_key: AtomicU64::new(0),
69              data: AtomicU64::new(0),
70          });
71          TT { table, mask: n - 1 }
74      pub fn resize(&mut self, mb: usize) {
75          *self = TT::new(mb);
78      /// Safe to call while other threads hold `&TT` for probe/store ...
80      pub fn clear(&self) {
81          for slot in &self.table {
82              slot.xor_key.store(0, Ordering::Relaxed);
83              slot.data.store(0, Ordering::Relaxed);
```

Each slot is 16 bytes (two `AtomicU64`s). The requested MB is converted using binary MiB; the number of slots is rounded down to a power of two (with a minimum of 1024), so the actual allocation is approximately at most the requested size, subject to allocator overhead. `resize` constructs a new zeroed table and drops the old one. `clear` zeroes both atomics in the existing table. The TT is wrapped in `Arc<Mutex<TT>>` for lifecycle mutations, while searches share `&TT`.

The existing TT-aging report (`docs/reinforcement/29-tt-aging.md`) independently records that `ucinewgame` clears the complete table and that `position`/`go` do not implicitly clear it. That remains consistent with the lines above. Auto-sizing is orthogonal to generation aging: it changes initial capacity, not replacement policy or game-boundary semantics.

## Real-world UCI smoke

The exact command was:

```text
cd /home/ubuntu/Unchessed-UCI-Engine
printf 'uci\nisready\nsetoption name Hash value 1\nisready\nposition startpos moves e2e4 e7e5\ngo depth 2\nisready\nsetoption name Clear Hash\nisready\nucinewgame\nisready\nquit\n' | target/release/unchessed-adapter 2>&1
```

Exact salient output (the engine also printed its full option list):

```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
readyok
readyok
info string [Unchessed] book: King's Gambit Accepted (C39) [main] — opponent ~1500, playing the popular stuff
bestmove f2f4
readyok
readyok
```

**Verified by this run:** UCI option declaration succeeds; `Hash=1` is accepted without a protocol error; readiness succeeds before and after the option and lifecycle commands; a real start-position search reaches `bestmove`; `Clear Hash` and `ucinewgame` return to a ready state. **Not verified by this black-box transcript:** exact allocated bytes, because the engine emits no allocation diagnostic or hashfull line in this short search. Source inspection, rather than output, verifies resize/clear behavior.

## Test and environment caveats

The requested workspace test was attempted:

```text
cd /home/ubuntu/Unchessed-UCI-Engine
cargo test --workspace --lib
```

It failed before compilation:

```text
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`
```

No lockfile, toolchain, source, or default was changed to work around this blocker. The real UCI smoke used the existing `target/release/unchessed-adapter` binary, so it is a runtime smoke of the checked-out artifact, not a newly rebuilt test binary.

The host memory observation was also recorded:

```text
free -h
               total        used        free      shared  buff/cache   available
Mem:           5.8Gi       1.3Gi       4.0Gi        10Mi       1.1Gi       4.6Gi
Swap:          2.0Gi          0B       2.0Gi        0B

grep -E '^(MemTotal|MemAvailable|MemFree|SwapTotal|SwapFree):' /proc/meminfo
MemTotal:        6129996 kB
MemFree:         4217700 kB
MemAvailable:    4813160 kB
SwapTotal:       2097148 kB
SwapFree:        2097148 kB
```

The checked cgroup v2 files did not produce values in this shell (they were absent or unreadable), so no container memory ceiling could be verified here. That absence is important: host `MemTotal`/`MemAvailable` cannot safely be assumed to be the process's effective limit in every deployment.

## Safe detection research

The [Linux `proc_meminfo(5)` manual](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html) defines `MemTotal` as usable physical RAM and `MemAvailable` (since Linux 3.14) as an estimate of memory available for starting new applications without swapping. This makes `MemAvailable` materially safer than `MemFree` for a Linux fallback, but it is still a system-wide estimate and does not by itself account for a process-specific cgroup/job limit.

The [Rust `sysinfo::System` documentation](https://docs.rs/sysinfo/latest/sysinfo/struct.System.html#method.available_memory) states that `available_memory()` returns available RAM in bytes and distinguishes it from `free_memory()`; the documentation also requires a memory refresh (`refresh_memory` or an appropriate memory refresh kind) before relying on current values. `sysinfo` is a practical cross-platform abstraction, but its documented portability caveat is significant: Windows and FreeBSD do not report “available” memory in the same way, and the API falls back to free memory there. A portable implementation must therefore define conservative fallback behavior rather than treating all platforms as equivalent.

The [UCI protocol reference](https://backscattering.de/chess/uci/) defines `Hash` as a spin option whose value is MB for hash-table memory and says it should be supported by engines. It also advises engines to use a very small hash first because a GUI can set the option during boot and then synchronize with `isready`. The current engine's 128 MB default is valid protocol behavior, though it is not the reference's smallest-default recommendation. Changing the advertised default is a user-visible behavior change and should not be smuggled into a memory-detection patch.

For comparison, the [Stockfish terminology documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html) describes Hash as the RAM used to store positions in the game tree. Stockfish is useful as an engineering comparison, not as evidence that its exact sizing policy transfers to this engine or to arbitrary GUI/container environments.

## Design assessment

A safe automatic default should be defined as a bounded policy, not as “allocate a fixed percentage of whatever RAM the host reports.” The preferred decision order is:

1. **Effective process/container ceiling:** on Linux, inspect a usable cgroup v2 `memory.max` (and relevant v1 equivalents where supported), treating `max` as unlimited; on Windows use the job/container limit when available; on macOS and other platforms use the best documented process/system API available.
2. **Available headroom:** use a refreshed available-memory estimate where the platform provides one. Do not include swap as TT budget.
3. **Conservative fraction and reserve:** subtract a fixed safety reserve for the engine, NNUE/model, threads, stacks, allocator overhead, and the operating system, then take only a conservative fraction of the remaining headroom. The exact fraction is a policy choice requiring measurement; it should be specified before implementation rather than copied from another engine.
4. **Hard bounds:** clamp to the existing 1–2048 MB option range and impose a low-memory floor/fallback. Preserve 128 MB if detection fails, is inconsistent, or reports too little confidence.
5. **Observability:** emit an `info string` only if protocol/UI compatibility permits, recording source (`cgroup`, platform available memory, or fallback), detected value, and selected MB. Do not expose raw sensitive system details unnecessarily.

Potential approaches have different trade-offs:

| Approach | Benefit | Main failure mode | Assessment |
|---|---|---|---|
| Fixed 128 MB (current) | Predictable, low risk, reproducible | Underuses large machines; user must configure | Keep as current shipped baseline |
| Fraction of host `MemTotal` | Simple and stable | Over-allocates in containers/shared hosts; ignores current pressure | Unsafe as sole signal |
| Fraction of `MemAvailable` | Tracks current headroom; Linux semantics are better | System-wide value can still exceed cgroup/process budget; fluctuates | Useful fallback, not sole authority |
| Fraction of effective cgroup limit | Respects container ceiling | Limit may be absent/unlimited; does not indicate current pressure | Preferred ceiling when available |
| Cross-platform crate/API plus bounds | Broad coverage | Platform semantics and crate/version maintenance; fallback complexity | Plausible implementation path, needs explicit tests |
| Auto-size plus preserve user `Hash` | QoL without overriding explicit choices | Must distinguish startup default from explicit setoption | Required behavioral contract |

The most defensible user contract is: auto-sizing applies only when no explicit `Hash` has been received; an explicit `Hash` always wins and retains the existing clamp and synchronous resize semantics. Whether to advertise the computed value as the UCI `default` is a separate question: UCI GUIs may use it before sending settings, so a dynamic declaration can reduce reproducibility and complicate logs. Keeping the declaration at 128 while internally auto-sizing would be misleading; changing it dynamically could surprise GUIs. This argues for a carefully designed option/default protocol decision, not a one-line memory query.

## Recommendation and gates

**Recommendation: defer implementation and preserve 128 MB now; pursue only a bounded design/diagnostic if the quality-of-life benefit is important.** The source gap is real, but no evidence here shows that 128 MB causes a strength, latency, or reliability problem on supported deployments. The local machine had about 4.6 GiB available, yet the smoke intentionally used `Hash=1`; no claim about the best hash size or Elo can be inferred.

Before any implementation, specify platform support, cgroup handling, fallback, reserve/fraction, integer rounding, startup observability, and the exact meaning of explicit versus implicit Hash. Then test at low memory, at the existing 128 MB baseline, and on a large-memory host. Test repeated `Hash` resize, `Clear Hash`, `ucinewgame`, search completion, and allocation failure behavior. Because TT size can alter move ordering and search results, a candidate that changes the effective default requires fixed-position node/hashfull measurements followed by a real paired-game test with frozen binary, model, options, hardware, and time control. No such strength evidence was collected in this Tier 1 item.

## Verified versus assumed

| Claim | Status |
|---|---|
| `Options::default()` uses 128 MB | **Verified**, `unchessed-core/src/uci.rs:98–102` |
| UCI advertises `Hash` default 128, range 1–2048 | **Verified**, `uci.rs:259` |
| Startup constructs one shared TT from `opt.hash_mb` | **Verified**, `uci.rs:204–212` |
| TT uses two atomic words per slot and binary-MB sizing | **Verified**, `tt.rs:48–71` |
| Explicit Hash clamps and replaces the table | **Verified**, `uci.rs:578–582`, `tt.rs:74–76` |
| `Clear Hash` clears entries | **Verified**, `uci.rs:584`, `tt.rs:78–85` |
| `ucinewgame` joins worker and clears TT before game reset | **Verified**, `uci.rs:353–364` |
| Current code auto-detects system memory | **Negative verified result**: no such path found in the inspected option/startup code; fixed 128 is used |
| Hash resize allocates exactly the requested byte count | **Assumed/incorrect as an exact claim**: power-of-two slot rounding and allocator overhead mean approximate allocation |
| Host `MemAvailable` equals this process's safe budget | **Not verified and unsafe to assume**; cgroup values were unavailable in this shell |
| Auto-sizing improves playing strength | **Not tested; no claim** |
| Workspace tests pass | **Negative result**: Cargo.lock v4 parsing blocked before compilation |
| Real UCI option/readiness/search path works in existing binary | **Verified** by the transcript above |

## Decision record

No Tier 2 or Tier 3 work was started. No default, source, lockfile, binary, or deployment behavior was changed. The appropriate next action is a scoped design review and, only with approval, a small cross-platform detection prototype with explicit fallback and memory-pressure tests—not an immediate default change or match campaign.

## References

1. [Linux `proc_meminfo(5)` — `MemAvailable`](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html).
2. [Rust `sysinfo::System` — memory APIs](https://docs.rs/sysinfo/latest/sysinfo/struct.System.html#method.available_memory).
3. [UCI protocol reference — `Hash` option](https://backscattering.de/chess/uci/).
4. [Stockfish Wiki — terminology (`Hash`)](https://official-stockfish.github.io/docs/stockfish-wiki/Terminology.html).
5. [`docs/reinforcement/29-tt-aging.md`](29-tt-aging.md) — previously verified TT lifecycle and game-boundary behavior.

---

**Bottom line:** the fixed 128 MB default and lifecycle are verified; automatic sizing is absent; a real UCI smoke passed; the workspace test path was blocked by the local Cargo toolchain/lockfile mismatch; safe auto-sizing is plausible only with effective-limit detection, conservative bounds, explicit fallback, and later real-game validation.
