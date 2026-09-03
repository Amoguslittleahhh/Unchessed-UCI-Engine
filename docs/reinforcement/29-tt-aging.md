# 29 — Transposition-table generation and aging

**Investigation ID:** `tier1-tt-generation-aging`
**Tier:** 1 (protocol completeness / infrastructure)
**Repository:** `Unchessed-UCI-Engine`, branch `manus/research-facilities`, commit `667e9068c26b8201e81435d4a00c2d4a5ec9fd13`
**Status:** Research and design only. No implementation, default change, match, or SPRT was performed.

## Executive conclusion

The engine has a lock-free, single-slot-per-index transposition table (TT) with **depth-preferred replacement but no generation, age, or search-epoch field**. A colliding entry from an old root search is therefore protected whenever it is deeper than the incoming non-exact entry. This is the exact gap named by the master brief: replacement is depth-aware, but not freshness-aware.

In the current UCI lifecycle, however, `ucinewgame` performs a complete TT clear. The command handler joins any worker, calls `TT::clear()`, resets the opponent model and persona, resets clocks and game identity, and creates a new `Game`. Thus, when a GUI sends the command, old-game TT entries cannot survive into the next game. The current code does **not** need generation aging to compensate for a correctly delivered game boundary; it needs generation aging primarily because UCI explicitly permits clients that do not send `ucinewgame`, and because a clear is a blunt operation that discards useful within-process cache state.

The repository's actual match/deployment scripts invoke `cutechess-cli -each proto=uci`; this is strong evidence that the normal test harness uses a UCI game lifecycle, but this checkout does not capture or assert the engine's received command stream. I therefore cannot claim that every deployment, GUI, wrapper, or remote runner reliably sends `ucinewgame`. The UCI specification itself says an engine must not rely on the command: if no `ucinewgame` precedes the first `position`, the GUI may be one that does not support it, and the engine should not expect later notifications [1].

**Recommendation: defer implementation as a strength project, but add a cheap diagnostic before reconsidering it.** First measure the actual boundary signal in each supported deployment with a transparent UCI wrapper that logs inbound `ucinewgame`, `position`, `go`, `isready`, and process identity, while preserving byte-for-byte forwarding. If the supported production path is confirmed to send `ucinewgame` before every new game, generation aging is unlikely to be a high-priority Elo opportunity. If missing boundaries are observed, first fix the harness/deployment contract or make a narrowly scoped lifecycle diagnostic; do not silently assume that generation aging can infer a new game from arbitrary `position` commands. Any future replacement change remains subject to fixed-position correctness checks and the repository's real paired-game SPRT gate.

## Question and decision boundary

The master brief marks TT generation/aging as absent beyond depth-preferred replacement and asks whether this is a practical issue given actual deployment, specifically whether the harness reliably sends `ucinewgame` [2]. This report answers four narrower questions:

1. What replacement policy and TT lifecycle are actually implemented?
2. What does generation aging add relative to the current clear-on-new-game behavior?
3. Do the checked-in deployment and test harnesses establish reliable `ucinewgame` delivery?
4. What is the cheapest useful diagnostic, without implementing the feature?

No playing-strength claim is made. A cache policy can change node counts, move ordering, and results indirectly; unit tests or a protocol transcript cannot establish Elo.

## Repository inspection

### TT representation and replacement

`unchessed-core/src/tt.rs` defines each TT slot as two `AtomicU64` words: packed move/score/depth/bound data and `xor_key = key ^ data`. A probe reconstructs the key and treats a mismatch as a miss, which is appropriate for the lock-free Lazy SMP design. The table has one slot per indexed hash position; it is not a multi-entry cluster or bucket.

The relevant store policy is explicit at `tt.rs:161–193`:

* a different key claims the slot;
* an exact-bound entry replaces an existing same-key entry;
* a same-key non-exact entry is retained when the incoming depth is not greater;
* an old colliding entry is otherwise replaced only because it has a different key, not because it is old;
* depth is clamped to 127 and scores to `i16`.

There is no generation counter in `Entry`, no current-generation state in `TT`, no `new_search()` method, and no age term in the replacement decision. `hashfull()` reports occupancy of a 1,000-slot sample, but it has no age filter. The comments correctly describe this as **depth-preferred replacement** and accept rare concurrent replacement races as a cache-quality cost rather than a correctness failure.

`TT::clear()` writes zero to both atomics for every slot. `TT::resize()` constructs a new table, also clearing it. These are the only repository mechanisms that remove stale entries globally.

### Search integration

`search.rs` probes the TT at the normal negamax node and stores search results through `TT::store()`. The search API receives `&TT`; it does not advance a generation at root-search entry. `go_with_root_hints()` can be called repeatedly for one `go`, including helper searches and adapter side-searches, but none of those calls starts a generation because no generation API exists. Consequently, the table is deliberately persistent across root searches within a game, and all entries compete by key/depth/bound only.

This persistence is not automatically a bug. Reusing entries from earlier root searches is standard TT behavior, and the Chessprogramming reference notes that current engines generally profit from previous searches rather than clearing at every root [3]. The trade-off is that entries from positions no longer reachable from the current root can remain protected by their depth.

### UCI new-game lifecycle

`unchessed-core/src/uci.rs:353–378` handles `ucinewgame` as follows:

| Action | Verified behavior |
|---|---|
| Stop prior search | Joins the worker before mutating shared state. |
| TT reset | `tt.lock().unwrap().clear()` clears every slot. |
| Adapter state reset | Reinitializes `OpponentModel`, including the experimental detector flag. |
| Persona reset | Replaces `PersonaState` with its default. |
| Clock/game bookkeeping | Clears the opponent-clock state, increments game ID, and creates a fresh `Game`. |
| Optional model state | Reloads the experimental Unarchitectured candidate when enabled. |
| Readiness | `isready` returns `readyok`; the handler itself does not asynchronously acknowledge the clear. |

`setoption name Clear Hash` also calls `TT::clear()`, and changing `Hash` resizes the table. `position` does not clear the TT. A process that receives a new game's `position` without `ucinewgame` therefore keeps both TT entries and adapter state from the previous game. This matters more for the adapter than for the full-strength reviewer because opponent-model and persona state are also intentionally reset only by `ucinewgame`.

`run_go()` holds the TT for the duration of the worker and uses the same shared table for the main search and Lazy SMP helpers. There is no implicit clear or generation transition at `go`, `position`, or root search. The engine is permissive about a first position before any `ucinewgame`, consistent with the UCI specification's warning that some clients never send the command.

## Literature and design comparison

The Chessprogramming Wiki describes replacement as necessary because a finite table fills quickly. It identifies the competing values of deep entries and recently searched/leaf-near entries, and lists depth-preferred, always-replace, bucket, and mixed schemes [3]. Its aging section says that modern engines commonly retain useful entries across root searches but age old entries so that entries no longer likely to occur from the current root eventually lose replacement priority [3]. This supports the diagnosis that generation aging is a **cache policy optimization**, not a correctness requirement when keys and bounds are validated.

Stockfish provides a useful contemporary comparison, not a promise that its exact policy transfers to Unchessed. Its public `src/tt.cpp` packs a 5-bit generation into each entry and increments the current generation in `TranspositionTable::new_search()` at every root search. Replacement evaluates depth minus an age penalty (`relative_age`), and `save()` allows an older entry to be overwritten even when its depth would otherwise protect it [4]. Stockfish also uses three-entry clusters, whereas Unchessed currently uses one slot, so the collision and replacement dynamics are materially different. Stockfish's design demonstrates a practical mechanism; it does not establish that adding it here would yield positive Elo.

The UCI protocol reference is especially important for deployment. It defines `ucinewgame` as a notification that the next `position`/`go` search belongs to a different game, and asks the GUI to send `isready` afterward if the operation may take time. Crucially, it says that if the GUI did not send `ucinewgame` before the first `position`, the engine should not expect subsequent `ucinewgame` commands because the GUI may not support the command [1]. Therefore a compliant engine must tolerate absent boundaries; it may use a persistent TT, but it cannot treat notification delivery as universal.

## Deployment and harness assessment

The checked-in scripts are UCI-driven. For example, `wsl-workspace/scripts/run_internal_ladder_v2.sh:11–20` launches two engine processes through `cutechess-cli`, specifies `-each proto=uci`, repeats rounds, and runs concurrent games. `scripts/sprt-history/sprt_rook.sh:13–24` does the same. Other repository scripts use the same `proto=uci` pattern. The adapter's own diagnostic tool `tools/unarchitectured_v1_depth_time_calibration.py` explicitly sends `ucinewgame`, then `position`, and later `go`.

This establishes that the intended harness protocol is UCI and that at least one in-repository direct client sends `ucinewgame`. It does **not** establish, from source inspection alone, that:

* the installed `cutechess-cli` version sends the command between every repeated game;
* every external GUI or server wrapper preserves it;
* a crashed/restarted or pooled engine process receives the same lifecycle;
* a client that starts with `position` and never sends `ucinewgame` is rejected or detected;
* concurrent cutechess games map one process to one game boundary in the way an operator assumes.

No cutechess binary is available at the paths hard-coded by the WSL scripts in this sandbox, and no protocol trace from a real match was available. I therefore classify delivery as **likely in the intended cutechess path, but unverified**, rather than reliable. The UCI specification explicitly makes that distinction necessary.

A further practical point is that `ucinewgame` currently clears the entire table. If the harness sends it correctly, the generation feature would replace a known full clear with an age transition only if the lifecycle code were deliberately redesigned. That could preserve useful entries across games, but it would also preserve entries belonging to arbitrary prior roots and would interact with the adapter's separate state reset. It should not be introduced merely because generation aging is standard in a stronger engine.

## Cheapest practical diagnostic (design only)

The recommended first diagnostic is a **transparent UCI logging wrapper**, not a TT modification and not a game match. The wrapper should:

1. launch the exact production engine command;
2. log timestamp, process ID, and every line received from the GUI/harness before forwarding it unchanged to the engine;
3. log every engine output line, especially `readyok`, `bestmove`, and `info hashfull`;
4. preserve stdin/stdout buffering and exit status so it cannot alter protocol timing materially;
5. run a very small repeated-game smoke test using the same `cutechess-cli` command and options as production;
6. count, per engine process, the number of `ucinewgame` lines and the number of completed `bestmove` responses, checking that each new-game boundary precedes the next game's first `position`.

A shell or Python wrapper is sufficient; it needs no engine change. For a GUI deployment, the same wrapper can be configured as the engine executable. For a runner that cannot wrap commands, a short protocol-recording proxy can sit between the runner and engine. The diagnostic should also test a deliberately missing-`ucinewgame` transcript manually, confirming that the engine still answers and documenting that TT and adapter state persist by design.

The minimum acceptance report should include process reuse, games observed, `ucinewgame` count, first-position-after-boundary count, unmatched boundaries, and whether `isready` follows each boundary. A zero count is not automatically a bug because UCI permits unsupported clients; it is a deployment fact that changes the priority decision. A mismatch in a deployment that claims game isolation is an integration defect and should be fixed there before changing TT policy.

A useful optional black-box signal is to compare `info hashfull` on the first search after a boundary. This is not a proof: hashfull is sampled occupancy, not a generation metric, and a clear table may still report zero while an aged table might report a low nonzero value. It can nevertheless reveal an unexpected absence of clearing in the current implementation. The primary evidence must remain the captured command stream.

## If implementation is later reconsidered

No implementation is recommended in this Tier 1 report. If the diagnostic demonstrates materially missing boundaries or if an owner explicitly wants cross-game cache retention, the smallest defensible design would be a separate Tier 2 plan:

* add a compact generation/epoch field to each slot, with wraparound-safe relative-age arithmetic;
* advance the epoch at each root `go` or at an explicitly defined search boundary, not merely on `position` (because UCI may send multiple analysis positions within one logical game);
* use age as a modest replacement penalty rather than blindly rejecting deep entries;
* retain key validation and lock-free torn-read handling;
* decide separately whether `ucinewgame` still clears adapter state while only aging or partially clearing the TT;
* add deterministic collision/replacement tests, wraparound tests, same-key bound/depth tests, and concurrent stress tests;
* measure nodes, NPS, TT hit/cutoff rates, hashfull, and first-move stability on fixed positions with cold and warm tables;
* run the required real paired-game cutechess SPRT before any default or search-behavior change.

A generation advanced at every `go` would age entries rapidly in the adapter because one user move can invoke opponent-observation probes, book-refutation searches, and the main search. That is not necessarily wrong, but it illustrates why “one generation per root search” must be defined against this engine's multiple search calls rather than copied mechanically from Stockfish. Conversely, advancing only on `ucinewgame` would not solve stale entries across root searches within an analysis session. The boundary semantics and intended reuse policy must be specified before coding.

## Verified versus assumed

| Item | Status |
|---|---|
| Master brief read | Verified: item 21 asks for TT aging and actual `ucinewgame` deployment assessment. |
| Reinforcement documents 00–12 consulted | Verified; conclusions and reporting gates were used for continuity. |
| Branch and commit | Verified: `manus/research-facilities`, `667e9068c26b8201e81435d4a00c2d4a5ec9fd13`. |
| Current TT is depth-preferred with no generation field | Verified in `unchessed-core/src/tt.rs:20–193`. |
| Current TT is shared by Lazy SMP and lock-free | Verified in `tt.rs` comments and `search.rs` integration. |
| `ucinewgame` clears TT and resets adapter game state | Verified in `unchessed-core/src/uci.rs:353–378`. |
| `position` does not clear TT | Verified by UCI handler inspection. |
| Root searches advance no generation | Verified: no generation API exists and no call occurs in `run_go`/search setup. |
| Checked-in harnesses use UCI/cutechess | Verified in the inspected WSL and SPRT scripts. |
| Every production client sends `ucinewgame` | **Not verified**; no live protocol trace was available. |
| Engine build | Verified: `cargo build --release -p unchessed-engine` was not the correct package target; metadata identified `unchessed-adapter`, which was built successfully for the diagnostic. |
| Minimal UCI smoke diagnostic | Verified: adapter answered `uciok`, `readyok`, produced `bestmove`, and after an explicit `ucinewgame` answered the next `position`/`go`; `hashfull` was nonzero during the first search and returned to zero on the first depth-1 search after the clear. |
| Elo, node, hit-rate, or stale-entry impact | Not measured. |
| Implementation | Not performed. |

The smoke transcript demonstrated protocol liveness and the visible effect of the existing clear; it did not prove that an external harness sends the boundary, nor did it measure whether retaining old entries would help or hurt.

## Recommendation

**Defer TT generation/aging as a strength change.** The current implementation already has a reliable in-engine response when `ucinewgame` arrives: full TT clearing plus game-state reset. The intended cutechess harness is likely to deliver the command because it uses `proto=uci`, but that fact was not captured at the engine boundary and UCI expressly allows clients that omit it. The cheapest next action is therefore deployment observability, not replacement-policy code.

If logging shows missing boundaries in a supported deployment, fix the lifecycle contract or document process isolation first. Reopen generation aging only after a measured workload demonstrates that full clears cost useful warm-cache work or that stale deep entries measurably displace current entries. Preserve defaults until a real paired-game SPRT supports a candidate; do not claim that Stockfish's generation scheme transfers automatically to Unchessed's one-slot table and adapter-specific multiple-search-per-move lifecycle.

## References

[1]: https://www.wbec-ridderkerk.nl/html/UCIProtocol.html "Universal Chess Interface protocol reference; isready, setoption, and ucinewgame semantics"
[2]: ../../upload/pasted_content_6.txt "Unchessed AI master research and engine-strength brief, Tier 1 item 21"
[3]: https://chessprogramming.org/Transposition_Table "Chessprogramming Wiki: transposition-table replacement and aging overview"
[4]: https://github.com/official-stockfish/Stockfish/blob/master/src/tt.cpp "Stockfish source: generation field, relative age, and age-aware replacement"
[5]: https://github.com/official-stockfish/Stockfish/blob/master/src/tt.h "Stockfish TT API: new_search and generation lifecycle"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/29-tt-aging.md`

**Final disposition:** **defer; diagnose protocol delivery first; no implementation.**
