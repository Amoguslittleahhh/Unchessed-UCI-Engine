# 26 — UCI AnalyseMode

**Investigation ID:** `uci-analyse-mode`
**Tier:** 1 (protocol completeness / infrastructure)
**Status:** Research and design only; no implementation, default change, match, or SPRT was performed.

## Executive conclusion

**Recommendation: defer implementation as a low-priority compatibility option; do not treat it as a strength improvement.** `UCI_AnalyseMode` is a standard UCI check option whose semantic contract is intentionally broad: it tells an engine whether it should behave differently while analysing rather than playing a game. The protocol itself gives learning as an example, not a required behavior. It does not define a universal search algorithm, contempt policy, time policy, or output change associated with the flag.[1]

For this repository, the useful behavior people normally expect from analysis mode—neutral draw handling and no adaptive/game-playing behavior—is already mostly selected by the command shape `go infinite`. The engine’s current code classifies `go infinite` as non-game mode, gates the adaptive persona and contempt path on game mode, and passes a neutral draw score to search otherwise. Consequently, adding a parsed `UCI_AnalyseMode` bit with no explicitly specified effect would be cosmetic. Adding it as an override that silently changes Contempt would create a second, potentially surprising control for behavior that is already directly tunable through `Contempt`.

The best future design is therefore a **small, explicit protocol compatibility option** only if GUI interoperability or user-visible configuration requires it: advertise `option name UCI_AnalyseMode type check default false`, accept it case-insensitively, and use it as an analysis-context hint—not as an independent contempt value. In a true analysis context, the effective draw score should be neutral (`0`) and adaptive move-selection/persona logic should remain disabled. The existing explicit `Contempt` option should continue to control game searches. Any implementation would need tests for state transitions, `go infinite`, fixed-depth analysis, and interaction with `Adaptive`, `UCI_LimitStrength`, `Contempt`, and `ucinewgame`; it should not change the default or claim Elo benefit without a paired-game SPRT.

## What the UCI specification means

The authoritative UCI description defines `UCI_AnalyseMode` as a `check` option: “The engine wants to behave differently when analysing or playing a game.” It says the GUI sets it false when the engine is playing a game and true otherwise, and gives engine learning as an example of a possible difference. The same specification says UCI-prefixed options receive special GUI treatment and should be set automatically rather than presented as ordinary tuning controls.[1] The specification does **not** say that analysis mode must disable contempt, increase search depth, ignore clocks, clear learning, or alter evaluation. Those are engine-specific policies.

The specification’s example sequence is instructive: after `ucinewgame`, a GUI that is about to analyse sets `UCI_AnalyseMode true`, then sends `position` and `go infinite`.[1] Modern Stockfish documentation likewise describes UCI as a command protocol and documents `go`, `ucinewgame`, and standard options, but current Stockfish releases no longer expose `UCI_AnalyseMode` or a Contempt tuning option in their ordinary option list.[2] That is useful evidence that the flag is a compatibility convention rather than a universally necessary runtime feature. It also argues against copying a historical Stockfish “analysis contempt” policy without first defining what Unchessed should promise.

`go infinite` has a separate, unambiguous UCI meaning: search until `stop`, which is the normal GUI analysis workflow. A GUI may set both the flag and `go infinite`; an engine must not assume that every GUI does so. Therefore the robust design should support both signals, with a documented precedence rule, rather than making analysis depend on a GUI remembering a nonessential option.

## Repository inspection

The master brief lists `UCI_AnalyseMode` as confirmed absent and asks whether it matters given the existing `Contempt` option. I inspected the UCI layer, search limits, adapter configuration, and reinforcement reports 00–12. No implementation was added.

The current UCI advertisement in `unchessed-core/src/uci.rs` emits options for hash, threads, MultiPV, search/evaluation controls, adaptive behavior, strength limiting, `UCI_Elo`, and `Contempt`, but no `UCI_AnalyseMode` (source around `uci.rs:245–323`). The option parser handles `contempt` and clamps it to `0..=100` (`uci.rs:665–668`), but has no AnalyseMode key. Unknown options are therefore not a supported stateful analysis signal.

The important existing behavior is already mode-aware:

| Area | Current behavior | Evidence | Consequence |
|---|---|---|---|
| Analysis detection | `Limits::is_game_mode()` returns false for `infinite`; it treats clocks, movetime, depth, and nodes as game-mode requests when not infinite | `search.rs:113–122` | `go infinite` is already the engine’s analysis signal |
| Adaptive/persona logic | `adaptive_now = ident_adaptive && (Adaptive || UCI_LimitStrength) && game_mode` | `uci.rs:1345–1357` | Analysis by `go infinite` bypasses adaptive move selection and fixed-strength selection |
| Draw score | Analysis/non-adaptive path sets `draw_score = 0`; only adaptive game searches call `draw_score_for` | `uci.rs:1624–1629` | `go infinite` already neutralizes the contempt path |
| Search representation | Search stores the supplied value as `root_draw`, clamped to `-100..=100` | `search.rs:988–999` | Contempt affects draw outcomes through the search’s draw score, not as a general static-eval offset |
| Contempt policy | `draw_score_for` returns zero when non-adaptive or defending; otherwise it returns `-(contempt/2)` in Clinch or `-(contempt/3)` in other modes | `adapt.rs:740–750` | Current Contempt is conditional, adapter-driven, and deliberately not a universal analysis offset |
| New game | `ucinewgame` joins the worker, clears TT, resets opponent model/persona, and starts a new game | `uci.rs:353–378` | An AnalyseMode state must be reset or deliberately persisted; the protocol expects GUI-controlled state |

There is also an important distinction between **analysis mode** and arbitrary bounded searches. Current tests classify fixed depth and node-limited searches as game mode because those forms are used by engine-match harnesses (`search.rs` tests around `game_mode_detection`). This is a pragmatic repository convention, not a complete UCI semantic definition. A GUI may issue `go depth 20` for analysis, so blindly using the current `is_game_mode()` result as the final meaning of `UCI_AnalyseMode` would make bounded GUI analysis behave differently from `go infinite` in surprising ways.

The existing reports consistently require separating verified behavior from design assumptions, keeping defaults unchanged, and requiring a real paired-game SPRT before any live search behavior changes. Reports 00–12 also establish that the adapter/persona subsystem is experimental and default-sensitive, while previous protocol investigations favor cheap correctness and explicit gates over speculative strength claims. This item fits that pattern: it is primarily an interoperability and semantics decision, not a search-strength opportunity.

## Does AnalyseMode matter given Contempt?

**It matters only if the engine promises a distinct analysis policy.** Contempt and AnalyseMode are not equivalent concepts:

* `Contempt` is a user-selectable preference for how the engine values draw outcomes in a game-oriented search. In this codebase it is an adapter-controlled draw score and is not applied universally.
* `UCI_AnalyseMode` is a context signal from the GUI. It says “this search is analysis rather than play”; it does not specify the magnitude or direction of contempt.
* `go infinite` is a search-limit command and, in this repository, currently doubles as the analysis-context signal.

With current defaults, `go infinite` already produces `draw_score = 0`, regardless of the configured Contempt value, because `adaptive_now` is false. Thus the most common reason to add AnalyseMode—turning off game-oriented contempt during analysis—is already satisfied for infinite analysis. There is no evidence that adding a redundant option would improve playing strength, evaluation calibration, or search speed.

There are two real compatibility gaps. First, a GUI may set `UCI_AnalyseMode true` and then use a bounded `go depth` or `go nodes` analysis request; the current code would classify those limits as game mode and could enable adaptation if configured. Second, a GUI may set the option false while sending `go infinite`; an explicit command-level analysis signal should not be overridden by a stale option. These are correctness/UX concerns, not grounds for changing the default policy.

## Design-only recommendation

If compatibility demand justifies implementation later, use this contract:

1. **State and advertisement.** Add a boolean `analyse_mode` to UCI options, default `false`, and advertise the fixed-name UCI option. Parse `true`/`false` case-insensitively. Do not expose it as a normal strength-tuning control or change `Contempt` when the option changes.
2. **Effective context.** Define `analysis_context = analyse_mode || limits.infinite`. This makes `go infinite` safe even when a GUI does not support the option, while honoring the explicit flag for bounded analysis. The explicit `false` value must not disable the intrinsic analysis meaning of `go infinite`.
3. **Behavior in analysis context.** Disable adaptive persona/weakening selection and use a neutral draw score (`0`) for the search. Do not alter evaluation weights, pruning, MultiPV, time management, or NNUE behavior solely because of the flag. This is a minimal and unsurprising interpretation.
4. **Behavior in game context.** Preserve existing `Adaptive`, `UCI_LimitStrength`, and `Contempt` behavior. A user who wants contempt in a bounded analysis must be able to request it only if the project explicitly chooses that policy; the safer default is that analysis is neutral and the report must document the precedence.
5. **Lifecycle.** Treat the setting like other UCI options: it remains until changed, including across `position`; `ucinewgame` should reset search/game state but should not silently reset a GUI option unless the project documents that exception. `isready` must remain the synchronization point after any future expensive reconfiguration.
6. **Tests before any integration.** Add parser/advertisement tests; verify `go infinite` is neutral with `Contempt=100`; verify `UCI_AnalyseMode=true` neutralizes bounded depth/node searches; verify false plus `go infinite` remains analysis; verify game-mode searches preserve current contempt and adaptation; and verify option changes during a running search follow the UCI waiting-state rule rather than racing with a worker.

An alternative design—having AnalyseMode merely mirror `go infinite` and not affect bounded searches—is simpler but offers little beyond protocol acknowledgment. It should be chosen only if a GUI compatibility requirement specifically demands the advertised option. A design that automatically writes `Contempt=0` into the user’s option state is **not recommended**, because it conflates context with configuration, loses the user’s selected value, and makes transitions difficult to reason about.

## Verification, limitations, and non-results

| Category | Result |
|---|---|
| Repository branch | Verified: `manus/research-facilities` |
| Existing docs | Read master brief and reinforcement docs 00–12; conclusions above are consistent with their design-only/default-preserving rules |
| Source inspection | Verified absence of `UCI_AnalyseMode`; verified current `go infinite`, adaptive gating, Contempt parsing, draw-score wiring, and `ucinewgame` reset behavior |
| External research | Verified UCI specification semantics in the published protocol text and cross-checked current Stockfish UCI documentation |
| Build/tests | Attempted focused Rust tests, but this host’s Cargo 1.75.0 cannot parse the repository’s lockfile version 4 (`requires -Znext-lockfile-bump`). No test result is represented as passing |
| Implementation | None; no source or default was changed |
| Strength evidence | None; no cutechess match, SPRT, benchmark, or Elo claim was made |

## Final decision

**Defer.** Keep the current behavior and document that `go infinite` is the supported analysis signal and already neutralizes the adapter’s Contempt path. Reopen only for a concrete GUI compatibility issue or after a small protocol test matrix demonstrates a bounded-analysis inconsistency worth fixing. If implemented, make it a context flag with neutral analysis semantics, not a hidden Contempt setter and not a new search heuristic. Any behavior affecting game searches or defaults remains subject to the master brief’s paired-game SPRT gate.

## References

[1] [Description of the Universal Chess Interface — option and `UCI_AnalyseMode` specification](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf). The protocol text defines the option as a GUI-controlled check and gives learning as an example rather than a mandated behavior.

[2] [Stockfish Wiki, “UCI Protocol and Stockfish Commands”](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html). Official documentation for standard UCI commands/options and current Stockfish option behavior.

[3] [Chessprogramming Wiki, “UCI”](https://www.chessprogramming.org/UCI). Background on UCI’s role and engine/GUI division of responsibilities.

[4] Repository source: [`unchessed-core/src/uci.rs`](../../unchessed-core/src/uci.rs), [`unchessed-core/src/search.rs`](../../unchessed-core/src/search.rs), and [`unchessed-core/src/adapt.rs`](../../unchessed-core/src/adapt.rs).

[5] Repository-local context: reinforcement investigations [`00-synthesis.md`](00-synthesis.md) through [`12-tier2-calibration.md`](12-tier2-calibration.md) and the master brief at `/home/ubuntu/upload/pasted_content_6.txt`.

**Report status:** complete; design-only recommendation, no implementation or strength claim.
