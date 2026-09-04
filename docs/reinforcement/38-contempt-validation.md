# Contempt validation

## Executive summary

The advertised `Contempt` option is real, but it is not a direct evaluation bonus and it is not an Elo control. In the current implementation it is converted into a **negative draw score supplied to the search** only when the adaptive path is active and the current persona is not `Defend`. The default is `Contempt=25`; that produces a draw score of `-8` centipawns in `Match` (integer division) or `-12` in `Clinch`, while `Defend` produces `0`. `Adaptive=false` also produces `0` regardless of the option value. The current code has one narrow unit test for these sign/defend invariants, but no sweep, fixed-position draw-choice validation, game result study, or Elo evidence for 25.

A real UCI fixed-position probe was feasible and was run against the existing release binary. It verified that the option is accepted and can alter shallow search work/PV selection in an adaptive drawish position, while leaving the displayed score nearly unchanged in the tested K-v-K position. That result is expected: the probe position is not recognized as a terminal dead-position draw by this engine, and a draw score influences terminal/repetition draw treatment rather than being a simple root score offset. It is therefore a wiring/smoke test, **not evidence that 25 is strong or Elo-optimal**.

**Recommendation: defer tuning and preserve the default.** Keep the existing option and behavior, add cheap deterministic observability/tests that expose the selected mode and effective draw score, then run a predeclared, paired, fixed-opening game experiment before considering any default change. The next experiment should measure outcomes and draw rates against a clearly specified opponent class; it must report score behavior separately from Elo. No Tier 2/3 implementation, default change, training, or match campaign was started here.

## Scope and constraints

This investigation covers exactly one item: Contempt validation. The repository was inspected on branch `manus/research-facilities`; no source code or default was changed. Existing worktree changes belonging to other numbered investigations were left untouched. The standing rule requiring real-world testing whenever feasible was followed with a UCI probe and a build/test attempt. No cloud spend or expensive training was used.

## Verified implementation trace

The UCI option is advertised by the existing release binary as:

```text
option name Contempt type spin default 25 min 0 max 100
```

The relevant source path is:

| Location | Verified behavior |
|---|---|
| `unchessed-core/src/adapt.rs:740-750` | `draw_score_for` returns `0` when `!cfg.adaptive` or `prev == Mode::Defend`; otherwise `Mode::Clinch` returns `-(contempt / 2).clamp(0, 50)` and all other modes return `-(contempt / 3).clamp(0, 33)`. |
| `unchessed-core/src/adapt.rs:713-716` | `Clinch` is entered only after move 28 in a drawish position (`abs(eval) < 60`) when `cfg.contempt > 0`, or held while within ±100 cp. |
| `unchessed-core/src/uci.rs:1619-1629` | The worker reads the current adaptive state, calls `draw_score_for` when `adaptive_now` is true, and otherwise passes `0`. |
| `unchessed-core/src/search.rs:998` | The resulting value is clamped to ±100 and stored as `root_draw`; it is a search draw value, not an evaluation-file parameter. |
| `unchessed-core/src/adapt.rs:1353-1359` | Existing test `draw_score_respects_defend` verifies only that the default is negative in `Clinch` and `Match`, and zero in `Defend`. |

For the default `contempt=25`, integer arithmetic gives `-12` in `Clinch` and `-8` in `Match`. The sign is from this engine's perspective: a draw is mildly worse while trying to win. The `Defend` exception makes a draw a rescue rather than something to reject. The option therefore has a second interaction with the persona state machine: setting a positive contempt permits late-game `Clinch` entry, whereas zero disables that entry. This is distinct from the numeric draw score itself.

The search API names the parameter `draw_score`, and `root_draw` is initialized from it. The source inspection does not establish every downstream terminal/repetition call site in this report, but it does establish that the value is passed into search state rather than added to every static NNUE/HCE score. Consequently, a displayed `info score cp` should not be interpreted as “evaluation plus Contempt.”

## Real fixed-position UCI probe

### Exact command

The following command was run from the repository against the pre-existing `target/release/unchessed-adapter` binary. Book use was disabled, and the same bare-king FEN was searched to depth 5 for three Contempt values with Adaptive both enabled and disabled:

```bash
cd /home/ubuntu/Unchessed-UCI-Engine
for c in 0 25 100; do
  for a in true false; do
    printf '\n--- contempt=%s adaptive=%s ---\n' "$c" "$a"
    printf 'setoption name OwnBook value false\nsetoption name Adaptive value %s\nsetoption name Contempt value %s\nposition fen 8/8/8/3k4/8/8/4K3/8 w - - 0 1\ngo depth 5\nquit\n' "$a" "$c" |
      timeout 20s target/release/unchessed-adapter 2>/dev/null |
      grep -E '^(info depth 5|bestmove)'
  done
done
```

The complete captured output is preserved at `/home/ubuntu/jobs/job_KZC1ecIq_a/contempt_probe.txt`. The relevant output was:

```text
--- contempt=0 adaptive=true ---
info depth 5 multipv 1 score cp -13 nodes 1081 nps 1081000 hashfull 0 time 1 pv e2d3 d5e5 d3e3 e5f5 e3d4
bestmove e2f3
--- contempt=0 adaptive=false ---
info depth 5 multipv 1 score cp -13 nodes 548 nps 548000 hashfull 0 time 1 pv e2d3 d5e5 d3e3 e5f5 e3d4
bestmove e2d3
--- contempt=25 adaptive=true ---
info depth 5 multipv 1 score cp -13 nodes 1140 nps 1140000 hashfull 0 time 1 pv e2d3 d5e5 d3e3 e5f5 e3d4
bestmove e2f3
--- contempt=25 adaptive=false ---
info depth 5 score cp -13 nodes 562 nps 562000 hashfull 0 time 1 pv e2d3 d5e5 d3e3 e5f5 e3d4
bestmove e2d3
--- contempt=100 adaptive=true ---
info depth 5 multipv 1 score cp -14 nodes 1208 nps 1208000 hashfull 1 time 1 pv e2f3 d5e5 f3g4 e5e4 g4g5
bestmove e2d3
--- contempt=100 adaptive=false ---
info depth 5 multipv 1 score cp -13 nodes 489 nps 489000 hashfull 0 time 1 pv e2d3 d5e5 d3e3 e5f5 e3d4
bestmove e2d3
```

The output verifies three limited facts. First, the option is accepted. Second, with `Adaptive=false`, changing Contempt from 0 to 100 did not change the displayed depth-5 score/PV in this probe, consistent with the explicit zero path. Third, with `Adaptive=true`, the high value changed the reported PV at depth 5 and changed node count, while the score moved only from -13 to -14 cp. This is search behavior, not an Elo measurement.

There is an important negative result. The FEN is king versus king, but the engine's current search did not emit an immediate rule-draw result; it searched legal king moves and displayed ordinary centipawn scores. This means it is not a suitable direct terminal-draw fixture for validating the numeric draw score. It also shows why a future validation harness must use positions that reach a recognized repetition/terminal draw through legal move sequences, or add a narrowly scoped test fixture around the search's actual draw predicate. No claim is made here that the engine's lack of dead-position detection is caused by Contempt; that is outside this item.

### Build/test check

The existing release binary was usable for the probe. A source-level focused test was also attempted:

```bash
cd /home/ubuntu/Unchessed-UCI-Engine
cargo test -p unchessed-core draw_score_respects_defend -- --exact --nocapture
```

Exact result:

```text
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:
  lock file version 4 requires `-Znext-lockfile-bump`
```

Thus the unit test source was inspected, but no passing Cargo test result is claimed. The blocker is the installed Cargo toolchain's inability to parse this workspace's lockfile version; changing the lockfile or toolchain was not justified for this Tier 1 investigation.

## Historical and external evidence

The [Chessprogramming Wiki definition of Contempt Factor](https://www.chessprogramming.org/Contempt_Factor) describes contempt as an estimate of superiority/inferiority assigned as the draw score: it is intended to avoid early draws against an apparently weaker opponent, or to prefer draws against a stronger opponent. This supports the current code's conceptual placement in draw handling, but it does not validate the project-specific value 25.

The [Stockfish issue on calibrating contempt against Leela](https://github.com/official-stockfish/Stockfish/issues/2222) is useful historical evidence, not a transferable constant. The discussion distinguishes static contempt from dynamic contempt and explicitly notes the need for serious data; one comment describes reducing draw rate with a small expected-score cost as potentially useful against a slightly stronger opponent, while another reports that a Stockfish contempt value around 24 had been observed as a modest regression against master and useful against lower-tier engines. The same issue includes a small computer-chess-event comparison of Stockfish contempt 0 versus 100, but it is not a controlled, modern, project-matched SPRT and cannot establish an Elo value for Unchessed.

The [Lc0 WDL rescale/contempt documentation](https://lczero.org/blog/2023/07/the-wdl-rescale/contempt-implementation/) was searched as a relevant implementation reference, but the page could not be extracted by the available text fetcher. It is therefore not treated as verified evidence in this report. Search-result material and forum anecdotes were not used as quantitative proof.

Historical repository artifacts confirm that the shipped adapter option was logged as default 25 in prior exhibition logs, but those logs do not isolate Contempt or provide a controlled value comparison. No Contempt-specific completed SPRT, Elo estimate, or default-selection rationale was found in the inspected prior reinforcement documents or repository history. This is a negative result, not evidence that no test has ever existed outside the repository.

## Validation design

A defensible validation should separate two questions:

1. **Score/search behavior:** Does changing Contempt produce the intended draw aversion only in recognized draw decisions, preserve mate/win/loss ordering, respect Adaptive and Defend neutralization, and avoid unintended score shifts in non-draw positions?
2. **Playing strength/Elo:** Against a specified opponent distribution and time control, does the option improve expected match score, tournament points, or a chosen practical objective enough to justify its risk?

These questions require different evidence. A fixed-position test can validate legal behavior, effective draw-score signs, terminal/repetition handling, PV stability, and unintended score differences. It cannot establish Elo. Conversely, a game match can estimate outcome effects but cannot by itself prove that a particular root score is being applied correctly.

The recommended Tier 1 validation sequence is:

| Stage | Design | Pass/fail or decision rule |
|---|---|---|
| Deterministic wiring test | Exercise `draw_score_for` over Contempt `{0,1,25,50,100}`, modes `Match`, `Clinch`, and `Defend`, with Adaptive on/off. Assert exact integer results, monotonicity, caps, and zero neutralization. | Any sign, cap, mode, or Adaptive regression rejects the candidate. |
| Draw-fixture search test | Use legal repetition fixtures (and any already-supported terminal draw fixtures), record effective draw score, root outcome, PV, mate scores, and nodes at fixed node/depth budgets. Include `Contempt=0,25,100`, Adaptive on/off, and `Defend`. | No false mate/result, illegal move, or non-draw score regression. Confirm that only recognized draw choices respond. |
| Position corpus screen | Use a fixed, versioned corpus partitioned into repetition-prone/endgame, objectively winning, objectively losing, and tactical positions. Record cp/mate, PV, nodes, and draw/repetition decisions. | Reject if non-draw positions receive broad unexplained changes or if draw aversion causes tactical/mate safety failures. |
| Paired game experiment | Same binary/model/options except Contempt; identical paired openings with colors swapped; fixed hardware, threads, hash, book, and time control. Compare 0, 25, and one or two predeclared alternatives, not a post-hoc sweep. | Report W/D/L, expected score, draw rate, adjudications, crashes, and a sequential confidence interval/SPRT result. Do not promote on draw-rate change alone. |

For a future game study, the primary comparison should be the shipped `Contempt=25` against `Contempt=0`, with an optional exploratory arm such as 50 only if budget allows. The opponent set must be declared in advance: self-play alone answers a different question from play against weaker/stronger engines. If the purpose is default quality for unknown users, a mixed opponent distribution is more relevant than a single adversarial engine. Opening pairs and color swaps are essential because contempt is asymmetric with respect to win-seeking and defending states.

The report from such a study must preserve binary and model hashes, UCI options, book/opening identities, raw PGNs and logs, hardware, time control, and exact statistical parameters (`elo0`, `elo1`, alpha, beta, and whether the result is an upper bound, lower bound, or inconclusive). A lower draw rate is a behavioral observation, not automatically a strength gain. A change in average cp is also not an Elo result. A candidate should not replace the default unless the completed paired test supports the stated objective without unacceptable tactical or practical regressions.

## Verified versus assumed

| Status | Statement |
|---|---|
| **Verified** | The UCI option is advertised with default 25 and range 0–100 by the existing release binary. |
| **Verified** | `draw_score_for` returns zero for inactive Adaptive and `Defend`; otherwise it returns a negative integer with `Match` divisor 3 and `Clinch` divisor 2, each capped. |
| **Verified** | Positive Contempt enables the late drawish `Clinch` mode entry condition. |
| **Verified** | The UCI worker passes the value into search as `draw_score`; search stores it as bounded `root_draw`. |
| **Verified** | A real depth-5 UCI probe showed no displayed-score change with Adaptive disabled and a small PV/node change at Adaptive=true, Contempt=100, on the tested K-v-K FEN. |
| **Verified** | The K-v-K probe was not recognized as an immediate dead-position draw by this binary. |
| **Verified** | The source-level Cargo test was blocked by lockfile version 4/toolchain incompatibility. |
| **Verified** | Public historical sources explain the purpose of contempt and show that other engine communities treated its value as an empirical tradeoff, not a universal constant. |
| **Not verified** | That `Contempt=25` improves Unchessed Elo, expected score, tournament performance, or human-facing results. |
| **Not verified** | That 0, 25, 50, or 100 is optimal against any particular opponent class. |
| **Not verified** | The complete downstream draw/repetition semantics beyond the observed `root_draw` handoff. |
| **Assumed/design proposal** | A repetition-fixture harness and paired SPRT are the appropriate next validation tools; they were not implemented or run here. |

## Recommendation and decision

**Keep the default at 25 and defer tuning.** The implementation is coherent enough to justify validation, but the evidence is insufficient for a default change or an Elo claim. The immediate low-cost follow-up is an observability/test patch that reports the active persona and effective draw score in a diagnostic mode and adds deterministic unit coverage for the full table. After that, run the fixed repetition/endgame screen and only then a small, provenance-complete paired game experiment. Do not infer strength from the current UCI cp output, from the historical Stockfish number 24, or from reduced draw rate alone.

This item is complete as research. No Tier 2/Tier 3 implementation, expensive training, default modification, commit, or push was performed.

## References

1. [Chessprogramming Wiki — Contempt Factor](https://www.chessprogramming.org/Contempt_Factor).
2. [Stockfish issue #2222 — Calibrate contempt against Leela](https://github.com/official-stockfish/Stockfish/issues/2222).
3. [Lc0 — The WDL rescale/contempt implementation](https://lczero.org/blog/2023/07/the-wdl-rescale/contempt-implementation/); searched but not counted as verified because the available fetcher returned no extractable page content.
4. Repository source: `unchessed-core/src/adapt.rs`, `unchessed-core/src/uci.rs`, and `unchessed-core/src/search.rs` on branch `manus/research-facilities`.
5. Real probe artifact: `/home/ubuntu/jobs/job_KZC1ecIq_a/contempt_probe.txt`.
6. Prior synthesis: `docs/reinforcement/31-tier1-master-synthesis.md`.
7. Task context: `/home/ubuntu/upload/pasted_content_7.txt`.

## Reproducibility metadata

The repository branch was `manus/research-facilities`. The probe used `/home/ubuntu/Unchessed-UCI-Engine/target/release/unchessed-adapter` and the exact command shown above. The source test command and its exact Cargo error are recorded above. The report path is `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/38-contempt-validation.md`.
