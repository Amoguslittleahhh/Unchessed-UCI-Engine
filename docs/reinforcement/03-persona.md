# 03 — Real-play persona and engine-detection measurement

**Topic:** Real-play persona and engine-detection measurement  
**Investigation scope:** `adapt.rs`, UCI options and lifecycle, self-play/exhibition/SPRT tooling, archived logs, artifacts, and tests.  
**Repository state inspected:** `/home/ubuntu/Unchessed-UCI-Engine`, branch `manus/rustc-bootstrap-trial`; no tracked repository files were changed.

## Required structured fields

| Field | Value |
|---|---|
| **ID** | `03-persona` |
| **Topic** | Real-play persona and engine-detection measurement |
| **Summary** | The adapter implements the persona state machine and EngineDetectV2 as live, UCI-gated code, but the repository does **not** yet produce a machine-joinable, truth-labelled real-game telemetry stream. Existing `info string` records can reconstruct a coarse historical mode sequence, not a controlled candidate-versus-baseline flip-rate comparison or detection accuracy. A combined real cutechess run reached 5,537/10,000 games at approximately +2.1 Elo but was interrupted before an SPRT bound; it measured Elo only, not the stated behavioural outcomes. The safe next engineering step is a default-off, append-only telemetry option plus an offline parser/manifest. Actual behavioural claims require new labelled real games; any default or search-behaviour change still requires a completed real SPRT. |
| **Implemented-ready work** | Default-off UCI options `PersonaSmooth` and `EngineDetectV2`, persistent `PersonaState`, game-mode gating, raw UCI diagnostic lines, a two-game cutechess smoke harness, and a same-binary candidate/baseline SPRT harness already exist. A precise no-default-change telemetry extension is specified below. |
| **Blocked / not yet evidenced** | There is no committed current real-play corpus containing option state, game/run IDs, detector booleans/signals, per-observation timing/difficulty, and independent expected outcomes. The 5,537-game raw PGN/log is explicitly local and uncommitted. No current labelled human-on-clock dataset exists. The installed Cargo 1.75 cannot parse the v4 lockfile, so Rust unit tests were not compiled here. |
| **Requires a real SPRT** | Flipping either default; any code path that changes search or move selection; and any claim that an observed behavioural improvement is Elo-neutral/positive under the shipped conditions. Telemetry itself may be shipped default-off after ordinary compile/unit/protocol checks because it should only conditionally print diagnostic data, but this must be demonstrated rather than assumed. |

## Evidence first

### 1. What executes in actual games

`run_go` activates adaptation only for the Adapter, game-mode `go` commands, and `Adaptive` or `UCI_LimitStrength`; `go infinite` is excluded. It feeds pending opponent moves only when there is adequate remaining time, then chooses a persona and writes a per-decision `mode=` line. Consequently, normal fixed-movetime self-play can exercise persona selection, but **cannot exercise the clock tell**: it supplies no `wtime`/`btime`, and `opp_time_used` remains absent. At less than 10 seconds on the adapter’s clock, the observation probe is skipped altogether. [1][2][3]

The UCI surface is already fail-safe for the two investigated changes. Both `PersonaSmooth` and `EngineDetectV2` are advertised only by the adaptive Adapter and default to `false`; `ucinewgame` rebuilds both `OpponentModel` and `PersonaState`, then reapplies the detector option. The Rust regression test asserts precisely those defaults. [2][4]

The implementations are materially different. With smoothing off, `PersonaState::update` preserves the legacy raw-evaluation decision path. With it on, it uses alpha 0.35, two agreeing dwell votes, a confidence-dependent CLINCH pad, and immediate FULL/DEFEND/PUNISH emergency transitions. EngineDetectV2 changes opening weighting, clock thresholding, labelled-computer handling, and anonymous ceiling requirements. [1]

### 2. Existing measurements are useful but insufficient

| Evidence | What it measures | Result | What it cannot establish |
|---|---|---|---|
| Simulated persona harness | 2,500 AR(1) eval-trace games; simulated score and mode stream | legacy mean flip rate 0.1226; smooth 0.0523; reported 57.34% reduction | Real clock behaviour, live search variance, or Elo. The artifact itself says it is not cutechess. [5][6] |
| Synthetic detector tests | V2-like cp-loss/timing sequences plus 200 synthetic declared-human samples per Elo band | zero flags in each 1200–2500 band; fixed examples for Maia, Stockfish, opening premoves, and anonymous clean play | Accuracy against real labelled humans/engines or the legacy default detector. [7][8] |
| Combined real cutechess run | Elo/WDL of both options on versus both off, `Adaptive=true`, `tc=5+0.05` | interrupted at 5,537 games: 2125–2095–1317, score 0.503, about +2.1 Elo | A formal SPRT decision, the independent effect of either option, flip rate, or detection accuracy. [9] |
| Archived exhibition logs | Historical free-form mode/transition/observation diagnostics | 24 logs contain 1,167 `mode=` decisions, 59 transition lines, and 1,130 opponent-observation lines; a naïve aggregate transition fraction is 59/(1,167−24) = **5.16%** | Candidate-vs-baseline comparison: the logs were imported 2026-08-11, predate the later options, and do not record their state or truth labels. [10] |

The real run is valuable evidence that the *combined* opt-in configuration did not show a meaningful Elo loss in its observed trajectory. It is **not a passed SPRT**: its own report is explicit that neither bound was crossed. It should therefore not be re-described as a formal Elo acceptance, nor treated as flip-rate/detection validation. [9]

The archived logs prove a useful narrow point: current human-readable output is sufficient to see historical mode events. For example, it records `mode=MATCH`, opponent cp-loss estimates, then `persona MATCH -> FULL` when the old ceiling detector fires. It is not suitable as a ground-truth telemetry dataset. It lacks a run/game ID, option vector, `engine_suspect` boolean, detector reason, `suspicion`, observation weight, low-loss streak, `had_choice`, clock-use value, and expected label. [10][11]

### 3. Important detector measurement caveat

`tools/elo_detector.py` calls itself a replica, but it always applies the V2 opening discount and V2 `engine_suspect` rule. It has no `experimental_detect` switch, does not represent the default legacy rule, and its `observe` calls in tests always use difficulty weight 1.0. This is visible in both the tool and the test that expects a 1600 Maia computer **not** to be suspect—where current default Rust deliberately retains the legacy opposite result until `EngineDetectV2=true`. It is a reasonable targeted V2 misfire test, but it must not be used as evidence that the default detector or live weighted observation path has been replicated. [1][7][8]

There is a second semantic issue: `engine_suspect` is not a generic “is this software?” classifier. Under the documented V2 contract, a Maia-1600 computer is intentionally **negative** (MATCH target), while a strong declared engine is positive (FULL). Detection reporting must therefore carry an independent `expected_suspect` / “should force full” label, not score every computational opponent as a false negative. A separate anti-fingerprinting study may use `actual_origin=engine`, but that is a different target metric. [1][8][11]

## Implementation-ready, non-default telemetry design

No source change was made in this investigation because the task forbids changes to tracked repository files. The following is intentionally narrow: when disabled, it preserves UCI output and all live decisions; when enabled, it only emits parseable `info string` records.

### A. Add an opt-in UCI diagnostic switch

Add `AdapterTelemetry` as an Adapter-only UCI check option, default `false`, stored in `Options` and copied into `GoJob`. Do **not** make it affect `AdaptConfig`, search limits, random seeding, selection, timing, or model state. It should be advertised adjacent to `PersonaSmooth` and `EngineDetectV2` and handled using the same case-insensitive boolean convention. [2]

Add a monotonically increasing process-local `game_id` on `ucinewgame`, and `decision_index`/`observation_index` in `Game`. This is necessary because a single engine process can play many games and cutechess can interleave games at concurrency greater than one. No player names, FENs, or raw UCI opponent strings need be printed; a harness can join a pseudonymous `run_id` and `game_id` to an external manifest.

Emit stable, flat `key=value` records under a distinct prefix. Values that might contain whitespace should be omitted or encoded, rather than parsing prose `trend()`/selection reasons. Suggested schema version 1:

```text
info string [UnchessedTelemetry] v=1 event=opponent_observation run=<runner-id> game=17 ply=23 source=probe move=e7e5 adaptive=1 limit_strength=0 persona_smooth=1 engine_detect_v2=1 low_time=0 cp_loss=14 difficulty_weight_milli=800 legal_count=31 had_choice=1 opp_time_used_ms=1820 samples=9 estimate_elo=2187 confidence_cp=244 weight_milli=7410 suspicion_milli=0 low_loss_streak=2 is_computer=0 declared_elo=none suspect=0 suspect_reason=none action_full=0
info string [UnchessedTelemetry] v=1 event=persona_decision run=<runner-id> game=17 ply=24 decision=12 raw_eval_cp=-87 ema_cp=-53 mode_before=MATCH mode_after=MATCH candidate=MATCH dwell=0 emergency=none adaptive=1 limit_strength=0 persona_smooth=1 engine_detect_v2=1 selected_move=g1f3
```

This requires side-effect-free snapshots exposed by `adapt.rs` rather than UCI reaching private fields. `OpponentModel::telemetry_snapshot()` should include estimate, confidence, weight, suspicion, streak, declared/computer state, `engine_suspect`, and a stable reason enum for **both** legacy and V2 thresholds. `PersonaState::telemetry_snapshot()` should expose EMA, candidate, dwell, and mode. `PersonaState::update` may return an update record (before/after mode and emergency) or UCI may calculate `before != after`; the former is preferable because it preserves why a transition bypassed dwell. The telemetry must log an explicit book observation or an `observation_skipped` event too, so missing observations are measurable rather than silently absent.

`action_full` must remain separate from `suspect`: the latter can be true while `UCI_LimitStrength` prevents FULL, and it can be unavailable when adaptation is disabled. The stream should identify `clock_available`/`opp_time_used_ms=none`; otherwise fixed-movetime runs can be incorrectly presented as negative clock-tell evidence. [1][2]

### B. Add an offline parser and label manifest, not in-engine accuracy logic

Implement a stdlib-only `tools/analyse_adapter_telemetry.py` with two inputs: telemetry text and a JSONL manifest. It should reject unknown schema versions, malformed fields, duplicate `(run, game, event-index)` keys, and any row where baseline/candidate option state changes mid-game. Its outputs should be a JSON report and optional per-game JSONL—not a tracked artifact by default.

The runner-owned manifest should contain only externally known information:

```json
{"run":"2026-09-02-persona-v2","game":17,"opening_id":"book:123","arm":"v2","expected_suspect":false,"opponent_class":"human_like_engine","label_source":"controlled-maia","clock_protocol":"wtime-btime","options":{"Adaptive":true,"PersonaSmooth":true,"EngineDetectV2":true,"OwnBook":false}}
```

For human play, store a consented, pseudonymous opponent ID and an explicit label policy. Report results separately for (a) declared/metadata-assisted classification and (b) anonymous behaviour-only classification. Do not inject a declaration and call the outcome an independent detector benchmark: `UCI_Opponent` is intentionally an input to the model. If the harness sets this option, it must do so **after** `ucinewgame`, because the current new-game handler resets the model. [1][2]

### C. Reuse runners safely

The current exhibition runner already captures every engine stdout line, protocol command, calculated elapsed time, PGN, and move list. Its hard-coded paths and module-level arguments make it a poor reusable measurement CLI, but it is the smallest implementation base for a new `--telemetry-out`, `--run-id`, `--manifest-out`, and explicit engine/options command-line interface. [12]

For paired engine-only samples, add a **companion diagnostic run**, not a replacement for the Elo harness. The current cutechess SPRT script writes PGN and a tournament log but does not request engine I/O. cutechess documents `-debug` as “Display all engine input and output,” so `-debug > telemetry.log 2>&1` can preserve the opt-in records. Use `-concurrency 1` for a simple unambiguous first parser integration test, then rely on emitted game IDs before increasing concurrency. This diagnostic run is behavioural observation, not Elo evidence and not an SPRT. [13][14]

`tools/selfplay.py` should not be the sole measurement harness: it discards all lines before `bestmove`, writes no PGN/labels, varies Troll/Adaptive by game number, and sends `go movetime`, eliminating clock telemetry. It remains appropriate as a crash/protocol sanity check. [3]

## Metric contract and acceptance criteria

### Persona flip rate

For each game, define an **eligible persona decision** as a `persona_decision` event with `adaptive=1`. Let `D_g` be the count, and let `F_g` be the number of adjacent eligible events whose `mode_after` differs. Report `F_g / max(D_g - 1, 0)`, with games having fewer than two decisions reported separately rather than assigned zero. Report both pooled `sum(F_g)/sum(D_g-1)` and the unweighted per-game mean, the five mode shares over `D_g`, and counts by emergency type. Book plies and skipped/low-time observations belong in coverage statistics, not the denominator unless a persona decision actually happened.

Compare arms by paired opening (both colours) with a paired bootstrap confidence interval for the flip-rate difference. The primary behavioural criterion can be precommitted as a reduction with a confidence interval excluding zero, but that threshold is a product choice and **not** Elo acceptance. The reported simulated 57.34% is a hypothesis/benchmark, not an acceptance threshold for real play. [5][6]

### Detection policy accuracy

At each telemetry observation, compare `suspect` with manifest `expected_suspect`; report TP, FP, TN, FN, precision, recall, specificity, balanced accuracy, FPR, and first-positive detection ply. Use game-clustered bootstrap intervals, not per-ply independent intervals. Stratify at minimum by `opponent_class`, metadata-assisted vs anonymous arm, `clock_available`, opening/middlegame (`samples < 8` vs later), `had_choice`, and low-time/skip coverage.

A controlled minimum matrix is: strong declared engine (expected positive); human-like declared engine such as Maia-1600 (expected negative by product contract); declared humans at several levels (negative); anonymous strong engine (positive after evidence); and consented human clock games (negative). Report an “insufficient evidence” rate when there are too few non-book, non-low-time observations. This avoids treating no observation as a correct negative and avoids mis-scoring the designed Maia exception.

## Verification plan and gates

| Change / claim | Verification before merge or publication | Status from this investigation |
|---|---|---|
| Default-off telemetry patch | Current Rust toolchain: `cargo test --workspace --release`; unit tests for exact fields, reset/incremented `game_id`, legacy/V2 reason enum, no emitted telemetry when false; UCI transcript test confirming default output unchanged; parser fixtures for valid, malformed, interleaved, and missing-label data | **Not run/blocked locally.** System Cargo 1.75 stops before compilation because `Cargo.lock` v4 requires a newer Cargo. [15] |
| V2 Python replica credibility | Parameterize legacy versus V2, mirror Rust difficulty/book behaviour, and add cross-language golden vectors generated from the Rust model snapshots | **Needed.** Current tool is a targeted V2 approximation, not a mode-selectable replica. [1][7] |
| First diagnostic capture | Two-game smoke with telemetry disabled and enabled; validate records merge to PGN/manifest and no mode events are lost; then a small `-debug`, concurrency-1 cutechess companion run | **Ready once patched and a current binary/cutechess host is available.** Existing smoke only checks option presence and has no telemetry assertions. [13][16] |
| Real behavioural result | Pre-registered controlled matrix with paired openings, saved raw telemetry + manifest + PGN, parser report with coverage and clustered CIs | **Blocked on new labelled real-game data/logs.** Existing archive has no labels/option vector; 5,537-game raw output is uncommitted. [9][10] |
| Default flip or behaviour/strength claim | Complete real cutechess SPRT for the isolated candidate configuration; do not infer from simulation or diagnostic telemetry | **Still required by policy.** The existing combined run stopped before an SPRT bound. [9][13] |

## Recommendation

1. **Do not change either default from this evidence alone.** Keep `PersonaSmooth=false` and `EngineDetectV2=false` until the product owner chooses the behavioural target and a completed relevant SPRT is available.
2. **Implement `AdapterTelemetry=false` plus the snapshot-based stream and offline parser first.** It is isolated observability work, does not alter move selection when disabled, and closes the exact gap that makes current logs non-joinable.
3. **Repair the Python detector tool’s naming/parameterization before using it for comparative claims.** Preserve a V2-focused test mode if useful, but make legacy/V2 selection explicit and cover weighted/book observations.
4. **Collect a labelled companion diagnostic corpus before making persona or detector product claims.** Separate intent-aware `expected_suspect` policy accuracy from generic engine-origin/anti-fingerprinting research, and separate metadata-assisted from anonymous detection.
5. **Run real SPRT only after a behavioural choice implies a live/default change.** The telemetry corpus measures the desired outcomes; it cannot substitute for an Elo gate.

## References

[1]: file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/adapt.rs "OpponentModel, EngineDetectV2, PersonaState, and tests (inspected lines 64–315 and 390–502, 1047–1268)"
[2]: file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/uci.rs "UCI options, game lifecycle, game-mode adaptation and emitted diagnostic lines (inspected lines 44–130, 240–345, 370–414, 793–801, 1129–1527)"
[3]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/selfplay.py "Current fixed-movetime self-play sanity driver"
[4]: file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/uci.rs "Default-off UCI regression test (inspected lines 1652–1659)"
[5]: file:///home/ubuntu/Unchessed-UCI-Engine/artifacts/persona-stability-sprt.json "Simulation artifact and explicit non-cutechess note"
[6]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/persona-stability-and-sprt-correlation.md "Simulation methodology and limitations"
[7]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/elo_detector.py "Python detector approximation"
[8]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/test_elo_detector.py "Synthetic detector/misfire tests"
[9]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/persona-smooth-detect-sprt-result.md "Interrupted 5,537-game real cutechess run and scope limitation"
[10]: file:///home/ubuntu/jobs/job_c6EDrcuw_a2/log_audit.txt "Read-only aggregate and representative archived log audit performed for this report"
[11]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/elo-detector-high-level-integration.md "V2 detector product contract and remaining live validation"
[12]: file:///home/ubuntu/Unchessed-UCI-Engine/scripts/exhibition/play_3min_game.py "Exhibition runner’s raw UCI logging and clock handling"
[13]: file:///home/ubuntu/Unchessed-UCI-Engine/scripts/sprt-history/sprt_persona_smooth_detect.sh "Existing same-binary candidate/baseline SPRT configuration"
[14]: https://manpages.ubuntu.com/manpages/xenial/man6/cutechess-cli.6.html "cutechess-cli `-debug` documents engine I/O display"
[15]: file:///home/ubuntu/Unchessed-UCI-Engine/rust-toolchain.toml "Repository requests stable Rust; local Cargo 1.75 lockfile incompatibility was observed during this investigation"
[16]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/test_sprt_persona_smooth_detect.py "Structural harness test; no cutechess execution"
