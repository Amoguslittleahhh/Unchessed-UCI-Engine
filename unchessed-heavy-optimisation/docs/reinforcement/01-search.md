# Classic alpha-beta search tuning — evidence-first investigation

**Investigation ID:** 01-search  
**Repository / branch:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/rustc-bootstrap-trial` (`818ef9dd5bb7be64fd6085f7c1910b953390da6e`)  
**Scope:** `unchessed-core/src/search.rs`, directly related UCI/TT/SEE code, tests, scripts, and documentation.  
**Disposition:** No tracked repository file was changed. No training, self-play, SPSA, or SPRT was run.

## Summary

The engine already contains a credible classic alpha-beta baseline: iterative deepening, PVS, quiescence, TT probing/storage, null-move pruning, reverse and per-move futility, ProbCut, check extensions, LMR, killers/history, SEE ordering, aspiration windows, and bounded repetition detection. Its **13 numeric search parameters plus the default-off ProbCut SEE filter are already exposed through UCI**, and the present defaults are internally consistent across Rust defaults, UCI advertisement, and UCI clamps. That is valuable tuning plumbing, but it is **not strength evidence**: the calibration audit classifies all classic-search constants as hand-picked / untested.

The most concrete safe improvement found is a **node-limit correctness fix**. `go nodes N` is only examined every 2,048 nodes because node limits share the periodic stop/time poll. Therefore a request such as `go nodes 1` can execute to the first 2,048-node poll before aborting. This affects only explicitly node-limited searches; ordinary searches (`node_limit = None`) retain their current control flow. It is measurable, testable, and should be treated as a correctness/contract improvement—not an Elo claim.

The best next tuning enabler is **opt-in search telemetry**, followed by exposing the currently hard-coded **reverse-futility depth ceiling** alongside the already tunable RFP margin. Both can preserve existing runtime behavior by default. Any actual pruning activation, change to a live default, or strength claim remains blocked on a real paired-game SPRT; the local checkout does contain the shipped NNUE checkpoint, so **data/checkpoint availability is not the blocker**.

## Evidence

### Implemented search and existing parameterization

| Finding | Direct evidence | Interpretation |
|---|---|---|
| The classic alpha-beta machinery is already substantial. | [`search.rs:483–825`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L483-L825) implements negamax/PVS, TT, RFP, NMP, ProbCut, per-move futility, check extension, LMR, killers/history, and TT writes; [`search.rs:377–480`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L377-L480) implements quiescence with SEE and delta pruning; [`search.rs:992–1080`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L992-L1080) implements aspiration re-search. | Do not spend a round re-implementing aspiration windows, futility, or ProbCut: the backlog’s older “check whether it exists” entries are stale relative to the branch. |
| Search defaults are deliberately UCI-configurable. | [`SearchParams`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L17-L82) contains 13 numeric tuning fields and `probcut_see_filter`; UCI advertises them at [`uci.rs:274–291`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/uci.rs#L274-L291) and clamps them in its `setoption` handler. | The plumbing needed to run controlled baseline-vs-candidate searches is present. Defaults must not be interpreted as tuned merely because they are exposed. |
| Configuration drift is currently guarded. | `python3 tools/check_search_param_consistency.py --repo .` completed with **24 options checked, 0 drifts**; `pytest -q tools/test_search_param_consistency.py` completed with **12 passed**. The checker verifies default, advertised range, and handler clamp for each numeric knob, plus default-off check options. | Any added UCI tuning control should extend this checker and its negative controls in the same change. |
| Existing documentation explicitly calls the constants hand-picked. | [`docs/parameter-calibration-audit.md:31–58`](file:///home/ubuntu/Unchessed-UCI-Engine/docs/parameter-calibration-audit.md#L31-L58) classifies `SearchParams` as “hand-picked default” and says the tuning run has never been executed. | There is no basis in this checkout to claim a change in RFP/NMP/LMR/aspiration/ProbCut/futility values gains strength. |
| A prior futility correctness fix demonstrates why fail-soft behavior needs regression coverage. | [`search.rs:727–740`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L727-L740) raises `best` to the futility floor before pruning; the history for `6b81bb3` describes the prior overly pessimistic TT score and reports a historical real SPRT for the bundle of fixes. | Preserve this floor in any refactor. Do not use that historical bundled result as validation for new pruning changes. |

### Concrete node-limit correctness evidence

`Searcher::check_limits` tests **all** stop conditions only when `self.nodes & 2047 == 0` ([`search.rs:269–288`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L269-L288)). `self.nodes` is incremented on entering negamax and qsearch ([`search.rs:377–381`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L377-L381), [`search.rs:494–519`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L494-L519)). Thus `node_limit` is not a per-node limit: when `N` is positive, the abort is not observed until the first multiple of 2,048 at or above `N`.

A release UCI diagnostic used a clean process, `Threads=1`, `Adaptive=false`, `OwnBook=false`, `Hash=16`, start position, and fixed `go nodes N`. The table records the last **completed iterative-deepening** `info` line—intentionally not a claim that this was the exact internal node total after the interrupted next iteration.

| Requested `go nodes` | Last completed `info` nodes | Completed depth | Returned move | Source-level implication |
|---:|---:|---:|---|---|
| 1 | 1,884 | 4 | `d2d4` | Search entered the next iteration and could not observe the request until node 2,048. |
| 24 | 1,884 | 4 | `d2d4` | Same first-poll behavior. |
| 500 | 1,884 | 4 | `d2d4` | Same first-poll behavior. |
| 2,048 | 1,884 | 4 | `d2d4` | Poll occurs at 2,048; completed-depth reporting remains at 1,884. |
| 4,096 | 1,884 | 4 | `d2d4` | Search reaches the second poll during depth 5 before returning the depth-4 snapshot. |
| 4,097 | 4,555 | 5 | `e2e4` | The next possible node-limit abort is at 6,144; depth 5 completes first. |

The raw transcripts and sweep are preserved as investigator scratch artifacts:

- `/home/ubuntu/jobs/job_c6EDrcuw_a0/node-limit-sweep.tsv`
- `/home/ubuntu/jobs/job_c6EDrcuw_a0/nodes-1.txt`
- `/home/ubuntu/jobs/job_c6EDrcuw_a0/nodes-2048.txt`

This matters in two reachable paths, not only in an artificial UCI command. `Limits` explicitly treats a node-limited request as game mode ([`search.rs:113–123`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L113-L123)), and the adaptive probe uses `depth: 6, nodes: 25_000` ([`uci.rs:1492–1513`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/uci.rs#L1492-L1513)). Correcting it will make node-limited probes honor 25,000 rather than the next 2,048-node boundary; this is an intentional behavioral correction for that explicit limit, not a default search-tree change.

### Existing test coverage and the gaps relevant to tuning

The focused Rust command `cargo test -p unchessed-core search::tests` completed with **21 passed, 0 failed** on local stable `rustc/cargo 1.98.0`. It covers bounded repetition, quiescence versus static evaluation, mate fixtures, hostile root hints, legal-only/forced-move cases, basic hanging material, time budgets, and MultiPV. The broader safety assets include seven unique mate-in-one fixtures in [`benchmarks/matetrack.epd`](file:///home/ubuntu/Unchessed-UCI-Engine/benchmarks/matetrack.epd), exercised by [`matetrack_suite_finds_every_forced_mate`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L1296-L1341).

The coverage does **not** currently include:

1. An assertion that `go nodes N` stops at `N` (or a documented bounded overshoot), despite node limits being a public UCI input and an internal probe budget.
2. Per-feature counters for TT hits/cutoffs, RFP/NMP/ProbCut/futility actions, LMR reductions/re-searches, quiescence pruning, or aspiration fail-high/fail-low cycles. The current `InfoEvent` only exposes depth, score, cumulative nodes, time, hashfull, and PV ([`search.rs:195–204`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L195-L204)).
3. Differential tests asserting that a newly exposed parameter at its legacy default produces identical fixed-depth results and node totals against the old hard-coded condition.
4. Tactical regression fixtures designed specifically for pruning boundaries (zugzwang/null-move, narrow-window RFP/ProbCut, late quiet futility/LMR). The current mate suite is necessary but does not establish all pruning semantics.

### Small local diagnostics (not Elo measurements)

A release adapter build completed successfully. With the same one-thread/no-book HCE setup, fixed depth 6 from start position completed at **9,232 nodes in 5 ms**, and a normal opening tabiya completed depth 7 at **38,084 nodes in 30 ms**. These runs are only sanity baselines for a future deterministic harness; the host, evaluator, tiny 16 MB hash, and millisecond rounding make them unsuitable for ranking candidate parameters or inferring strength.

The default executable-directory lookup initially selected HCE because the checkout’s NNUE file was not next to `target/release/unchessed-adapter`. This is not a missing-checkpoint problem: explicit `setoption name EvalFile value /home/ubuntu/Unchessed-UCI-Engine/unchessed-nnue.bin` succeeded, and a small `go nodes 4097` diagnostic reached depth 5 and returned `e2e4`. The checkpoint is present at `unchessed-nnue.bin` (**23,071,768 bytes**). Any fixed-position regression/telemetry suite can therefore run both HCE and shipped-NNUE variants locally.

## Recommended changes

### Implemented-ready: correctness and measurement, with old default behavior preserved

#### 1. Honor `go nodes N` exactly (or at most one node beyond it)

**Change.** Split node-limit checking from the 2,048-node stop/time polling. Immediately after incrementing the existing node counter, test `node_limit` and set `abort` if `nodes >= limit`; retain the current 2,048-node cadence for atomic `stop` and wall-clock checks. In outline:

```rust
if let Some(limit) = self.node_limit {
    if self.nodes >= limit {
        self.abort = true;
        return;
    }
}
if self.nodes & 2047 != 0 { return; }
// existing stop and hard-time checks
```

**Why this is safe to ship as a correctness change.** It changes only searches that explicitly provide `Limits.nodes`; when it is `None`, the old periodic stop/time path is retained. It does not modify alpha-beta bounds, move ordering, pruning criteria, or a no-node-limit engine game. It should not be advertised as stronger or faster; the intended metric is exact budget adherence.

**Required tests.** Add a private `Searcher` test that sets `node_limit: Some(1)` and verifies `abort` occurs at one visited node; repeat at boundaries `1`, `2,047`, `2,048`, `2,049`, and an ordinary value such as `25,000`. Add a public `go` test that a node-limited abort still returns a legal fallback line rather than an empty response. If telemetry from recommendation 2 lands, assert `visited_nodes <= requested_limit` through the public API instead of relying only on private state.

**Caveat.** The adapter’s 25,000-node persona probe will become stricter by up to 2,047 nodes. That is correct for the stated budget, but it may change adaptive-mode decisions under that explicit capped probe. Test it for legality and response shape; it needs SPRT only if anyone proposes treating its result as a strength improvement or altering a live default based on it.

#### 2. Add opt-in, invariant-checked search telemetry before changing a tuning value

**Change.** Add an opt-in `SearchStats` snapshot (ideally a library API such as `go_with_stats` plus a UCI `SearchStats` check option defaulting to `false`) that reports, at minimum: total main/qsearch nodes; TT probes/hits/cutoffs; RFP prunes; null attempts/cutoffs; ProbCut attempts/cutoffs; per-move futility skips; LMR reductions/full-depth re-searches; quiescence SEE and delta skips; and aspiration fail-low/fail-high/re-search counts. Emit it as a clearly labeled UCI `info string` only when the option is on, or record JSON/TSV in a purpose-built fixed-position harness.

**Default-preservation rule.** `SearchStats=false` must retain the old output and tree. Because a branch on every counter increment can itself perturb nodes per second, benchmark the disabled path on a fixed FEN corpus before accepting the instrumentation. If that is material, prefer a compile-time feature or separate benchmark entry point rather than silently taxing normal engine play.

**Required tests.** Pin accounting invariants, including `tt_cutoffs <= tt_hits <= tt_probes`, `null_cutoffs <= null_attempts`, `probcut_cutoffs <= probcut_attempts`, and `qnodes + main_nodes == visited_nodes` under a documented definition. Exercise at least start position, the existing tabiya, the hanging-queen fixture, a mate fixture, and a zugzwang/null-move-sensitive fixture. Verify disabled telemetry gives the same fixed-depth line, score, PV, and node total as the legacy call using fresh TTs.

**Why it is measurable now.** It needs neither additional data nor a new checkpoint; HCE and the shipped NNUE checkpoint are locally usable. Telemetry can identify whether a candidate merely trades node categories or actually reduces work at a fixed depth. It cannot establish Elo.

#### 3. Parameterize the reverse-futility depth ceiling with its legacy default

**Change.** Add `rfp_max_depth: i32` to `SearchParams`, set its default to **6**, use it in place of the hard-coded `depth <= 6` condition at [`search.rs:539–545`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L539-L545), and expose an `RFPMaxDepth` UCI spin (suggested range `0..12`, where zero disables RFP). Extend `tools/check_search_param_consistency.py` and its mutation tests so struct default, advertisement, and clamp stay locked.

This is the narrowest useful parameterization because the RFP margin is already variable while its reach is not. The calibration audit independently identified that ceiling as a hard-coded control adjacent to the existing tunables ([`parameter-calibration-audit.md:177–181`](file:///home/ubuntu/Unchessed-UCI-Engine/docs/parameter-calibration-audit.md#L177-L181)). With default six, old search behavior is preserved exactly; alternative values must remain candidates, not new defaults.

**Required tests.** Add a default-value/advertisement/clamp test through the existing consistency checker and a differential fixed-depth fixture set showing `SearchParams::default()` and an explicit `rfp_max_depth: 6` have equal result/PV/node count from clean TT states under HCE and the shipped NNUE. Confirm RFP never fires in a PV node, in check, beyond the configured cap, or with mate-range beta; those safety guards must be unaffected.

### Requires a real SPRT before any default or strength conclusion

| Candidate / action | Current evidence | What is still required |
|---|---|---|
| Change any existing RFP, NMP, LMR, aspiration, ProbCut, or futility default. | The values are UCI-configurable but documented as hand-picked. No current SPRT result isolates a search-parameter change. | One candidate at a time, a real baseline-vs-candidate paired-game SPRT with `Threads=1`, fixed hash, book, and current evaluator. Use the project’s stated `elo0=0`, `elo1=5`, `alpha=beta=0.05` convention; re-baseline only after an accepted candidate. |
| Enable `ProbcutSeeFilter` by default. | It is intentionally **false**; its source comment says it changes the tree and needs a paired-game SPRT ([`search.rs:53–61`](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/search.rs#L53-L61)). | First use telemetry to quantify losing-capture skips and add tactical regressions. Then run both the standard `tc=5+0.05` and a conservative longer control, as the calibration audit’s Stage 2 prescribes. |
| Tune a newly exposed `RFPMaxDepth`. | Parameterization with default six is behavior-preserving, but values other than six alter pruning. | A real SPRT for each candidate that survives deterministic correctness and fixed-depth node diagnostics. No imported Stockfish/RubiChess magic number is valid evidence: their eval scale and search differ. |
| Add razoring, late-move pruning, singular extensions, IID, or multi-cut pruning. | The backlog lists them as research ideas, not validated improvements. Current tests do not target their boundary failures. | A design review, tactical/correctness fixtures, telemetry, and a separate real SPRT for each feature. These are new tree-changing pruning/extensions, not safe micro-tuning. |

The relevant external-SPRT infrastructure is not present in this sandbox: no `cutechess-cli` was found on `PATH`, and the project’s real harness refers to reviewer-local cutechess/book paths ([`scripts/research/wsl_sprt_persona_run.sh:3–21`](file:///home/ubuntu/Unchessed-UCI-Engine/scripts/research/wsl_sprt_persona_run.sh#L3-L21)). The calibration audit specifies the intended protocol and the need for real reviewer hardware ([`parameter-calibration-audit.md:185–214`](file:///home/ubuntu/Unchessed-UCI-Engine/docs/parameter-calibration-audit.md#L185-L214)). This is a **real-SPRT infrastructure blocker**, not missing NNUE data or checkpoints.

## Verification record

| Check | Result | Meaning / limitation |
|---|---|---|
| `cargo test -p unchessed-core search::tests` | **21 passed, 0 failed** (Rust 1.98.0). | Validates the existing focused test suite only; it is not an Elo test. |
| `cargo build -p unchessed-adapter --release` | **Passed**. | Enables the local UCI diagnostics; not a performance comparison. |
| `python3 tools/check_search_param_consistency.py --repo .` | **24 options checked, 0 drifts**. | Confirms configuration synchronization, not tuning quality. |
| `pytest -q tools/test_search_param_consistency.py` | **12 passed**. | Confirms the drift checker’s current mutation controls. |
| `python3 tools/rust_bracket_check.py --all` | **21 Rust files balanced**. | Supplementary syntax hygiene only. |
| Release UCI fixed-depth diagnostics | Startpos HCE depth 6: 9,232 nodes / 5 ms; tabiya HCE depth 7: 38,084 nodes / 30 ms. | Host-specific sanity observations, explicitly not benchmarks or strength evidence. |
| Explicit NNUE load | `setoption EvalFile .../unchessed-nnue.bin` succeeded; small node-limited diagnostic completed. | Confirms the shipped checkpoint is available for deterministic local search tests. |
| Real SPRT / tuning run | **Not run.** | No Cutechess/book setup on this sandbox; no strength claim is made. |

## Decision boundary

> **Implemented-ready now:** exact `go nodes` enforcement, opt-in/invariant-checked telemetry, and legacy-default parameterization of the RFP depth ceiling. These changes are useful because they improve correctness or observability while preserving ordinary default search behavior.
>
> **Not ready to call stronger:** every changed pruning threshold, every newly enabled pruning rule—including `ProbcutSeeFilter`—and every new alpha-beta heuristic. Deterministic tests and node measurements narrow risk; only a real paired-game SPRT can support a strength/default decision.
>
> **Not blocked on data/checkpoints:** the shipped NNUE file exists and loads explicitly. **Blocked on real-SPRT infrastructure:** this sandbox lacks the configured cutechess/book environment, so no local result can replace that gate.

## Structured fields

| Field | Value |
|---|---|
| `id` | `01-search` |
| `topic` | `Classic alpha-beta search tuning` |
| `summary` | `Existing alpha-beta tuning knobs are exposed but not strength-validated. Exact node-limit enforcement, default-off telemetry, and legacy-default RFP-depth parameterization are implementation-ready; tuning values/default flips require real SPRT.` |
| `evidence` | `Search source, UCI option/clamp checker, focused Rust tests, release UCI node-limit sweep, NNUE explicit-load check, and real-SPRT harness inspection.` |
| `recommended_changes` | `Exact node-limit polling; opt-in SearchStats plus invariants; RFPMaxDepth defaulting to 6 with consistency/differential checks.` |
| `verification` | `21 focused Rust tests passed; release adapter built; 24-option consistency check passed; 12 Python checker tests passed; no SPRT/training run.` |
| `report_file` | `/home/ubuntu/reinforcement_reports/01-search.md` |
