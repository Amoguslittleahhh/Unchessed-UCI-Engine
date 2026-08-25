# Engineering request: make the Unarchitectured v1 runtime forward pass fast enough to actually use

## Scope note

Unarchitectured v1 is the canonical, current architecture — the target to
build on and improve. Its predecessors (Apex v1, Hydra v1 through v4, and
any earlier lineage) were beta-testing iterations superseded by it; this
work is not an invitation to revert to or resume development on any of
them. Any follow-on architecture or training work belongs on top of
Unarchitectured v1, not as a parallel track exploring an earlier one.

## Status update — round 9

Your seventh pass — eleven commits from `ce3b6a0` through `4e3b623`,
plus the history-questions reconciliation — was reviewed, independently
verified, and merged onto `main` at `9cea2f4`. This is the largest single
batch reviewed so far, and it's genuinely strong work: real causal
explanations for round 7's SPRT results, real capability gaps found in the
model itself, and one real test bug caught and fixed along the way.

**Verified independently, not just trusted:**

- Full workspace build clean, `unchessed-core` 104/104 (after one fix, see
  below), `tools/` Python suite 187/187.
- **Reproduced three of the headline numeric claims from scratch**, not
  just read the committed JSON: `analyse_gab_contribution.py` gave
  baseline 0.2683 → GAB-zeroed 0.2100 (−0.0583, the "21.7% of top-1"
  figure) exactly; `analyse_unarchitectured_v1_ordering_risk.py` gave
  top-1 0.2683 (neural) vs. 0.0683 (movegen-order baseline) and mean
  regret 146.3cp vs. 290.5cp, matching the commit message; and the
  budget-percentage arithmetic (9.72ms / 108ms = 9.0%, /66ms = 14.7%,
  /2450ms = 0.4%) checks out exactly against round 7's own measured Elo
  trend.
- `tools/test_pentanomial_sprt.py` has a dedicated `ReproducesRound7Tests`
  class that validates the new SPRT tool's math against the reviewer's own
  real round-7 numbers (-26.1, -15.1, -5.8 Elo) — ran it, passes. Using
  the actual data I produced to validate your own tooling, rather than
  asking me to trust it, is exactly right.
- **Found and fixed one real test bug**, caught the same way this whole
  process has caught every prior one — by running it, not reading it.
  `unarchitectured_candidate_produces_exact_real_root_hints` (from
  "Harden root-hint pairing") asserted that a request for a different
  (truncated) move list must time out. It doesn't, reliably: real
  inference at this exit is single-digit milliseconds, well inside the
  100ms wait window, so the fresh (correctly-sized) request usually
  completes and returns `source="exact"` — which the old assertion
  treated as reusing a stale cached hint, when it's actually the cache
  correctly *not* reusing anything and computing something new and
  correct instead. Confirmed with diagnostic instrumentation
  (`source=exact, hints_len=19, truncated_len=19`) before rewriting the
  assertion to check the real safety property (never return a mismatched
  stale result) instead of an implementation detail of host speed. Your
  own commit message for that round flagged an almost-identical near-miss
  in your first attempt at this exact test — this is the same class of
  mistake surfacing once more, worth noting as a pattern: assumptions
  about *timing* on this fast a system are the recurring soft spot, not
  the logic itself.

**The headline findings, now on `main` and worth treating as load-bearing
context for any future work on this model:**

1. **Why the hint costs Elo, mechanistically** (not just "it's negative"):
   the policy is a genuinely *better* first-move orderer than anything
   free (top-1 0.2683 vs. 0.0683 movegen order, wins every metric) — the
   problem is purely structural. Benefit lands once, on the cheapest
   search pass; cost is charged to the move budget every time regardless.
   The budget-percentage arithmetic alone reproduces round 7's monotonic
   Elo trend from nothing but `Limits::budget` math.
2. **GAB is real load-bearing capacity, and it's underprovisioned**: this
   project's own GAB config (`d1=8, d2=32, d3=32`) is a quarter of the
   comparably-scaled paper config's smallest setting, and zeroing it costs
   21.7% of top-1 — more than half the model's entire margin over free
   MVV-LVA.
3. **Rating conditioning is confirmed genuinely inert** (0/200 moves
   change across a 600→3200 sweep, deltas 2-3 orders of magnitude below
   the gaps between candidate moves) — this closes the exact gap I
   flagged as unverifiable in the history-questions response, and both the
   oracle and student share identical conditioning code, so if the oracle
   turns out to have the same defect, it's architectural; if not, it's a
   training-time bug. That specific question (oracle-side) remains open —
   the oracle checkpoint isn't in this repo.
4. **The policy's magnitude, not just its ranking, carries real
   information**: ECE 0.0048, and correct-vs-wrong confidence separation
   survives a uniform-baseline normalization control (6.21x vs 4.26x). A
   positive result, and correctly guarded against looking good for
   uninteresting reasons (majority-bin domination) before being reported
   as one.
5. **Our weights exceed the int8 representable range by 2.06x** with no
   weight clipping in the trainer — a real, quantitative explanation for
   the earlier rejected int8-activation prototype's parity failure, not
   just "it didn't work."
6. **A matetrack-style regression suite now exists** (7 positions, 5
   mating patterns, both colors) — catches a failure mode SPRT is
   structurally bad at (a specific forced win silently breaking) that
   nothing in this project covered before.

None of this reverses the standing conclusion — every one of these
findings is explicitly framed in your own commits as explaining or
refining round 7's negative result, never overturning it.
`UnarchitecturedHint` stays default-off, `runtime_safety_suite` stays
false, and this round's work makes both defaults better-justified, not
closer to flipping.

## Status update — round 8

Your sixth pass (`fac25a5` "Add dev environment setup: requirements file
and Rust bracket checker", `bd80e1a` "Close the last named safety gap:
real-checkpoint hint disagreements") was reviewed, independently
re-verified, and merged onto `main` at `ffffc29`. This directly answers the
one item round 7 left explicitly open: a hint-vs-correct-move disagreement
test using the real checkpoint on a real position, not a synthetic
adversarial ranking.

Verified independently:

- Full workspace build clean (this closes the loop on something you
  couldn't check yourselves — you'd never compiled these two new Rust
  tests, correctly disclosed as "NOT COMPILED" in the commit message).
  `unchessed-core` 95/95, `tools/` Python suite 59/59, both matching your
  claimed counts exactly.
- **Regenerated `tools/find_unarchitectured_v1_hint_disagreements.py`'s
  report from the real exported checkpoint independently** (not just
  trusted the committed JSON) — every logit, rank, and gap value in the
  output matched what's transcribed into `search.rs` and the committed
  artifact exactly, including the headline finding: the real checkpoint
  ranks a forced back-rank mate **10th of 17** legal moves.
- The generator-bug catch in your own commit message (`queen_sac_smothered`
  wasn't actually mate — `Rxe8` refutes it) is real and matches what the
  regenerated tool reports now: it's gone, replaced by
  `back_rank_mate_with_own_pawns`, and the tool's "derive every mate claim
  itself" fix (playing the move, checking mate, separately enumerating all
  mates so a claimed-unique mate can't quietly have two) is exactly the
  kind of self-verification this whole process has been asking for.
- The two new Rust tests do what they claim: `real_checkpoint_ranking_
  cannot_suppress_back_rank_mate` includes its own "sanity-check the
  premise" assertion (fails loudly if the model ever stops disagreeing,
  so the fixture can't silently rot into testing nothing) — genuinely good
  practice, not something asked for.
- `tools/rust_bracket_check.py --all` runs clean against all 21 tracked
  `.rs` files, and the dev-environment tooling (`requirements-dev.txt`,
  documented rationale for not vendoring wheels or a Rust toolchain) is a
  reasonable, well-justified addition — solves a real problem (work
  redone from scratch each session because it lived in ephemeral `/tmp`)
  without overreaching into something this project doesn't need.

**This closes the loop on everything round 7 named as open.** The safety
suite now covers: adversarial synthetic hints (round 5), broader
attacker-type/mate-pattern coverage (round 7's own safety-suite work,
knight/bishop/queen checks and a second mate pattern), and now real-model
disagreement (this round). `runtime_safety_suite` correctly stays `false`
in the capability manifest — your own commit message says it plainly: this
closes the last *named* gap in the suite, not the flag, and flipping the
flag would wrongly imply the runtime is cleared for integration when round
7's SPRT evidence says the opposite.

**Nothing here changes the standing conclusion**: `UnarchitecturedHint`
stays default-off. No configuration tested across three real SPRT batches
(round 7) ever trended positive. This round adds real safety-net depth, not
a reason to revisit that.

## Status update — round 7

This round was done directly by the project owner and reviewer on real
deployment hardware (a Core Ultra 9 285H with a working local Rust
toolchain, WSL2/Ubuntu, and a real `cutechess-cli` + opening book already
present from this project's earlier SPRT history), not by arena. It answers
round 6's items 2 and 5 (deployment-CPU benchmark, SPRT) directly, plus
fixes a real bug found along the way.

**Bug found and fixed first: the inference-thread default was actively
harmful.** `aegis_v4_runtime.rs`'s internal matmul parallelism
(`UNCHESSED_INFERENCE_THREADS`) spawns fresh OS threads per call via
`std::thread::scope` rather than reusing a pool, and one forward pass makes
dozens of such calls across 8 layers — that spawn/join cost was never
amortized. Measured directly on this real hardware, sweeping 1-8 threads:
**1 thread (sequential) was fastest at 7.92ms, and every higher count was
monotonically slower**, up to 16.42ms at 8 threads. The previous default
(`available_parallelism().min(4)`) measured 11.01ms — already worse than no
internal splitting at all. Fixed by defaulting to 1
(`5d5a36e`, already pushed to `main`); full 8/256 exit went from 12.67ms to
9.72ms on this machine as a result. A regression test
(`inference_threads_defaults_to_sequential`) locks the default in.

**Also measured real NPU dispatch latency** on this machine's actual `Intel
AI Boost` NPU via OpenVINO (`4ab1c20`, also pushed) — not requested by this
prompt, but relevant context: real NPU dispatch is ~0.3-0.6ms (lower than
the doc's old ~1ms literature guess), CPU still wins by 11-20x at
NNUE/small-model scale, and the gap narrows to ~1.5x at ~4.2M-param scale.
Doesn't change any conclusion here, but `docs/npu-viability-285h.md` now
has measured numbers instead of estimates throughout, including a corrected
forward-pass figure (12.67ms measured vs. the doc's old "~3-4ms est.").

**Then, on the corrected build, a real (non-formal) SPRT-style batch was
run** using the exact config `scripts/sprt-history/sprt_unarchitectured_v1_hint.sh`
already specifies (`tc=5+0.05`, `UnarchitecturedMinTime=1000`, `elo0=0
elo1=5 alpha=0.05 beta=0.05`), against the real exported checkpoint, same
binary for both sides (`option.UnarchitecturedHint=true/false`):

- A 20-game smoke test first (matching round 0's original scale, same
  discipline that caught round 0's catastrophic failure): 0.425 score,
  **not** a collapse like round 0's 0-20-0 — cleared the cheap sanity check.
- A 600-game batch followed (300 rounds, concurrency 12): **Hint scored
  172-217-211 (0.463) against Baseline. Elo difference -26.1 ± 22.4 (95% CI
  roughly [-48.5, -3.7]). LOS 1.1%. SPRT llr -1.1**, trending toward the
  reject bound (-2.94) but not formally crossed.

**A data-integrity check on that first batch, done honestly rather than
skipped:** the model path used was `/tmp/model.unarchv1`, and a later,
unrelated `wsl` session found that file missing — worth investigating
before trusting the result, since a silently-failed model load would make
`Hint` behaviorally identical to `Baseline` and turn the whole test into
noise, not a real measurement. Root cause: `/tmp` in this environment is
tmpfs-backed and gets wiped whenever the WSL2 VM cycles between separate
invocations, but the smoke test and the 600-game batch each ran as one
continuous process from a single script invocation, so the VM never went
idle mid-run — the file should have persisted throughout. To resolve this
with certainty rather than trust the reasoning, **the entire 600-game batch
was replicated** with the model referenced from a stable path inside the
repo checkout instead of `/tmp` (eliminating the risk entirely), same
config, same book, freshly rebuilt binary:

- **Replication (600 games): Hint scored 184-210-206 (0.478) against
  Baseline. Elo difference -15.1 ± 22.5 (95% CI roughly [-37.6, +7.4], now
  includes zero). LOS 9.5%. SPRT llr -0.665**, well short of either bound —
  not significant on its own.

**Both runs point the same direction — negative, never positive — across
1,200 total games, which is what actually matters here.** The replication's
smaller magnitude and zero-inclusive CI are consistent with ordinary
run-to-run variance for an effect this size at this sample size, not with
the original result being a measurement artifact: a genuinely broken/inert
`Hint` (identical to `Baseline`) would have produced a result centered on
0.500 with no consistent direction, not two independent 600-game samples
that both landed 2-3.5 points below it. **This is not round 0's
catastrophic failure, but the combined evidence is a real, if modest,
strength regression — not neutral, and never positive in either run.**

**A live depth/time calibration was also run** (new tool,
`tools/unarchitectured_v1_depth_time_calibration.py` — this is the
"integrated depth/NPS across a real position set" item from round 6's
checklist, now actually done) across 3 positions and 8 remaining-clock
values from 500ms to 120s, comparing achieved search depth and wall time
with the hint on vs. off. **Depth loss from the hint was negligible: mostly
0, occasionally +1 (hint reaches deeper), rarely -1**, and the hint's own
charged inference cost was tiny (0-2ms) at this exit. This means the
**"clock tax reduces depth" mechanism proposed in the first version of this
status update is not well supported** — the regression's actual cause is
still unexplained by anything measured so far. It may be the move-ordering
itself occasionally steering the first pass toward a worse line that costs
more to refute even at equal nominal depth, or it may need more than 1,200
games to characterize properly. Either is a legitimate answer; don't treat
the earlier "clock tax" explanation as settled — it wasn't well-supported by
this round's own follow-up data.

**The conservative-threshold retest was also run, closing this out for
now.** Same reviewer, same real hardware, same binary, same book. Since
`UnarchitecturedMinTime=30000` can never fire at `tc=5+0.05` (base time is
only 5000ms, so `wtime`/`btime` can never reach 30000ms), the time control
had to change too — `tc=60+0.6`, 300 games (150 rounds), same
`elo0=0 elo1=5 alpha=0.05 beta=0.05` SPRT bounds:

- **Conservative config: Hint scored 72-77-151 (0.492) against Baseline.
  Elo difference -5.8 ± 27.7 (95% CI roughly [-33.5, +21.9], comfortably
  includes zero). LOS 34.1%. SPRT llr -0.207**, far from either bound —
  genuinely inconclusive/neutral, not a lean in either direction.

**The three real batches form a clean, monotonic trend:**

| Config | Games | Elo | LOS |
|---|---:|---|---:|
| `MinTime=1000`, `tc=5+0.05` (original) | 600 | -26.1 ± 22.4 | 1.1% |
| `MinTime=1000`, `tc=5+0.05` (replication) | 600 | -15.1 ± 22.5 | 9.5% |
| `MinTime=30000`, `tc=60+0.6` (conservative) | 300 | -5.8 ± 27.7 | 34.1% |

Paying the tax less often makes the harm shrink toward indistinguishable
from zero. **Caveat, stated plainly rather than glossed over: the
conservative retest necessarily changed both `UnarchitecturedMinTime` *and*
the time control at once** (there was no way to isolate just one variable,
since the conservative threshold is physically unreachable at the original
fast time control) — so this doesn't cleanly prove *which* change removed
the harm, only that the actual shipped configuration (which pairs both)
shows no detectable regression across 300 games.

**Conclusion for round 8: this is a reasonable stopping point, not a green
light.** The default-shipped config (`UnarchitecturedMinTime=30000`) shows
no measurable harm — that's a real, useful finding. But nothing here ever
trended positive at any config, so there is still no evidence this feature
helps, only evidence that at the conservative setting it no longer
measurably hurts. That is not sufficient to enable it by default; it would
take a real positive result to justify that, which doesn't exist. If
there's appetite for more, a genuinely isolated test (same time control,
only `UnarchitecturedMinTime` varied) would cleanly separate the two
confounded variables above — but that's optional polish, not a blocker.

**Do not enable `UnarchitecturedHint` by default under any of this.**
`main`'s default remains `false`, no code changed to enable it, and none of
this round's results changes that.

## Status update — round 6

Your fifth pass (`2b3677a` "Wire default-off Unarchitectured v1 UCI
candidate") was reviewed, independently re-verified, and merged onto `main`
at `d0e5666`. This closes step 1 of round 5's list — it's the first commit
since round 0 to actually touch the live UCI path, so it got the most
thorough verification of any round yet: reading the diff, running the unit
tests, and separately driving the real compiled binary by hand over stdin.

Verified independently, not just trusted:

- Read the full `uci.rs` diff line by line: `unarchitectured_hint` defaults
  to `false`, is only ever set `true` inside the `setoption` handler for
  exactly `UnarchitecturedHint value true`, and every other code path
  (`ucinewgame` with the option still off, a failed model load) resets it
  to `false` and logs why rather than silently leaving stale state.
- **Ran the actual compiled `unchessed-adapter.exe` over stdin, three ways,
  on the reviewer's machine:**
  - No `setoption` at all: zero `Unarchitectured` info strings, identical
    `bestmove` behavior to before this round.
  - Enabled, real checkpoint, `wtime`/`btime` 5000ms (below the 30s
    `UnarchitecturedMinTime` default): `info string [Unchessed]
    Unarchitectured hint skipped-low-time actions=0 charged=0ms` — the
    exact fix for round 0's root cause, confirmed live, not just in a test.
  - Enabled, real checkpoint, 60000ms clock: `info string [Unchessed]
    Unarchitectured hint exact actions=20 charged=3ms`, legitimate
    `bestmove` returned.
  - Also ran their own `tools/smoke_unarchitectured_v1_uci.py` against the
    real exported checkpoint — passed.
- Grepped every construction site again on the merged tree: still zero
  references to the hint worker outside the gated `setoption`/`go` paths
  and tests.
- The new safety tests (adversarial back-rank mate for both colors, a
  stalemate position ignoring a hostile fake hint, an out-of-range/NaN hint
  entry that can't corrupt legal root-move handling) pass and, read
  individually, actually test the scenario their names claim.
- Full workspace build clean, `unchessed-core` 82/82, `tools/` Python suite
  25/25.
- The SPRT launcher script added this round
  (`scripts/sprt-history/sprt_unarchitectured_v1_hint.sh`) is correctly
  configured to actually stress the risk, not pass trivially: it sets
  `UnarchitecturedMinTime=1000` (the floor, not the comfortable 30s
  default) at `tc=5+0.05` — the same fast time control that caught round
  0's failure — so a real run of this script would be a genuine test, not
  a softball. It wasn't run (no `cutechess-cli` in either sandbox), which
  was disclosed honestly rather than faked.

**Step 1 is done and verified. Step 2 — provenance-disjoint calibration —
is now the clear next blocker, and everything else below still stands.**

## Status update — round 5

Your fourth pass (`896066e` "Add fail-closed Chessformer root-hint
integration trial", `6115dcf` "Canonicalize Unarchitectured v1 training
scripts") was reviewed, independently re-verified, and merged onto `main`
at `9ced983`. This is the most consequential change reviewed so far — it's
the first one that touches `search.rs` since round 0's catastrophic
failure — so it got the most careful review of any round.

Verified independently, not just trusted:

- `go()`'s signature and behavior are provably unchanged: it calls the new
  `go_with_root_hints` with an empty hint slice, and the root-ordering sort
  only takes the policy-hint branch when `!root_hints.is_empty()` — with no
  hints, it's the exact same `sort_by_key(|r| -r.score)` as before. Checked
  by reading the diff line by line, not by trusting the "fail-closed"
  framing in the commit message.
- Grepped `unchessed-adapter/`, `unchessed-reviewer/`, and `uci.rs` for
  `UnarchitecturedHintWorker::new` and `go_with_root_hints` — zero matches.
  Nothing in either shipped binary or the UCI layer constructs the worker
  or calls the new entry point. It is genuinely unreachable, not just
  claimed to be.
- New tests pass and test what they claim: `root_hints_cannot_override_a_
  forced_mate` (an adversarial hint ranking the mating move dead last still
  finds and plays the mate) and `precharged_root_hints_keep_only_move_
  legal_and_bounded` (a forced-single-legal-move position stays correct
  under a hostile hint and returns well within its time budget).
- `UnarchitecturedHintWorker` is a real nonblocking design: bounded
  one-request `try_send` queue (a full queue reports `accepted=false`
  immediately, never blocks the calling thread), and `latest_exact` only
  returns a result on an exact key match over position hash + legal
  actions + rating + time class + persona + exit — a stale result from a
  different position can't leak into a decision.
- Reproduced their two benchmark claims on the reviewer's machine: the
  fixture-disjoint calibration numbers matched exactly (deterministic —
  fixed positions, fixed weights), and the integrated-search trial
  reproduced the same 7/8 best-move-agreement figure with depth/NPS
  numbers in the same range as reported.
- The `config/a100_hydra_v5_training.json` deletion in the canonicalize
  commit was checked, not just accepted: confirmed it was a genuinely
  stale duplicate (dangling reference to a `a100_hydra_v4_training.json`
  that doesn't exist anywhere in this repo, and a fixed `steps_per_epoch`
  the project's own docs call forbidden) with a functioning canonical
  replacement (`config/unarchitectured_v1_training.json`) already present
  — not silent data loss.
- Full workspace build clean, `unchessed-core` 77/77, `tools/` Python
  suite 25/25.

**This is good, disciplined work, but it does not get us any closer to an
actual SPRT gate on its own** — see the full list below of what's still
missing before one is even possible, not just advisable.

## Status update — round 4

Your third pass (`b556f13` "Reduce Chessformer runtime dispatch and cache
overhead", `3485789` "Make adopted runtime audits standalone") was reviewed,
independently re-benchmarked, and merged onto `main` at `1d6da41`.

Verified independently:

- Full workspace build clean, `unchessed-core` test suite still 74/74 (no
  new tests this round, none removed).
- Reproduced a real speedup on the reviewer's own machine: **13.15ms →
  10.92ms (~1.2x)** on the full 8/256 forward pass, in the same direction as
  your reported ~1.08x (different host, expected — same pattern as every
  prior round).
- The complete integer-vs-dequantized drift gate got *tighter*, not looser:
  max observed component 9.4e-5 this round vs. 5e-4 allowed (round 3 was
  already well inside the bound; this is an improvement in the same
  direction, not a regression risk).
- The `verda_gpu_profile` missing-module issue flagged in round 3's status
  update is fixed: `tools/unarchitectured_v1_architecture_audit.py` now runs
  standalone (`python tools/unarchitectured_v1_architecture_audit.py
  --strict` passes cleanly on the reviewer's machine with no internal-only
  imports), and a new regression test
  (`tools/test_unarchitectured_v1_architecture_audit.py`) locks in that it
  stays that way. Full `tools/` Python suite: 21/21 passing.
- Good discipline continued: you tried the pairwise-summation idea from
  round 2's status update (to tighten the narrow-exit Python tolerance),
  found it didn't materially help, and said so plainly in the rejected-
  experiments section instead of quietly dropping it or claiming it worked.
  Also rejected AVX-512 VNNI (slower — frequency throttling) and a reduced-
  degree exponential approximation (passed parity, regressed latency) for
  the same honest reason: measured, didn't help, said so.

**Speed is hitting diminishing returns, and that changes what's worth
asking for next.** Round 2 was ~5.2x, round 3 was ~1.2-1.4x, round 4 is
~1.2x again but now firmly in cache/dispatch micro-optimization territory.
The full path is around 11-13ms on the reviewer's machine — much better
than the original 89ms, but still not "cheap enough to pay every move," and
another round of kernel tuning is unlikely to close that gap by itself. See
"What's needed" below for the resulting change in priority: this round's
ask leads with the *integration* work, not more raw speed.

## Status update — round 3

Your second pass (`cff6083` "Add retained-int8 Chessformer matrix inference",
`d152fdc` "Document rejected int8 activation prototype") was reviewed,
independently re-benchmarked, and merged onto `main` at `c97dd0b`. This is
exactly the untried angle round 2's status update called out — retaining the
package's symmetric int8 weights instead of dequantizing to `f32` on load,
dynamically quantizing each activation row to int16, and doing the dominant
matmuls as AVX2 i16×i8→i32 integer products.

Verified independently (not just trusted from the commit/doc):

- Full workspace build clean, `unchessed-core` test suite 74/74 (the 71
  pre-existing tests plus the three new ones this round added:
  `integer_microkernels_match_scalar`, `activation_quantizer_matches_scalar`,
  `integer_matrix_path_stays_close_to_dequantized_path`).
- Reproduced the backend speedup on the reviewer's own machine: **1.22x**
  (your sandbox reported 1.23x–1.39x depending on thread count — same
  ballpark, different host, consistent with how round 1's numbers also
  didn't transfer 1:1 across hosts).
- The new `integer_matrix_path_stays_close_to_dequantized_path` gate (int8
  path vs. the existing f32 path, 5e-4 max drift across every exit's logits,
  regret mean/log-scale, evidence, and representation) passed on the
  reviewer's machine with deltas in the same range as reported.
- Good discipline worth continuing: you tried quantizing *activations* to
  int8 too, it failed the existing Python parity gate (about 1.01e-2 drift
  vs. the required 5e-3), and instead of loosening the tolerance to make it
  pass, you reverted it and documented the rejected experiment in
  `docs/unarchitectured-v1-runtime-optimization.md`. Keep doing exactly
  that — a documented rejection is more valuable than a quietly-passing
  gate that got weakened to let broken work through.

Two things fixed or dropped on adoption, not blocking, but worth knowing:

1. `tools/unarchitectured_v1_architecture_audit.py` (also adopted this
   round, minor 4-line change) imports a `verda_gpu_profile` module that
   does not exist anywhere in this repository — it could not be run on the
   reviewer's machine. It's a peripheral Python safety-gate script, not part
   of the Rust runtime work, so it wasn't a blocker for adopting the real
   change, but it means that script has never actually been executed
   end-to-end outside your sandbox. If you touch it again, either commit the
   missing module or drop the import if it's dead weight.
2. The round-1 and round-2 commits on your branch carried a much larger
   README rewrite (new "Architecture summary," "Runtime forward
   performance," "Autonomous fail-closed safety," "Efficient data-centre
   training," "Production NNUE," etc. sections, plus links to
   `config/architecture_registry.json` and
   `docs/unarchitectured-v1-safety-integrity-report.md`) that was never
   adopted onto `main` and doesn't match `main`'s actual current README
   structure — several of those linked files don't exist in this tree at
   all. On merge, that whole block was dropped except the one paragraph of
   verified-true runtime-performance numbers (relocated into `main`'s
   existing README structure) and one factual "known blockers" paragraph.
   This is not a request to resubmit that README rewrite — see the
   pre-flight checklist below for why.

## Before you start — pre-flight checklist

This is what the reviewer actually had to check by hand this round, in
order to trust and merge your work rather than just paste-and-push it. Doing
these yourself before reporting a round done will make review faster and
reduce how much gets dropped on adoption:

- [ ] **Diff against `main`, not just your own branch's history.** Your
  branch has accumulated commits/docs across multiple rounds that were
  never actually merged onto `main` (see the README point above). Before
  writing a commit message that says something like "X now exists" or
  "the existing gates still pass," check whether `main` actually has the
  file/section you're referring to — `git diff main...HEAD --stat` from a
  fresh checkout, not from memory of your own branch's state.
- [ ] **Don't add README/doc content that references files outside this
  round's own commit.** If a doc links to
  `config/architecture_registry.json` or similar, either that file is part
  of the same commit or the link will 404 the moment it's merged onto a
  tree that doesn't have it. Grep for every path you reference in prose and
  confirm it resolves in the tree you're actually committing, not just the
  sandbox's broader (unmerged) state.
- [ ] **Run the full existing test suite fresh, not just the new tests
  you added.** `cargo test --workspace --release` from a clean state.
  Report the actual number your run produced, not a number carried over
  from an earlier round's report or your sandbox's cumulative branch state
  (round 2 caught a test-count mismatch this way; keep it from recurring).
- [ ] **Verify every Python tool you touch or add actually runs standalone
  in this repo** — `python <script>.py --help` or equivalent from a fresh
  clone, not just inside your own sandbox where extra internal-only modules
  might be importable. If it needs something not in this repo, say so in
  the commit message rather than silently shipping a script that can't run.
- [ ] **Benchmark numbers are host-specific — say so explicitly.** Report
  your sandbox's absolute numbers, but don't imply they'll transfer 1:1 to
  another machine (round 1 and round 2 both saw real but non-identical
  speedup ratios on the reviewer's machine vs. your sandbox — that's
  expected and fine, just don't frame a sandbox ratio as a guaranteed
  result elsewhere).
- [ ] **If an experiment fails a correctness gate, document the rejection
  instead of loosening the gate.** This round did this well (the int8
  activation prototype) — keep it up. A tolerance should only widen when
  there's a specific, understood, legitimate reason (e.g. the documented
  float-accumulation-order noise on the pooled/evidence values), never
  just because a change needs it to pass.
- [ ] **Still no search integration without a fresh SPRT gate.** This
  keeps being restated every round because it's the one rule that's already
  caught a real catastrophic failure (round 0's 0-20-0). The bar doesn't
  lower as the forward pass gets faster.
- [ ] **Push what you want reviewed.** A commit that only exists locally
  or only on your own branch's unpushed history can't be verified — make
  sure `git log origin/<your-branch>` actually shows what you think it
  shows before reporting a round as done (this was a real bug caught after
  round 1: a commit was made but never pushed).

## Status update — round 2

Your first pass (`c91937e`, "Accelerate Unarchitectured v1 Chessformer
runtime") was reviewed, independently re-benchmarked, and merged: real,
verified **75ms → 14.55ms per forward pass (~5.2x)** measured on the
reviewer's own machine, not just the number reported in your commit message.
Now on `main` at `464480e`.

Two things from that pass to know about:

1. **A real bug was found and fixed before merging, not just "unvalidated"
   as the commit message framed it.** The pooled board-token accumulator
   (`pooled`, used for the value head and the final representation) was
   hard-coded to the full `D_MODEL` (256) length and fed directly into
   width-sized `scaled_add`/`dot_product` calls for the narrower matryoshka
   exits (128, 192) — a genuine length mismatch that failed your own
   included `elastic_exit_shapes_and_finiteness` test on the reviewer's
   machine. Fixed by slicing `pooled[..width]` at both call sites (final
   pooling, value head). A new test, `narrow_exits_match_python_reference`,
   locks this in against real Python-reference numbers (`--all-exits` on
   `reference_forward_unarchitectured_v1.py`) so it can't silently regress — go run it
   before building on top of this area. Its evidence/representation checks
   use an intentionally wider tolerance (2e-2 vs. the usual 5e-3) because
   those two are both derived from the same 64-term sequential pooled sum,
   whose accumulation order legitimately differs from PyTorch's internal
   reduction; logits and best-move selection — the values that actually
   decide anything — still hold to 5e-3 for both narrow exits. If you can
   tighten that gap (e.g. pairwise/tree summation instead of a linear
   sequential accumulate) that's worth doing, but it's not blocking.

2. **The narrow/shallow exits (128, 192) are now correct, not just
   fast-and-broken, but they still aren't wired into anything.** Full scope
   for actually using them (distillation-time supervision, or a real
   accuracy-vs-speed tradeoff at inference) is still open and untouched.

Where things stand against the "what's needed" section below: 14.55ms is a
huge improvement but still a long way from "cheap enough to pay once per
move" (the original SPRT failure was caused by ~89ms/move being a
devastating tax at real time controls — 14.55ms is ~6x better but likely
still too much at fast time controls; nobody has re-run the SPRT gate with
this new number, so don't assume it's safe yet). Keep pushing on speed
(int8 integer dot products at inference time specifically remain untried
and are probably the next-biggest win), and don't re-attempt the
move-ordering integration without a fresh SPRT gate — the same discipline
that caught the first attempt's catastrophic failure.

## Context

Unchessed AI is a from-scratch Rust UCI chess engine
(github.com/Amoguslittleahhh/Unchessed-UCI-Engine). Its main evaluator is a
small NNUE. Separately, a much larger transformer architecture
("Unarchitectured v1" / `UnarchitecturedV1Student`, 4.2M-param student distilled
from a 58M-param oracle) was trained end-to-end on real Lichess data this
session and produces a real, calibrated, quantized checkpoint (student's
architecture: 8 layers, d_model=256, 8 heads, per-layer learned geometric
attention bias from templates, LoRA-adapted policy heads, a per-legal-move
regret head, an evidential WDL value head). That whole training pipeline —
data mining, Stockfish teacher-labeling, safety-gated training, export,
strict validation — is done and not what this prompt is about.

This prompt is about the *runtime* side: a pure-Rust inference forward pass
for that architecture now exists (`unchessed-core/src/aegis_v4_runtime.rs`,
committed on `main` at `cfd33ef`), and it is **numerically correct** —
cross-checked against an independent Python reference
(`tools/reference_forward_unarchitectured_v1.py`) run on the same real exported
checkpoint, on two different real positions, matching within 5e-3 on every
output value and picking the identical best move both times. There's also
an end-to-end test confirming the real `Position → PositionInput` converter
(real movegen, real board state, the mover-perspective vertical-flip
transform) produces correct model input, not just hand-built fixtures.

**The problem: it's far too slow to actually use.** Release-build
benchmark: **~89ms per forward pass** (`cargo test -p unchessed-core
--release benchmark_forward_pass -- --ignored --nocapture`), computed with
straightforward nested loops — no SIMD, no BLAS, no batching, nothing.

## What was tried and why it failed

The obvious "smallest, safest" integration point seemed to be: compute the
forward pass once per `go()` call, use the policy logits to set the
*initial* move-ordering at the root before any real search score exists
(doesn't touch eval, doesn't touch node-level search logic, real search
scores completely override it after the first iteration). This was wired
in, SPRT-gated per the project's normal discipline (never trust a change
without a real game-play test), and **failed catastrophically** — a 20-game
smoke test came back 0-20-0 for the version with the hint enabled, not a
subtle regression.

Root cause: `go()` is called once **per move**, for the entire game — not
once per game. At real time controls (the smoke test used `tc=5+0.05`,
matching this project's other SPRT gates), paying a fixed extra ~89ms on
*every single move* is a devastating tax relative to the ~100-250ms
per-move think times the engine normally uses at that time control —
routinely 40-60%+ of a move's entire budget spent on the forward pass
instead of search, showing up directly as consistently shallower depth and
a severe strength loss. The wiring was reverted (nothing broken landed on
`main`); only the validated, unwired `aegis_v4_runtime.rs` module was kept.

## What's needed

Rounds 7 through 9 together closed the entire integration checklist *and*
the diagnostic question of *why* — round 7 (real hardware, three real SPRT
batches, all negative or neutral, never positive) established the result;
round 8 closed the last named safety-suite gap; round 9 explained the
mechanism (structural cost-vs-benefit timing, not a bad model) and
surfaced real capacity gaps in the model itself (GAB 4x underprovisioned,
rating conditioning inert, weights 2.06x outside int8 range). The
integration-feature question is as settled as offline analysis can make
it; what's actually open now is different in kind.

**Two separate tracks going forward, and they shouldn't be conflated:**

1. **The `UnarchitecturedHint` feature itself** — genuinely optional at
   this point, not blocking. The isolated `MinTime` retest (below) is the
   only thing left that could change the picture, and even a positive
   result there wouldn't overturn round 9's finding that the cost is
   structural, not a bad hint — it would just mean the conservative
   threshold pays that structural cost rarely enough not to matter.
2. **A model retrain** — round 9's findings (GAB capacity, rating
   conditioning, weight clipping for int8, theme-balanced sampling) are a
   real, well-justified backlog for a *future* training run, explicitly
   flagged as retrain-only in every commit that found them (inference-time
   ablations on frozen weights don't prove a bigger GAB would score
   better — that needs its own training run and its own SPRT). This is a
   much bigger undertaking than a prompt round and shouldn't be treated as
   the next quick task — it's recorded now so it doesn't need
   re-discovering later, not so it gets started next.

**What's left is genuinely optional, not blocking, and this is a
legitimate stopping point for the feature itself if you'd rather not
pursue it further:**

1. **A cleanly isolated retest, if there's appetite for more precision.**
   Round 7's conservative-config result changed both
   `UnarchitecturedMinTime` *and* the time control at once (the
   conservative threshold is physically unreachable at the original fast
   time control) — so it proves the shipped config looks safe, but not
   cleanly *why*. Isolating the two variables (same `tc=60+0.6`, compare
   `MinTime=1000` vs `MinTime=30000` directly) would settle that.
2. **No configuration has ever trended positive across four real SPRT
   batches now.** The conservative config's neutral result means it's *not
   measurably harmful*, which is worth knowing, but it is not evidence the
   feature helps. Enabling it by default would need a real positive result,
   which doesn't exist at any tested config. Retiring the feature as
   "tested thoroughly at multiple configs, safety-net closed, never shown
   to help" is a complete, legitimate answer — this multi-round process
   exists to produce exactly that outcome when it's the true one.

Do not enable `UnarchitecturedHint` by default under any outcome here —
nothing across rounds 7 or 8 changes that.

If you'd rather keep pushing raw speed instead this round, the doc's own
"Remaining performance work" list is honest about what's left and none of
it is likely to be another 5x:

   - calibrated int8 *activations* (not just weights) via per-channel or
     per-group scaling — the per-token symmetric attempt failed parity at
     1.01e-2 vs. the required 5e-3; a different calibration scheme might
     close that gap, but it's unproven;
   - prepack/transcode matrices for wider deployment microkernels;
   - stop materializing duplicate f32 matrix copies once fallback/non-x86
     deployment constraints are settled;
   - a persistent inference worker instead of scoped per-call threads, if
     integration work demonstrates enough call volume to matter; and
   - caching exact full-position results keyed by state/model/persona.

Whichever direction: **any change to `aegis_v4_runtime.rs` must keep
passing its existing test suite** — the two Python-cross-check parity
tests (`start_position_matches_python_reference`,
`midgame_position_matches_python_reference`) and the real-movegen
end-to-end test (`position_to_input_matches_hand_built_start_position`).
Those are the only thing currently proving this port is numerically
correct; don't trade correctness for speed without re-validating against
the same Python reference (`tools/reference_forward_unarchitectured_v1.py`, which
loads the same real exported checkpoint at
`artifacts/unarchitectured-v1-final.unarchv1` and computes the identical
forward pass independently in PyTorch — regenerate its printed numbers and
update the Rust tests' expected values if a deliberate precision tradeoff
changes them beyond the current 5e-3 tolerance).

## Where to look

- `unchessed-core/src/aegis_v4_runtime.rs` — the module itself, including
  the benchmark test and all correctness tests.
- `tools/reference_forward_unarchitectured_v1.py` — the independent Python reference
  implementation used to validate correctness. Keep it in sync with any
  intentional behavior change.
- `unchessed-core/src/unarchitectured_v1.rs` — the binary package format
  this reads tensors from (already solid, byte-for-byte verified against
  its Python packer earlier this session — not in scope for this prompt).
- `tools/train_unarchitectured_v1_student_a100.py` — the original PyTorch reference
  architecture this was ported from (`UnarchitecturedV1Student.forward_path`),
  useful if any part of the port's correctness needs re-deriving from
  first principles while restructuring for speed.
