# Engineering request: make the Unarchitectured v1 runtime forward pass fast enough to actually use

## Scope note

Unarchitectured v1 is the canonical, current architecture — the target to
build on and improve. Its predecessors (Apex v1, Hydra v1 through v4, and
any earlier lineage) were beta-testing iterations superseded by it; this
work is not an invitation to revert to or resume development on any of
them. Any follow-on architecture or training work belongs on top of
Unarchitectured v1, not as a parallel track exploring an earlier one.

## Current status (round 16 rejected again, closer; round 13 stands as the last merged NNUE work)

**Reviewer follow-up (real hardware, not arena) closed the round-13 ask.**
Rented a Verda `CPU.32V.128G` box (~$1, deleted right after), trained the
defended recipe on all 108M real self-play positions, and ran the real
SPRT: **−155.6 ± 47.7 Elo vs the shipped default** (188 games), best
val-MAE 47.8cp — the best number in the whole investigation, but round
13's own pre-committed rule ("still >100 Elo behind → cloud 178M stays
NO-GO") still applies even at the optimistic end of that interval.
**Cloud 178M remains NO-GO.** Full trend table, the go/no-go reasoning,
and what's actually worth trying next (stronger labels, not more of the
same): `docs/nnue-v4-108m-recipe-result.md`.

**Round 15 (`01a0581c` @ `b3e1e03`) is rejected, not merged.** Two
independent problems, either one disqualifying on its own:

1. **It does not compile.** `cargo test --workspace --release` fails:
   `run_go`'s signature in `uci.rs` still declares
   `persona: Arc<Mutex<Mode>>` while the call sites were changed to pass
   `Arc<Mutex<PersonaState>>` — an incomplete rename, caught by rustc in
   under 10 seconds. Both new docs this round hedge with "(need rustc)",
   which now reads as "this was never actually compiled" rather than a
   footnote.
2. **Even fixed, it changes live adapter behavior with no opt-out and no
   real SPRT.** `decide_mode` was fully replaced by `PersonaState::update`
   in the UCI worker's `adaptive_now` path — unconditionally, no new UCI
   option gating it. The only validation is a Python simulation
   (`tools/persona_stability_sprt.py`, synthetic AR(1) eval traces) that
   the round's own doc honestly calls "not a cutechess SPRT." That's a
   real behavior change to what real opponents actually experience,
   shipped without the one rule this project has enforced since round 0.
   The `elo_detector.py` misfire-threshold changes in the same commit
   have the identical problem: real detection-logic changes, simulation-only
   validation, no opt-out.

Send back for: (a) fix the compile error, (b) gate both behavior changes
behind a UCI option that defaults to the *old* behavior until a real
cutechess SPRT (Adaptive=true both sides, same as
`scripts/sprt-history/sprt_punish_latch.sh`) validates the new one. The
underlying ideas (EMA/dwell smoothing, the four misfire cases) may well
be real improvements — the simulation numbers are plausible — but
"plausible in simulation" is not the bar this project has held
`aegis_v4_runtime.rs` or the NNUE retrain to, and it shouldn't be lowered
here just because the change lives in `adapt.rs` instead.

**Round 16 (same branch, `502eb26`..`72bf78f`) fixed both real problems
from round 16's ask and is closer, but still rejected — this time on a
correctness bug in the new logic itself, not on process.** Verified
independently:

- **Compiles clean** (`cargo test --workspace --release`, after the usual
  Windows Smart App Control re-touch-and-rebuild — unrelated to this
  branch). No leftover `<<<<<<<`/`=======`/`>>>>>>>` markers anywhere in
  the tree (round 16 also had to clean up literal merge-conflict markers
  left in `README.md` from an earlier commit in the same branch — worth
  noting for its own sake: that should never have been committed in the
  first place).
- **Genuinely gated**: `PersonaSmooth` and `EngineDetectV2` are both new
  UCI options, both default `false`; with both false, behavior is
  byte-for-byte the pre-round-15 `decide_mode`/`is_computer` path. This
  is the actual fix that was asked for.
- **But two of the new feature's own tests fail**, reproducibly
  (`cargo test -p unchessed-core --lib`, 116 passed / 2 failed):
  `persona_state_dwell_ignores_one_move_clinch_spike` (a single CLINCH
  vote flips immediately — the "2 agreeing plies" dwell claim from the
  design doc isn't actually enforced for CLINCH) and
  `persona_state_ema_rejects_single_eval_spike_across_defend` (a
  sustained −400cp collapse fails to trigger the emergency DEFEND
  bypass — the exact safety-relevant case the doc's own table
  describes as "no dwell"). Both are real logic bugs in `PersonaState`,
  not test-authoring mistakes — read against the doc's own two headline
  claims, the implementation doesn't yet do what it says it does.
- **Minor, non-blocking**: `test_search_param_consistency.py`'s new
  `TestRealRepo::test_is_consistent` hardcodes `checked == 20`; the real
  repo now has 24 (no actual parameter inconsistencies found — `failures
  == []` passes — just a stale constant). Update the hardcoded number.

Because both flags default false, none of this reaches real games as
shipped — the gating is doing its job. But a feature that fails its own
unit tests isn't ready for a real SPRT yet either. Fix the two
`PersonaState` bugs, get `cargo test --workspace --release` fully green
on a machine with rustc (not just "should compile"), then it's ready for
the real cutechess SPRT gate.

## What's needed next

1. **NNUE: not more data.** The 108M result plus last round's Bayes-floor
   analysis (`docs/ieee-low-cp-val-mae-and-persona.md`, itself simulation-only
   but methodologically sound and independently reproduced) both point the
   same direction: the current 5000-node HCE labels cap achievable val-MAE
   around 48-56cp, and this net is already there. Worth exploring:
   stronger/deeper labels on the *existing* corpus (self-distillation from
   the shipped net at high node count, or a deeper HCE search), separating
   label-noise from architecture-capacity before assuming either is the
   fix. Not cloud spend on more of the same labels.
2. **Round 16's `PersonaState` bugs**: fix
   `persona_state_dwell_ignores_one_move_clinch_spike` and
   `persona_state_ema_rejects_single_eval_spike_across_defend` (both
   reproduce on `cargo test -p unchessed-core --lib`), get the full
   workspace test suite green on a machine with rustc, then it's ready
   for the real cutechess SPRT gate (`PersonaSmooth`/`EngineDetectV2`
   both default false already — that part is done). Do not flip either
   default without that SPRT.
3. **A retrain decision** for round 9's Unarchitectured v1 findings (GAB
   capacity, rating conditioning, int8 weight clipping), if the oracle
   checkpoint becomes available — see item 4, this is gated on that.
4. **Oracle-side rating conditioning**: narrowed, not closed. One
   hypothesis (student never saw rating variation) is ruled out with
   repo evidence; the deciding experiment (200 positions × 7 ratings on
   the oracle checkpoint) needs the oracle checkpoint, which isn't in
   this repo. If it ever becomes available, run it — cheap (CPU-minutes)
   and it decides where the real fix goes (oracle vs. distillation).
5. **A cleanly isolated `MinTime` retest**, still open from round 7 —
   must now also state which `HintExit` it used. Optional polish, not a
   blocker.
6. **`UnarchitecturedHint` stays default-off**. No config tested across
   four real SPRT batches has ever trended positive.

## Other open items

**DiffusionBlocks for a labeling-oracle retrain: answered, effectively
closed.** Full answer in `docs/research-notes-diffusionblocks-2506.14202.md`.
Bottom line: architecturally it would fit a chess value oracle with no
I/O rework, but there are no published regression results (only
classification/generation) and the code is ViT/CIFAR-only. At the real
scales in play (published chess transformers run ~9M-270M params), the
memory footprint fits comfortably on one 80GB card with ordinary
backprop — DiffusionBlocks' savings only become load-bearing past
~500M-1B params on a sub-48GB card, which isn't this project's scale.
Recommendation: defer/drop; revisit only if a future oracle retrain
targets that size on constrained hardware. No action needed unless that
changes.

## History (condensed)

- **Round 0**: root-hint wired directly into search, SPRT'd, failed
  catastrophically (0-20-0 at `tc=5+0.05`) — the ~89ms forward pass ate
  40-60%+ of the move budget every move. Reverted; only the validated,
  unwired forward-pass module (`aegis_v4_runtime.rs`) survived.
- **Round 2** (`464480e`): AVX2/FMA SIMD kernels, 75ms → 14.55ms (5.2x).
  Fixed a real pooled-accumulator length bug found on review.
- **Round 3** (`c97dd0b`): retained int8 weights + int16 activations,
  1.22x further. Rejected int8 *activations* honestly (parity failure),
  kept int16.
- **Round 4** (`1d6da41`): dispatch/cache reduction, 1.2x further; fixed
  a peripheral tool's missing-module issue.
- **Round 5** (`9ced983`): fail-closed root-hint trial — the first
  `search.rs` touch since round 0. Proven unreachable from real UCI play
  by construction and by grep.
- **Round 6** (`d0e5666`): wired the default-off `UnarchitecturedHint`
  UCI candidate. Verified live on the compiled binary, not just in tests.
- **Round 7** (real hardware, reviewer-run, not arena): fixed a real
  inference-thread default bug (sequential beats every parallel count on
  this hardware); measured real NPU dispatch latency; ran three real SPRT
  batches — **-26.1 Elo, -15.1 Elo (replication), -5.8 Elo (conservative
  config)** — never positive, the conservative config statistically
  neutral. Added a broader mate/check safety suite.
- **Round 8** (`ffffc29`): closed the last named safety gap — real-
  checkpoint hint disagreements (a forced mate ranked 10th of 17),
  verified by regenerating the report from the real checkpoint.
- **Round 9** (`9cea2f4`): explained *why* the hint costs Elo
  (structural budget cost, not a bad model — the policy is actually a
  *better* free orderer than movegen order). Found real capacity gaps:
  GAB provisioned at a quarter of the comparable paper config, rating
  conditioning confirmed inert (0/200 moves change across a 600→3200
  sweep), weights 2.06x outside the int8 range. Fixed one real test
  timing-assumption bug found via diagnostic instrumentation.
- **Round 10** (`d8fc659`): int8-activation calibration closed negative
  (5 schemes, all fail 5e-3 by 4-14x). Pivoted into real retrain
  infrastructure — four training corpora (~231,000 games), a working
  Rust toolchain in arena's sandbox, cloud self-play/pretrain pipeline
  design (Verda AI). Flagged to the project owner before adoption given
  the scope change (committing PGN data to git, infrastructure for real
  cloud spend); no cloud spend has occurred. Found and fixed a real
  Windows `core.autocrlf` bug corrupting committed PGN files on checkout.
- **Round 11** (`a936d8d`): inlined AVX2 attention, vectorized softmax
  exp, scratch reuse, per-head GAB streaming (all now default); added
  `UnarchitecturedHintExit`. Real speedup reproduced (1.12x full-exit
  forward pass on this host).
- **Round 12** (`9466820`): NNUE quiet-filter blocker fixed, 8 piece-count
  output buckets implemented end-to-end, DiffusionBlocks and oracle-rating
  research questions answered. No net trained, zero shipping change.
- **Reviewer follow-up** (real hardware, since round 12): three real
  SPRTs (940k/9M/27M positions, all lost badly), an ablation proving the
  8-bucket format wasn't the cause, root cause identified as the
  committed corpus being 0.5% of the real 178M-position corpus.
  `docs/nnue-v4-retrain-data-scaling-finding.md`.
- **Round 13** (`0b37ddd`): found and fixed a real trainer bug (every
  diagnostic net had exported a worse-than-best checkpoint), recovered
  the shipped launcher's actual recipe, declared cloud 178M NO-GO pending
  a local 108M SPRT. `docs/nnue-v4-training-recipe.md`.
- **Round 14** (`4f94f57`, merged): simulation-only Bayes-noise-floor
  analysis of NNUE val-MAE plus a fail-closed cloud launcher script
  (real token-gated safety rail, genuinely useful). Off the requested
  ask (was supposed to be the 108M SPRT) but technically sound and
  harmless; flagged as scope drift, not rejected.
  `docs/ieee-low-cp-val-mae-and-persona.md`.
- **Reviewer 108M run + rounds 15-16**: see "Current status" above.

## Correctness gates that must keep passing

Any change to `aegis_v4_runtime.rs` must keep passing
`start_position_matches_python_reference`,
`midgame_position_matches_python_reference`, and
`position_to_input_matches_hand_built_start_position` — the only things
currently proving the Rust port is numerically correct against
`tools/reference_forward_unarchitectured_v1.py`. Don't trade correctness
for speed without re-validating against that same Python reference.

## Pre-flight checklist

Established over rounds 1-13, still the standard before reporting a round
done:

- [ ] Diff against `main`, not your own branch's accumulated history.
- [ ] Don't reference files outside the round's own commit in docs.
- [ ] Run the full test suite fresh; report the actual number, not a
  carried-over one.
- [ ] Verify every Python tool runs standalone (`--help` with no
  internal-only imports).
- [ ] State benchmark numbers as host-specific — never imply 1:1
  transfer to another machine.
- [ ] If an experiment fails a correctness gate, document the rejection
  instead of loosening the gate.
- [ ] No search integration without a fresh SPRT gate — the one rule
  that's already caught a real catastrophic failure (round 0).
- [ ] Push what you want reviewed — verify with
  `git log origin/<branch>`, not local state.
- [ ] Any format needing byte-exact integrity across platforms (PGN,
  EPD, JSONL manifests) needs a `.gitattributes -text` rule — round 10
  found this the hard way.
- [ ] If the sandbox has no rustc, say so plainly and don't claim
  Rust-side verification you didn't do — "(need rustc)" as a footnote is
  not the same as "this compiles." Round 15 shipped a real compile error
  this way.
- [ ] A behavior change to code that runs in real games (`adapt.rs`,
  `search.rs`, anything reachable from `run_go`) needs a UCI option
  defaulting to the *old* behavior plus a real cutechess SPRT before the
  default flips — a Python/stdlib simulation is evidence for the writeup,
  not a substitute for the gate. Round 15 shipped an unconditional
  `adapt.rs` behavior change on simulation-only evidence.

## Where to look

- `unchessed-core/src/aegis_v4_runtime.rs` — the runtime module, forward
  pass, and all correctness tests.
- `tools/reference_forward_unarchitectured_v1.py` — the independent
  Python reference. Keep in sync with any intentional behavior change.
- `unchessed-core/src/unarchitectured_v1.rs` — the binary package format.
- `tools/train_unarchitectured_v1_student_a100.py` — the original
  PyTorch architecture this was ported from.
- `docs/unarchitectured-v1-integration-trial.md`,
  `docs/unarchitectured-v1-why-the-hint-costs-elo.md`,
  `docs/gab-capacity-finding.md`, `docs/rating-conditioning-finding.md`,
  `docs/fishtest-and-quantization-notes.md` — the round 7-9 findings that
  now constitute the retrain backlog.
- `unchessed-core/src/nnue.rs`, `tools/train_nnue.py` — the NNUE
  evaluator, now on file format v4 (8 piece-count output buckets).
- `docs/nnue-round-12-results.md`, `docs/nnue-dataset-quiet-filters.md`,
  `docs/research-notes-moe-2507.11181.md` — the NNUE retrain backlog:
  what's built, measured, and still needs a real retrain + SPRT.
- `docs/research-notes-diffusionblocks-2506.14202.md`,
  `docs/rating-conditioning-finding.md` — the two round-12 research
  answers (DiffusionBlocks: defer/drop; oracle rating conditioning:
  narrowed, blocked on the oracle checkpoint).
- `docs/nnue-v4-retrain-data-scaling-finding.md` — the reviewer follow-up:
  three real SPRT'd data points, the bucket-vs-dataset-size ablation, and
  the specific recipe-validation ask above. `scripts/nnue-pipeline/
  local_regen.sh` and `scripts/research/wsl_sprt_nnue_{v4,real9m,real27m}.sh`
  reproduce it exactly.
- `docs/nnue-v4-training-recipe.md` — round 13 answer: epoch/step count,
  shipped-launcher recipe, 108M gap, go/no-go. Trainer:
  `tools/train_nnue.py` + `tools/nnue_train_control.py`. Wrapper:
  `scripts/nnue-pipeline/train_recipe.sh`.
- `docs/nnue-v4-108m-recipe-result.md` — the 108M cloud SPRT that closes
  the round-13 ask: −155.6 Elo, cloud 178M still NO-GO by round 13's own
  rule, and what's actually worth trying next.
  `scripts/research/wsl_sprt_nnue_108m.sh` reproduces it.
- `docs/ieee-low-cp-val-mae-and-persona.md` — round 14's simulation-only
  val-MAE noise-floor analysis; not requested, technically sound.
- Rounds 15-16 (`adapt.rs` persona EMA/dwell, `elo_detector.py`):
  rejected both times, not in `main`. Round 16 fixed the compile break
  and added proper default-off UCI gating (`PersonaSmooth`,
  `EngineDetectV2`); still rejected on two failing tests in the new
  `PersonaState` logic itself. See "Current status" above.
