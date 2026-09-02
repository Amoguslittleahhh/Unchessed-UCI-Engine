# Engineering request: make the Unarchitectured v1 runtime forward pass fast enough to actually use

## Scope note

Unarchitectured v1 is the canonical, current architecture — the target to
build on and improve. Its predecessors (Apex v1, Hydra v1 through v4, and
any earlier lineage) were beta-testing iterations superseded by it; this
work is not an invitation to revert to or resume development on any of
them. Any follow-on architecture or training work belongs on top of
Unarchitectured v1, not as a parallel track exploring an earlier one.

## Current status (round 19 merged, `main` at `097f40a`)

**NNUE**: reviewer's 108M cloud SPRT closed the round-13 ask —
**−155.6 ± 47.7 Elo vs the shipped default**, best val-MAE 47.8cp (best
in the whole investigation). Applying round 13's own pre-committed rule,
**cloud 178M remains NO-GO** even at the optimistic end of that interval.
`docs/nnue-v4-108m-recipe-result.md`.

**Persona/adapter (`PersonaSmooth`, `EngineDetectV2`)**: merged in round
17 after two rejections (15: didn't compile, unconditional behavior
change, simulation-only; 16: compiled and gated, but 2 tests failed on
its own design claims). Both UCI options default `false`, old behavior
byte-identical when off. **Still no real cutechess SPRT** — neither
default may flip until one runs.

**Unarchitectured v1 runtime speed (the original ask)**: round 18's
GEMM-tiling/scratch-reuse refactor didn't compile (9 errors — duplicate
definitions, inconsistent call-site arg counts across
`attention_heads_dispatch`). **Round 19 fixed it and is merged.**
Verified independently: `cargo test --workspace --release` 118/118
passing, the three named parity gates
(`start_position_matches_python_reference`,
`midgame_position_matches_python_reference`,
`position_to_input_matches_hand_built_start_position`) all pass by
name, `pytest tools/` 377/378 (same known Windows-only gloo gap), no
Rust bracket issues. Real measured speedup on this host:
`benchmark_forward_pass` → **7.46ms/call** (200 calls, 1.49s total).
`UnarchitecturedHint` untouched, still default-off.

**Root cause of every prior compile failure identified**: arena's
sandbox (Debian 12 KVM/E2B-style VM) filters outbound HTTPS — GitHub
HTML/API works, but `rustup.rs`/the Debian CDN often fail TLS. Arena
confirmed this directly: ran `scripts/setup-rust-toolchain.sh`
(apt-then-rustup, idempotent), both paths failed with the exact TLS/apt
errors predicted, logged honestly in
`docs/setup-rust-toolchain-sandbox-blocker.md` instead of silently
skipping verification. **This is a real, standing environment
constraint, not a process failure** — arena cannot self-verify Rust
changes compile until someone with access to that sandbox's network
policy allowlists an internal mirror or proxy for
`static.rust-lang.org`/`rustup.rs`/the Debian archive. Until then, every
Rust-touching round gets independently compiled and tested here before
merge, same as always — that discipline is exactly what caught rounds
15 and 18.

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
2. **A real cutechess SPRT for `PersonaSmooth`/`EngineDetectV2`**
   (Adaptive=true both sides, same shape as
   `scripts/sprt-history/sprt_punish_latch.sh`), now that the code
   itself is merged, tested, and correct. Both options still default
   `false` — this is the only thing standing between them and being
   flippable. Do not flip either default without it.
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
- **Reviewer 108M run**: see "Current status" above,
  `docs/nnue-v4-108m-recipe-result.md`.
- **Rounds 15-17** (persona/adapter gating, same branch): round 15
  rejected outright (didn't compile; unconditional live behavior change
  on simulation-only evidence). Round 16 fixed the process problems
  (compiled, properly UCI-gated) but 2 of its own new tests failed on
  its own headline claims. Round 17 fixed both bugs — verified
  independently (118/118 Rust tests, 377/378 Python, no conflict
  markers) and merged (`63c7262`). Still needs a real SPRT before either
  new option's default can flip.
- **Round 18**: rejected, not merged. 9 real compile errors in an
  `aegis_v4_runtime.rs` GEMM-tiling refactor. **Round 19** fixed all of
  them and added a real toolchain bootstrap after identifying the actual
  root cause (arena's sandbox filters outbound HTTPS to
  `rustup.rs`/Debian's CDN). Verified independently (118/118 Rust tests,
  three parity gates pass by name, 7.46ms/call measured) and merged
  (`097f40a`). See "Current status" above.
- **Round 18**: rejected, not merged. 9 real compile errors in an
  `aegis_v4_runtime.rs` GEMM-tiling refactor (duplicate definitions,
  inconsistent call-site argument counts) plus a trailing duplicated
  code fragment. See "Current status" above.

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
- [ ] **Run `source scripts/setup-rust-toolchain.sh` first, every round
  that touches any `.rs` file.** It's idempotent (safe to re-run, no-ops
  if rustc is already there) and tries apt then rustup, reporting
  exactly what happened with each — known blocker in this sandbox
  specifically: outbound HTTPS is filtered, GitHub HTML/API works but
  `rustup.rs`/the Debian CDN often fail TLS. If the script's output
  says both paths failed, paste that actual output into the round's
  doc — that is a real, reportable environment blocker, not something
  to route around with "(need rustc)" as a footnote. "(need rustc)" is
  not the same as "this compiles" — rounds 15 and 18 both shipped real
  compile errors this way, after the toolchain script would have caught
  them in seconds if it had existed and been run.
- [ ] A behavior change to code that runs in real games (`adapt.rs`,
  `search.rs`, anything reachable from `run_go`) needs a UCI option
  defaulting to the *old* behavior plus a real cutechess SPRT before the
  default flips — a Python/stdlib simulation is evidence for the writeup,
  not a substitute for the gate. Round 15 shipped an unconditional
  `adapt.rs` behavior change on simulation-only evidence.

## Where to look

- `scripts/setup-rust-toolchain.sh`, `rust-toolchain.toml` — run the
  script first, every round touching `.rs`. See the pre-flight checklist.
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
- `unchessed-core/src/adapt.rs`, `uci.rs` — `PersonaState`
  (`PersonaSmooth` EMA/dwell) and `OpponentModel` misfire fixes
  (`EngineDetectV2`), merged in round 17. Both UCI options default
  `false`; needs a real SPRT before either can flip. See "Current
  status" above.
