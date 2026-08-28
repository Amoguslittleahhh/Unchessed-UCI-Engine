# Engineering request: make the Unarchitectured v1 runtime forward pass fast enough to actually use

## Scope note

Unarchitectured v1 is the canonical, current architecture — the target to
build on and improve. Its predecessors (Apex v1, Hydra v1 through v4, and
any earlier lineage) were beta-testing iterations superseded by it; this
work is not an invitation to revert to or resume development on any of
them. Any follow-on architecture or training work belongs on top of
Unarchitectured v1, not as a parallel track exploring an earlier one.

## Current status (round 10, `main` at `d8fc659`)

Your eighth pass — int8-activation calibration (`f6113bf`) plus a
17-commit pivot from diagnosis into actual retrain infrastructure
(`8bfbd44`..`3a596cc`) — was reviewed, independently verified, and merged.
This is a genuine change in kind from every prior round: not offline
analysis of the existing checkpoint, but real training-data curation (four
corpora, ~231,000 games), a Rust toolchain working in your sandbox, and a
full cloud self-play + pretrain pipeline design (Verda AI, 5M games, A100
trainer).

This got flagged to the project owner before adoption, not merged
unilaterally — committing ~231,000 games of PGN directly into git and a
pipeline whose point is spending real cloud money both cross a line past
what prior pure-analysis rounds needed. Confirmed no cloud spend has
happened yet.

**Verified independently:**

- Full workspace build clean, `unchessed-core` 104/104 (zero Rust engine
  code touched by any of the 17 commits).
- **Found and fixed a real bug your sandbox couldn't catch**: this
  reviewer's Windows checkout (`core.autocrlf=true`) silently converts
  every committed LF to CRLF, changing file sizes/game counts in the PGN
  corpora and breaking your own `manifest.json` byte/game-count checks —
  6 real test failures on first pass. Cloning fresh in WSL (no autocrlf)
  confirmed 318 passed, 21 skipped, 0 failures, proving the committed data
  itself was correct. Fixed via `.gitattributes -text` rules plus a full
  re-checkout from clean git objects; 336/337 pass on Windows now.
- One remaining Windows-only failure (`test_ddp_gloo_two_rank_smoke`,
  gloo rendezvous) is a real PyTorch/Windows platform gap, not a bug —
  it passes cleanly on the same WSL run referenced above.
- Int8-activation calibration (`f6113bf`): reproduced the study
  end-to-end; the qualitative conclusion (all 5 schemes fail the 5e-3
  gate by 4-14x, none close) reproduced exactly, and the holdout-
  generalization table matched the committed JSON to the last digit.

**Where things stand:** this round's content (training data, cloud
pipeline, pretrain trainer) is infrastructure for a *future* retrain, not
a change to anything shipped or running. `UnarchitecturedHint` stays
default-off, `runtime_safety_suite` stays false, and nothing has been
executed against real cloud compute — that decision, and any actual
spend, stays with the project owner explicitly.

## What's needed next

1. **A retrain decision, if you want to act on round 9's findings**
   (GAB capacity, rating conditioning, int8 weight clipping) using the
   infrastructure round 10 just built. This is the project owner's call,
   not something to proceed on by default — it means real cloud spend.
2. **A cleanly isolated `MinTime` retest**, still open from round 7:
   same time control (`tc=60+0.6`), only `MinTime` varied between 1000
   and 30000, to separate the two variables the conservative-config
   result confounded. Optional polish, not a blocker.
3. **`UnarchitecturedHint` stays default-off** either way. No config
   tested across four real SPRT batches has ever trended positive —
   only the conservative default has been shown *not measurably
   harmful*, which is not the same as evidence it helps.

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
- **Round 10**: see "Current status" above.

## Correctness gates that must keep passing

Any change to `aegis_v4_runtime.rs` must keep passing
`start_position_matches_python_reference`,
`midgame_position_matches_python_reference`, and
`position_to_input_matches_hand_built_start_position` — the only things
currently proving the Rust port is numerically correct against
`tools/reference_forward_unarchitectured_v1.py`. Don't trade correctness
for speed without re-validating against that same Python reference.

## Pre-flight checklist

Established over rounds 1-10, still the standard before reporting a round
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
