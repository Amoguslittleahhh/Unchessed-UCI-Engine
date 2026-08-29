# Engineering request: make the Unarchitectured v1 runtime forward pass fast enough to actually use

## Scope note

Unarchitectured v1 is the canonical, current architecture — the target to
build on and improve. Its predecessors (Apex v1, Hydra v1 through v4, and
any earlier lineage) were beta-testing iterations superseded by it; this
work is not an invitation to revert to or resume development on any of
them. Any follow-on architecture or training work belongs on top of
Unarchitectured v1, not as a parallel track exploring an earlier one.

## Current status (round 12, `main` at `9466820`)

Full sweep of round 11's "other open items": NNUE quiet-filter blocker
fixed and measured, NNUE 8 piece-count output buckets implemented
end-to-end (runtime + trainer), and both research questions
(DiffusionBlocks, oracle-side rating conditioning) answered. Reviewed,
verified, merged.

**Verified independently:** build clean, `unchessed-core` 110/110
(matches claimed count, +4 new v4 bucket tests), `train_nnue.py selfcheck`
ALL PASS (numpy-vs-model max diff 1.49e-08 on this host, same order of
magnitude as their 2.98e-08 — different seed, same conclusion), `pytest
tools/` 336/4/352 (the one failure is the already-known Windows-only
`test_ddp_gloo_two_rank_smoke` gloo gap, confirmed passing on WSL in
prior rounds — not new). Read the actual `nnue.rs` diff line by line: the
v4 bucket formula (`(pieces-1)/4` clamped to 7) matches between Rust and
the Python trainer exactly, v1-v3 files stay bit-identical (bucket
forced to 0), and the new `UNCHESSED_NNUE_MIN_BASE_SECS` env override
defaults to the original fail-closed 180s gate — no default behavior
changed anywhere in this round.

**Honest state, same as they reported**: no NNUE net has been retrained
and nothing has been SPRT'd. This round is better tooling + a verified
new file format, zero shipping change — the v4 format isn't used by
anything until a v4 net actually exists and passes SPRT.

## What's needed next

1. **An NNUE retrain**, now unblocked on both axes that were stopping
   it: the quiet-filter base-seconds gate (fixed) and the 8-bucket head
   (implemented). This is real compute but far cheaper than the
   Unarchitectured v1 retrain items — no new cloud infrastructure needed,
   just the existing 108M-position corpus. Project owner's call on
   whether/when to spend the compute; if it happens, it needs a fresh
   SPRT gate before touching the default evaluation, same rule as
   everything else.
2. **A retrain decision** for round 9's Unarchitectured v1 findings (GAB
   capacity, rating conditioning, int8 weight clipping), if the oracle
   checkpoint becomes available — see item 3 below, this is now gated on
   that.
3. **Oracle-side rating conditioning**: narrowed, not closed. One
   hypothesis (student never saw rating variation) is ruled out with
   repo evidence; the deciding experiment (200 positions × 7 ratings on
   the oracle checkpoint) needs the oracle checkpoint, which isn't in
   this repo. If it ever becomes available, run it — cheap (CPU-minutes)
   and it decides where the real fix goes (oracle vs. distillation).
4. **A cleanly isolated `MinTime` retest**, still open from round 7 —
   must now also state which `HintExit` it used. Optional polish, not a
   blocker.
5. **`UnarchitecturedHint` stays default-off**. No config tested across
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
- **Round 12**: see "Current status" above.

## Correctness gates that must keep passing

Any change to `aegis_v4_runtime.rs` must keep passing
`start_position_matches_python_reference`,
`midgame_position_matches_python_reference`, and
`position_to_input_matches_hand_built_start_position` — the only things
currently proving the Rust port is numerically correct against
`tools/reference_forward_unarchitectured_v1.py`. Don't trade correctness
for speed without re-validating against that same Python reference.

## Pre-flight checklist

Established over rounds 1-12, still the standard before reporting a round
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
- `unchessed-core/src/nnue.rs`, `tools/train_nnue.py` — the NNUE
  evaluator, now on file format v4 (8 piece-count output buckets).
- `docs/nnue-round-12-results.md`, `docs/nnue-dataset-quiet-filters.md`,
  `docs/research-notes-moe-2507.11181.md` — the NNUE retrain backlog:
  what's built, measured, and still needs a real retrain + SPRT.
- `docs/research-notes-diffusionblocks-2506.14202.md`,
  `docs/rating-conditioning-finding.md` — the two round-12 research
  answers (DiffusionBlocks: defer/drop; oracle rating conditioning:
  narrowed, blocked on the oracle checkpoint).
