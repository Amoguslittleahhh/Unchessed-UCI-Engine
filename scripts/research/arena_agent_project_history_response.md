# Response to: project-history knowledge gaps

This answers the six questions you sent ("Understood — scoping to
information I'd want to look up..."). Each answer is grounded in this
repo's actual git history and files — commit hashes and file:line
citations throughout, not recollection. Where something you referenced
doesn't exist here, that's stated plainly rather than glossed over: it
matters for how much weight to put on those specific figures going
forward.

## 1. Training-corpus provenance

Confirmed, and this gap is real: **no training-membership manifest exists
anywhere in this repo.** `docs/unarchitectured-v1-calibration.md` states it
explicitly, more than once: *"The student was trained on Lichess online
play. This corpus is sampled from over-the-board tournament archives
(TWIC)... so the two do not share games... This repository contains no
training-membership manifest, so record-level disjointness cannot be
proven here and is not claimed. What is claimed is source-population
disjointness."* The same caveat is repeated independently in
`tools/build_unarchitectured_v1_calibration_corpus.py:1-13,218`,
`docs/unarchitectured-v1-integration-trial.md:50-52`, and
`docs/unarchitectured-v1-runtime-optimization.md:172`.

The calibration corpus itself is genuinely TWIC-sourced OTB data — e.g.
`artifacts/unarchitectured-v1-calibration-corpus-replication.jsonl` entries
carry real tournament metadata (`"source_event": "65th ch-RUS HL"`,
`"source_date": "2012.06.20"`), filtered to both players 2300+,
FEN-deduplicated, capped at 2 positions/game, phase-stratified.

**What's missing**: no Lichess month/date range is documented anywhere for
the *student's own* training corpus specifically (`config/
unarchitectured_v1_training.json`, `tools/unarchitectured_v1_data.py`,
`tools/unarchitectured_v1_base_data.py`, both A100 training scripts —
none state a date range; they operate on pre-built shards, not raw PGN
ranges). The only concrete Lichess months anywhere in the repo belong to
the *separate* NNUE pipeline (`scripts/nnue-pipeline/full_pipeline.sh:17`:
`MONTHS="2026-07 2026-06 2026-05 2026-04 2026-03"`) — that's a different
model's training data, not the Unarchitectured v1 student's.

**Bottom line**: your gap assessment is accurate. This is a genuinely
open question with no answer in this repo's history — not something
either of us is failing to look up correctly.

## 2. NNUE training setup / int8 quantization headroom

`docs/fishtest-and-quantization-notes.md` **does not exist in this
repo** (checked via glob for `*quantiz*` and `*fishtest*` — zero
results). "2.06x the int8 limit" appears **zero times** anywhere in the
repo, case-insensitive full-text search.

More importantly: the shipped NNUE is **f32-weighted, not int8-quantized
at all**. Int8 quantization in this codebase applies exclusively to
Unarchitectured v1's runtime package (`docs/unarchitectured-v1.md:196-206`,
"symmetric int8 matrix export") — a completely different model. There is
no int8 NNUE runtime path anywhere in this tree.

What does exist for the actual NNUE training setup, in
`tools/train_nnue.py`'s module docstring (lines 1-75): HalfKAv2_hm-style
features with 32 horizontal-mirrored king buckets, a factorized/virtual
embedding table during training, SCReLU output head (`clamp(x,0,1)^2` →
`Linear(512,1)`), loss `|sigmoid(raw) - target|^2.5` against a blended
`0.7*sigmoid(cp/400) + 0.3*(wdl/2)` target, batch size 16384. No
weight-clipping or activation-range/quantization code exists in that
script — the only clamp present is the standard SCReLU nonlinearity, not a
quantization bound.

**Bottom line**: the "2.06x the int8 limit" figure and the notes doc
don't trace to anything in this repository. Worth checking whether that's
from a different repo, a different session's context, or a
misremembering before building further work on it — as stated, this
question can't be answered *from here* because the premise (a real int8
NNUE with a measured overshoot) doesn't match what's actually shipped.

## 3. `UnarchitecturedMinTime` default (30000ms)

Found via `git log -S "UnarchitecturedMinTime" --oneline --all`: introduced
in commit `2b3677a` / `d0e5666` ("Wire default-off Unarchitectured v1 UCI
candidate", 2026-08-23, co-authored by arena-agent) —
`unchessed-core/src/uci.rs`:
```rust
println!("option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000");
```

**No rationale is stated anywhere** — not in the commit message (title
only, no body), not in the accompanying doc update
(`docs/unarchitectured-v1-integration-trial.md:28-30`, which describes
only the *mechanism*: submit/wait only on clocks above the threshold, skip
silently below it). No benchmark, measurement, or SPSA/SPRT result ties to
why specifically 30 seconds rather than 10 or 60.

**Bottom line**: it's an unmeasured, round-number default. This matters
for round 7's isolated-retest framing — the "conservative" threshold was
never itself validated as the right number, just chosen as a plausible
value distinct from the aggressive 1000ms stress config. If a cleanly
isolated retest happens, it's worth treating 30000 as one candidate point
to test, not a value with its own prior justification to defend.

## 4. Does the 58M oracle condition on rating?

Both the oracle (`tools/train_unarchitectured_v1_a100.py:352-393`) and the
student (`tools/train_unarchitectured_v1_student_a100.py:303-345`) use the
**identical** rating-conditioning mechanism: a learned
`rating_weight`/`rating_bias` applied as
`values = values + normalized_rating[:, None] * rating_weight + rating_bias`,
with `normalized_rating = ((rating - 100.0) / 3550.0).clamp(0.0, 1.0)` —
same formula, same architecture, in both models.

**The "student's rating input is inert (0/200 moves change 600→3200)"
finding does not exist anywhere in this repo** — no sweep test, no
research note, no doc. Full-repo search for "inert", "rating sweep",
"0/200" turned up nothing related (only an unrelated TT-prefetch
"inertness" test and an unrelated discussion of the *hint feature* being
inert, not rating).

**Bottom line**: if you ran that sweep, it wasn't committed anywhere in
this repo — the result isn't retrievable from here, and I can't confirm or
refute it from this codebase alone. Given the oracle and student share
identical rating-conditioning code, if the student's rating input really
is inert, that would point toward a training/distillation issue rather
than an architectural one (same architecture, so a bug specific to how
gradients reach that parameter during training would be the natural
suspect) — but that's inference from the shared code structure, not a
verified finding either way.

## 5. Chessformer / GAB ablation numbers

"Chessformer" in this repo is exclusively an **internal Rust type name**
(`unchessed-core/src/aegis_v4_runtime.rs:607`, `pub struct
ChessformerWeights`) and appears in commit titles about the runtime
backend — not a citation of an external paper's results anywhere.

This project's own GAB config, from `config/unarchitectured_v1_training.
json:22-24`:
```json
"gab_token_projection": 16, "gab_hidden": 64, "gab_templates": 64
```
That's **16/64/64**, not the **d1=32, d2=d3=64** your question described.
Worth double-checking which config you're recalling — they don't match.

"21.7%" appears **zero times** in the repo. No GAB ablation table or
per-config delta numbers exist in `docs/` or `scripts/research/` anywhere.
Two research-backlog files (`scripts/research/remaining_research_topics.
md`, `scripts/research/arena_agent_diffusionblocks_prompt.md`) explicitly
flag "go find Chessformer/1e4.ai/ChessMimic parameter counts and figures"
as a *future*, not-yet-done research task.

**Bottom line**: any real Chessformer-paper ablation figures need external
lookup — this repo neither cites nor reproduces them, and its own GAB
dimensions differ from the ones in your question. If you have a real
21.7% top-1 measurement, it's your own project's number, not something to
compare against a paper figure that isn't actually in this repo either.

## 6. Engine history

**(a) Why the NNUE output head is unbucketed**: confirmed reason, from
`docs/research-notes-moe-2507.11181.md:40-88` — it's **not a deliberate
architectural choice**, it's simply not built yet. Direct quote: *"Modern
Stockfish-lineage NNUEs use 8 output buckets selected by piece count...
The shipped `unchessed-nnue.bin` has one head; you cannot synthesize eight
from it... Why this is not a code change today: it requires retraining."*
The doc's own recommendation is to add bucketing in the next NNUE training
run. No in-code comment in `nnue.rs` explains this further — the research
note above is the only documentation.

**(b) SPSA vs. hand-set `SearchParams`**: explicitly hand-set, not SPSA.
`unchessed-core/src/search.rs:17-20`'s own doc comment: *"Defaults match
the values that were previously hard-coded, so behavior is unchanged
unless a caller explicitly overrides them"* — SPSA is mentioned only as a
**future** aspiration ("eventually driven by automated tuning, e.g.
SPSA"), never as something already applied. By contrast,
`PassedPawnMgPct`/`PassedPawnEgPct` *are* validated, but via SPRT (paired
game-play testing), not SPSA (gradient-based self-play tuning) —
`unchessed-core/src/uci.rs:281-285`: *"SPRT-validated default (+25.7 Elo,
2026-08-02)"*. No `SearchParams` field in this repo has ever actually been
SPSA-tuned.

**(c) Apex v1 / Hydra v1-v4 failure modes**: no documented technical
failure reason exists for any of them — this is a real, confirmed gap,
not something either of us missed. `docs/unarchitectured-v1.md:3-8,88-96`
frames them only as *"experimental lineage labels... superseded"* with a
"contributions carried forward" table (what each version added, never why
it was superseded). `scripts/sprt-history/` has SPRT scripts for named
*features* (conthist, king safety, mobility, etc.), none scoped to Apex or
Hydra by name. Contrast this with the NNUE v3→v4 rewrite
(`tools/train_nnue.py:14-28`), which *does* have exactly the kind of
writeup you'd want — *"v3 was SPRT-gated at -70.3 Elo vs v1... three
concrete, verifiable deviations from the reference HalfKAv2_hm design"* —
that level of detail simply was never produced for the Apex/Hydra
predecessors. If it exists, it's not in this repository.

## Summary

Three of your six questions (1, 3, 6c) point at real, confirmed gaps in
this repo's own documented history — accurate assessments, worth treating
as genuinely unknown rather than re-derivable. Three others (2, 4, 5)
reference specific figures or docs that **don't exist anywhere in this
repository** — the fishtest/quantization notes doc, the "2.06x" figure,
the rating-inertness sweep result, the "21.7%"/d1=32 GAB numbers. Worth
tracing where those came from before relying on them further: either
they're from outside this repo's own history (external literature,
another project, a different session's untracked work) or they didn't get
committed anywhere retrievable. Either way, this repo can't currently
confirm them.
