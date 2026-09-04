# Unchessed AI — corrections + new topics for `manus/research-facilities` (update 3)

Paste this alongside (not instead of) `manus_master_prompt.md` — it
corrects two stale items in that file, closes out the NNUE-ceiling
thread with two new real results, and adds a fresh batch of topics
found by re-scanning the codebase against a checklist of common engine
features not yet covered by any of the 32 existing investigation docs.

Same standing rules as always: **3,422-credit budget, main is
read-only, work on `manus/research-facilities`, post a plan before
Tier 2/3 spend, state verified vs. assumed, report negative results
plainly.**

**New standing rule for everything below: real-world testing is
mandatory wherever it's feasible within Tier 1's budget, not optional
polish.** For every topic that has a concrete code path to check (not
just a design question), that means: actually install whatever's
needed (`cargo build`, `pip install`, a Rust probing crate, whatever
the topic requires), actually run it against real positions/data, and
report real output — not "this should exist," not a literature
citation standing in for a check, not a design doc with no run behind
it. The check-extensions correction above is the cautionary example:
a claim that skipped actually reading the code (or in that case,
running it) turned out to be flatly wrong and wasted a whole item's
worth of the master prompt. Where a topic is genuinely design-only at
Tier 1 (e.g. deciding whether Syzygy integration is worth the cost
before writing any code) say so explicitly and explain why real
testing isn't possible yet — don't let "design-only" become the quiet
default for something that could actually be run cheaply.

---

## Corrections to `manus_master_prompt.md` — do these first, they're free

**Tier 1 items 6 and 7 in that file are wrong and already resolved.**
Manus's own `docs/reinforcement/14-check-extensions.md` and
`15-qsearch-see-pruning.md` found, with exact line numbers, that:
- Check extensions **do exist** (`search.rs:716-717`:
  `let ext = if gives_check { 1 } else { 0 }; let nd = depth - 1 + ext;`,
  plus a check-evasion depth floor around line 525).
- Qsearch **does** have real SEE-based pruning, not just ordering
  (`if !in_chk && scores[i] < -1_000_000 { continue; }`).

I independently re-read the code and confirmed both. **Delete or
strike through items 6 and 7 in the master prompt; don't re-spend
credits re-verifying them.** This was my error originally, not
Manus's — flagging it plainly rather than quietly editing the file out
from under anyone, per the project's own "report negative results (and
mistakes) as plainly as positive ones" rule.

If there's an appetite for search extensions research after this
correction, the honest open question is narrower than item 6 claimed:
singular extensions and recapture extensions are still apparently
absent (worth a quick grep to confirm before researching further) —
check extensions themselves are done and don't need re-inventing.

---

## NNUE-ceiling thread: two new real results, thread closed on this axis

Since the master prompt was written, the reviewer ran the actual
coverage-vs-capacity experiment `13-nnue-ceiling.md` designed, twice,
plus a follow-up:

1. **`docs/nnue-coverage-capacity-matrix-178m.md`** — hard
   piece-count-bucket resampling on the real 178M self-play corpus
   (27M-record pool, real depth, no duplication confound). Result: 4
   of 7 rare/mid buckets improved 5.7%-37.2% relative MAE, but the two
   most common buckets (57% of real games) regressed 10.9% and 13.9%.
   A genuine trade-off, not a clean win, in either direction.
2. **`docs/nnue-soft-reweighting-result.md`** — tried the cheaper
   alternative predeclared in `docs/reinforcement/32-nnue-soft-reweighting.md`
   (per-sample loss weighting instead of hard resampling, inverse-pool-frequency
   table, clipped [0.25x, 20x]). Result: **made rare buckets worse**,
   not better (bucket 2 regressed 19.5%), because those buckets have so
   few real training examples (44 and 1,030 out of 98,000) that
   upweighting them just amplifies overfitting to those specific
   examples rather than teaching generalizable rare-position structure.
   Root cause verified directly, not guessed at.

**Per the decision rule predeclared in doc 32 before that result
existed, this closes out piece-count-bucket reweighting as a research
direction at this data scale.** Update `13-nnue-ceiling.md` and
`31-tier1-master-synthesis.md` to reflect both results and this
closure.

**What's actually left open on NNUE ceiling, if pursued further**:
not another sampling/reweighting variant (diminishing returns now
demonstrated twice) but (a) the capacity axis doc 13 deliberately
deferred (wider model at fixed data), or (b) whether the
hard-resampling trade-off (better rare-bucket MAE, worse common-bucket
MAE) nets positive or negative in **actual games** — held-out MAE
across buckets doesn't establish which way Elo moves, and that
question hasn't been tested at all yet. A small SPRT/match comparing
the hard-BALANCED net against the shipped default would be the honest
next step if this axis is still worth resourcing, rather than a bigger
version of the same MAE diagnostic.

Given two real negative/mixed results in a row here, it's also
reasonable to just close this thread and redirect effort — your call,
consistent with "report negative results as plainly as positive ones"
rather than manufacturing a reason to keep going.

---

## New topics: engine features not covered by any of the 32 existing docs

Found by checking, not assuming — each has real evidence from a fresh
codebase scan (file:line citations below), specifically to avoid
repeating the check-extensions mistake. Spot-check the citations before
relying on them; a fresh scan can still miss things a targeted read
catches.

### Confirmed absent — real gaps

24. **Endgame tablebase probing.** No matches anywhere in
    `unchessed-core/src` for tablebase/Syzygy/`.rtbw`/`.rtbz`. Zero TB
    support. Research whether Syzygy probing (the near-universal
    standard, small WDL+DTZ files, permissively-licensed probing
    crates exist in Rust) is worth the integration cost, distinct from
    file *acquisition/hosting* (already flagged as Tier 3 in the master
    prompt) — the code-integration research itself is Tier 1 and free.
25. **Pondering.** No `ponder`/`go ponder`/`ponderhit` handling
    anywhere in `uci.rs` or `search.rs`. Research whether this is worth
    adding given the engine's existing time-management sophistication
    (situational time scaling, panic-mode fallback) — pondering
    interacts with time management in ways worth designing carefully,
    not bolting on.
26. **Classic pawn-structure eval terms.** Passed pawns exist and are
    tuned; isolated pawns, doubled pawns (as a standalone penalty, not
    just passed-pawn dedup), backward pawns, and pawn islands do not.
    Research which of these is most likely additive given how much
    other positional eval (mobility, outposts, rook files) already
    exists — these are some of the oldest, most standard eval terms in
    chess programming and their total absence is worth investigating.
27. **NNUE inference is float32 throughout, not quantized.**
    `nnue.rs`'s own doc comment states inference is f32, with AVX2
    float SIMD kernels. No int8/int16 quantized path. This is standard
    in essentially every competitive NNUE implementation (Stockfish's
    NNUE is int8/int16) specifically for the 2-4x inference speedup it
    buys. Research the speed/accuracy tradeoff and integration cost —
    this could matter more for practical strength than several of the
    NNUE-ceiling items already investigated, since search speed
    directly trades off against search depth.
28. **Chess960 castling representation.** No matches for
    "960"/"chess960"/"frc" in engine code; castling rights use a fixed
    KQkq bitmask with no generalized castling-rook-file support. The
    master prompt already lists "Chess960's final validation" as Tier
    3, implying partial work exists somewhere — research and report
    what state that's actually in (design-only? partially coded
    elsewhere? genuinely nothing?) since this scan found no trace in
    the core crate.
29. **No `bench` command / fixed NPS regression suite.** No standard
    UCI `bench` subcommand or fixed benchmark position set for tracking
    nodes/sec across commits — only ad-hoc microbenchmarks in
    `aegis_v4_runtime.rs` testing network/hint code specifically, not
    general search throughput. Research whether a lightweight
    fixed-position `bench` command (a few dozen standard positions,
    fixed depth or node budget, report total nodes/sec) is worth adding
    — this is cheap, and pairs naturally with the continuous-SPRT-infra
    item already in the master prompt as basic regression-tracking
    infrastructure.

### Present but soft/partial — worth a lighter look

30. **Contempt exists but appears untuned.** `Contempt` UCI option
    (default 25, range 0-100) feeds into draw scoring, but unlike
    knight-outpost (which cites a specific SPRT result in `eval.rs`),
    no validation was found for the contempt value or its effect.
    Research whether the default is arbitrary or has any real
    backing, and whether it's worth a validation pass.
31. **Lazy SMP helper-thread depth staggering uses a fixed
    modulo-3 cycle**, not something that scales with thread count
    (`uci.rs` around 1659-1685: offsets cycle 1,2,3,1,2,...). Research
    whether this saturates (stops adding useful search diversity) past
    a handful of helper threads, and whether a scaling scheme would
    help on higher core-count hardware.
32. **TT (hash table) size doesn't auto-scale to available system
    memory** — fixed default (128 MB), user must set `Hash` manually.
    Minor, common omission; research whether auto-sizing (a fraction of
    detected system RAM, common in other engines) is worth adding as a
    quality-of-life default.
33. **AVX2 SIMD unsafe blocks in `nnue.rs`'s hot inference path**
    weren't seen to carry explicit `// SAFETY:` comments justifying
    their preconditions (CPU feature-detection gating, slice-length
    invariants) in a quick pass. Research/verify this is actually
    guarded correctly (feature-detected before use, no way to call the
    AVX2 path on unsupported hardware) — this is a correctness/safety
    audit, not a strength research question, but worth a careful look
    given it's literally the code that runs on every single evaluation.

---

## Reporting format

Same as the master prompt: one doc per investigation under
`docs/reinforcement/` (or `docs/` for non-RL topics), continuing the
numbering (next free number is 33+, or wherever the branch's own
history has already gotten to — check before numbering). State
verified vs. assumed, real evidence, and an honest recommendation.
