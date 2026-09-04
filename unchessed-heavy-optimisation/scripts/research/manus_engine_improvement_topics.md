# Engine-strength topic list for `manus/research-facilities`

Everything below was checked against the actual current code, not
assumed — each item says what's already there and what's genuinely
missing. Same tiered structure and credit discipline as the original
research brief: Tier 1 is cheap (design/analysis/small bounded code),
Tier 2 needs a posted plan first, Tier 3 needs explicit human go-ahead.
Same standing rules apply: no default flip or search-behavior change
without a real paired-game SPRT; the three named `aegis_v4_runtime.rs`
parity gates must keep passing; state what you verified vs. assumed.

## What's already implemented (don't propose these as new — verify first, this list exists so you don't re-discover it the hard way)

Confirmed present in `unchessed-core/src/search.rs` / `uci.rs`:
reverse futility pruning, null-move pruning, late-move reductions,
aspiration windows, ProbCut (with an optional SEE filter), plain
futility pruning, killer-move and history-heuristic ordering, MVV-LVA
capture ordering, repetition detection (bounded scan, tested against a
full scan), Lazy SMP multi-threaded search sharing one lock-free TT,
exact `go nodes N` enforcement, and a dozen of the above as live UCI
options with a consistency-checker (`tools/check_search_param_consistency.py`)
already covering 24 options. Quiet-position NNUE label filtering
(M1/M2, arXiv:2412.17948) exists in `unchessed-datagen`. Passed-pawn,
mobility, rook, and knight-outpost evaluation terms are already
UCI-tunable (`PassedPawnMgPct/EgPct`, `MobilityPct`, `RookPct`,
`KnightOutpostPct`). A prior king-safety tuning attempt exists in this
project's history and was reverted after a real SPRT — check
`docs/parameter-calibration-audit.md` and any king-safety-named
docs/branches before re-proposing king-safety work from scratch.

## Tier 1 — cheap research/design, do freely

### 1. Search extensions: genuinely absent, worth a real look

**Confirmed**: there are no search extensions of any kind in
`search.rs` — no check extensions, no singular extensions, no
recapture extensions. This is a real gap relative to most engines at
this level of pruning sophistication (which already has RFP/NMP/LMR/
ProbCut/aspiration/futility, an unusually complete pruning suite for an
engine with zero extensions). Research, don't implement yet: which
extension would plausibly help most here given what's already pruning
aggressively (over-pruning without any compensating extension is a
known failure mode), starting with the cheapest and most standard —
check extensions (search one ply deeper when in check, since check
positions are exactly where pruning is riskiest and tactics are
sharpest). Design the fixed-position test suite and the SPRT plan
before writing search-changing code.

### 2. Quiescence search: does it prune bad captures via SEE?

Check (don't assume) whether `qsearch` in `search.rs` prunes clearly
losing captures using the existing `see.rs` SEE implementation, or
whether it searches every capture regardless of material outcome. If
it doesn't, that's a very standard, usually-safe qsearch optimization
(delta/SEE pruning) worth scoping — cheap to test since it only affects
qsearch node count, not correctness, and has a long track record in
other engines. If it already does this, say so and move on — don't
manufacture a finding.

### 3. Internal iterative reduction/deepening (IIR) for missing TT moves

When a node has no TT move to order first (a TT miss or an aged entry),
many modern engines reduce that node's depth slightly rather than
searching it at full depth with no ordering hint — a standard technique
alongside the pruning already present. Research whether this fits the
existing search structure and would plausibly help, given how much
pruning already depends on good move ordering here.

### 4. Multi-cut pruning research

A generalization of null-move pruning (try several moves at reduced
depth; if enough of them fail high, cut the node). Research fit and
expected value given NMP already exists — this may be redundant with
what's already there, or a real complement. Be honest if the answer is
"marginal given the existing NMP/ProbCut coverage."

### 5. Endgame tablebase (Syzygy) probing: research only, likely Tier 2/3 to implement

**Confirmed absent**: no tablebase support anywhere in the codebase.
This is one of the highest-confidence, best-understood Elo gains in
engine development (perfect endgame play + faster, more accurate draw/
win detection near the 50-move mark), but it's a real integration
project: a WDL/DTZ probing library, root-move filtering, and search-time
probing at shallow depth near the tablebase's piece-count cutoff. Scope
this properly before touching code: what tablebase files are actually
obtainable (size — even 3-4-5-piece Syzygy tables are a real download),
what crate/library would be used, and where root-probing and in-search
probing hooks would go. This is very likely Tier 2/3 once scoped (real
integration work, needs real endgame test positions and an SPRT), but
the research/scoping itself is Tier 1.

### 6. Pondering (thinking on the opponent's clock)

**Confirmed absent**: no `ponder` handling in the UCI layer. This is a
real practical feature (most GUIs support it, and it's effectively free
Elo in games with real clocks since the engine gets extra thinking time
it currently just doesn't use) but needs careful state handling —
research what `go ponder` / `ponderhit` / `stop` need from the existing
worker-thread structure in `uci.rs` before touching it, since this
interacts with the existing time-management and worker-lifecycle code.

### 7. Chess960 / Fischer Random support

**Confirmed absent.** A real feature many GUIs and tournaments expect
(`UCI_Chess960` option, castling-rights encoding changes, FEN parsing
changes). Scope what's actually required — this touches move generation
and FEN parsing, not just UCI — before proposing implementation.

### 8. Continuous testing infrastructure (meta, not chess)

Every SPRT in this project's history has been a manually-orchestrated
one-off run on whoever's real hardware was available. Research whether
a lightweight, self-hosted continuous-testing setup (in the spirit of
Stockfish's fishtest, scaled to this project's size) would be worth
building — something that could pick up a queued list of parameter/
search candidates and run them unattended against the current default,
instead of every SPRT needing a person to notice, provision hardware,
and babysit it. This is real infrastructure work, not a chess
improvement directly, but it would make every other item on this list
and the original research brief faster to actually validate. Scope the
smallest version that would help before proposing to build it.

## Tier 2 — needs a posted plan first

Anything from Tier 1 above that concludes "this is worth implementing"
graduates here. Post the plan (what changes, what the fixed-position
test suite looks like, what the SPRT will measure) and wait for
acknowledgment before writing search-behavior-changing code, per the
standing rules — this project has one well-documented catastrophic
failure (round 0, an unvalidated hint wired directly into search) that
is the entire reason this gate exists.

## Tier 3 — real compute, real SPRT, explicit human go-ahead

Tablebase file acquisition/hosting, any full search-extension or
IIR/multicut candidate's real SPRT campaign, and Chess960 support's
final validation all belong here once scoped. Same as every other
Tier 3 item in the original brief: don't start without explicit
go-ahead, not just an acknowledged plan.
