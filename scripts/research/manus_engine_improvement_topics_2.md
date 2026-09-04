# Engine-strength topics, part 2 — checked against the code, same rules as part 1

Continues `scripts/research/manus_engine_improvement_topics.md` — same
Tier 1/2/3 structure, same standing rules (no default flip without a
real SPRT, three named parity gates must keep passing, state verified
vs. assumed). One correction to part 1's framing while I was at it:
**time management is more sophisticated than I implied there** —
`search.rs` already has situational (sharp-vs-quiet position) soft-time
scaling and a real panic-mode fallback, tested down to the millisecond
budget. Don't propose "add adaptive time management" as if it's
missing; if there's a refinement worth making, it's to what's already
there, not a green-field feature.

## Tier 1 — cheap research/design

### 9. Razoring: confirmed absent, distinct from the futility pruning already present

At very shallow depth, when the static eval is far below alpha, many
engines drop straight to quiescence search instead of doing a full
reduced search — cheap, well-understood, and complements (doesn't
duplicate) the futility pruning already in `search.rs`. Research fit
and interaction with the existing RFP/futility margins before proposing
a candidate — these techniques can interact in ways that need a
combined fixed-position/SPRT check, not independent validation.

### 10. Late-move / move-count pruning: confirmed absent, distinct from LMR

LMR (already present) reduces the *depth* of late quiet moves. Late-move
pruning instead *skips* them entirely once a move-count threshold is hit
at shallow remaining depth — a different lever, commonly used alongside
LMR rather than instead of it. Research whether this is likely additive
given how much late-move reduction/pruning already exists here (LMR +
futility + RFP is already a lot of coverage in this exact space — be
honest if the marginal value looks small).

### 11. Countermove heuristic + continuation history: confirmed absent

Killers and history (both present) order moves by how good they were
*in general*. Countermove heuristic and continuation history instead
order by how good a move was as a *response to the opponent's last
move* (or last two) — a well-established, usually cheap-to-implement
addition to existing move-ordering infrastructure since killers/history
already exist to build on. Worth scoping as one of the more standard,
lower-risk items on this whole list.

### 12. UCI protocol completeness: `searchmoves`, `go mate N`

**Confirmed absent**: no `searchmoves` restriction (GUIs use this to
restrict analysis to specific candidate moves — a real, expected UCI
feature for analysis use, not just play) and no `go mate N` mode
(search specifically for a mate in N plies). Both are protocol
completeness gaps, not strength questions — lower priority than the
search items above, but cheap and low-risk to add since they don't
touch pruning/move-ordering logic, just the UCI command surface.

### 13. WDL-style output (win/draw/loss percentage reporting)

**Confirmed absent** as a UCI output. The NNUE is already trained with
a WDL-blended target (`train_nnue.py`'s `0.7·σ(cp/400) + 0.3·(wdl/2)`
loss) — there's already a real signal here, just not surfaced to the
GUI. Several modern engines report a `wdl` field in `info` output
(win/draw/loss per mille) for user-facing value. Research whether a
principled win/draw/loss estimate can be derived from the existing eval
scale (a real statistical fit, e.g. logistic in cp, calibrated against
this engine's own game outcomes — not just repurposing NNUE's internal
training blend, which isn't the same as a calibrated outcome
probability) before proposing to expose one.

### 14. Insufficient-material / dead-position draw detection

**Confirmed absent**: no explicit KvK / KB-vs-K / KN-vs-K / same-colored-
bishops draw detection anywhere in the codebase. Worth checking first
whether this matters in practice here — if the NNUE/HCE eval already
scores these positions near zero reliably and the engine converges to a
draw-ish result anyway, this may be a low-value addition; if there are
real cases where the engine misjudges a dead position as winning
because material is nonzero, it's a correctness-adjacent fix worth
making. Verify with real positions before deciding priority.

### 15. Transposition table aging/generation

**Confirmed**: `tt.rs`'s replacement policy is depth-preferred with no
visible generation/aging counter. In a long analysis session or a
multi-game match without `ucinewgame` between games (if that ever
happens in practice — check whether the GUI/harness this engine is
actually used with reliably sends `ucinewgame`), stale entries from a
much earlier position could linger and cost a rare suboptimal
replacement. Research whether this is a real practical issue given how
this engine is actually deployed (cutechess/GUI always sends
`ucinewgame`?) before treating it as a priority — it may be a
non-issue in practice.
