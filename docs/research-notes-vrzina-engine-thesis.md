# Research note — "Piece by Piece: Building a Strong Chess Engine" (Vrzina, VU Amsterdam BSc thesis, 2023-07-20)

Assessment of Sander Vrzina's bachelor thesis (supervisor W.J. Fokkink)
for relevance to this engine: a C++20/MSVC engine ("Tesseract") built
technique by technique, with each search feature benchmarked at fixed
depth on the Strategic Test Suite (STS).

## What it is

A construction log, not a method paper. Movegen: bitboards + 64-square
mailbox hybrid, PEXT slider lookups (BMI2; 5,248 bishop / 102,400 rook
entries, 861 KB; magic bitboards recommended as the portable fallback for
Zen2-and-older AMD / non-Intel), 16-bit packed moves (~50% movegen speedup
from the packing alone), legality at generation time including the
en-passant-pin edge case, bulk-counting perft (~405 MN/s perft-7;
Stockfish 15 does perft-6 in 2.2 s — different depths, bulk-count caveat
noted by the author). Search: features added one at a time (below).
Evaluation: basic hand-tuned HCE — material + positional, mobility,
king safety via castling-rights and pawn-shield heuristics (a
distance-based "king tropism" variant was tried and rejected as a
weak indicator), pawn structure. Engine play: UCI, time management = 1/20
of time remaining, draw detection partial (prevents voluntary
threefold in winning positions; cannot foresee opponent's forced draws),
no endgame tables. Final estimate **2400–2450 Elo**: the author's own
rough range from a ~30% win rate against CCRL-rated engines on their
hardware, not a controlled rating.

Per-feature measurements (their Table 7.1; STS depth 7; **cumulative** —
each row includes all previous features; aspiration windows and LMR were
tested individually at the end, not added; score max 15,000; EBF =
effective branching factor):

| Feature (cumulative unless noted) | Time (s) | Score | EBF |
|---|---|---|---|
| Alpha-beta | 25,820.4 | 7,176 | – |
| + transposition table | 14,504.0 | 7,151 | – |
| + iterative deepening | 4,352.9 | 7,307 | 6.94 |
| + MVV-LVA ordering | 709.5 | 7,314 | 5.99 |
| + quiescence | 266.2 | 8,520 | 4.23 |
| + PVS (alpha−1/alpha, off in qsearch) | 207.6 | 8,519 | 4.42 |
| + killer moves (2/ply) | 172.3 | 8,525 | 4.17 |
| + history heuristic | 163.3 | 8,522 | 3.92 |
| + null-move pruning (non-recursive) | 67.6 | 8,449 | 3.31 |
| + check extension (+1 ply in check) | 81.4 | 8,584 | 3.61 |
| aspiration windows (tested alone) | 83.9 | 8,584 | 3.82 |
| late move reductions (tested alone) | 64.1 | 8,124 | 2.90 |
| + Lazy SMP (4 cores) | 56.4 | 8,489 | 3.47 |

The **negative results** are the interesting part (details in the thesis):
aspiration windows — no speedup at any window size or widening strategy
tried; LMR — real speedup but Elo drop too large, removed; history as a
beta-cutoff signal — ineffective (a variant that increments on alpha
improvement helped ordering slightly); recursive NMP — Elo drop, kept
non-recursive; PVS inside quiescence — slightly worse; SEE — implemented
but too slow in their engine, MVV-LVA kept; Lazy SMP — 40% at 4 cores but
unresolved search instability/blunders the author could not debug.

## Audit: what this engine already has

Every movegen/search technique in the thesis exists in `unchessed-core` in
equal or more modern form. Static audit (grep/read; no Rust toolchain in
the sandbox, no Rust change in this round — nothing below required one):

| Technique | Tesseract | unchessed-core (evidence) |
|---|---|---|
| Slider attacks | PEXT (BMI2) | magic bitboards, compile-time (`movegen.rs:1`) — the portable path the thesis itself recommends |
| Packed moves | 16-bit, ~50% gain (§2.7) | 16-bit `Move(u16)`: `from \| to<<6 \| kind<<14` (`board.rs:83-102`) |
| Legality | at generation | in search: post-`make` `king_safe_after` filter in negamax (`search.rs:702`) and qsearch (`search.rs:456`) — the approach the thesis notes the faster engines use |
| TT | Zobrist, alpha/beta/exact, depth guard | same, plus depth-preferred replacement (`tt.rs:163,177`), `hashfull` reporting, child-position prefetch (`search.rs:701`) |
| PVS | alpha−1/alpha, off in qsearch | present, unnamed in code: non-first moves searched null-window, full re-search on the PV (`search.rs:772-776`) |
| Killers | 2/ply | 2/ply (`search.rs:254`; updated on cutoff at `search.rs:797`) |
| History | butterfly, last resort | butterfly `history[side][from][to]`, last resort (`search.rs:255,374`) |
| Qsearch ordering | MVV-LVA (SEE too slow) | full SEE *with pin handling* for ordering in search and qsearch (`see.rs`, `search.rs:362`) plus SEE pruning in qsearch (`search.rs:429`) |
| NMP | flat, non-recursive | depth-scaled `r = nm_base + depth/nm_divisor` (`search.rs:25-27,68`) |
| LMR | tried, removed (Elo) | `r = 1 + (late) + (non-PV)` with re-search (`search.rs:758-776`) |
| Aspiration | tried, no effect | kept: delta 25 from depth 4 (`search.rs:34-38,1003-1017`) |
| Check extension | +1 when in check | +1 when giving check (`search.rs:709`) — same one-extension conclusion |
| Futility / ProbCut / MultiPV / time mgmt | not in thesis scope | present (`search.rs:1,23,43-60`) |
| Multithreading | Lazy SMP, buggy | Lazy-SMP-style with staggered `start_depth` helpers (`search.rs:833-836`), `Threads` option capped by default (`uci.rs:86-87,238`) |
| Bulk-count perft | benchmarking aid | not needed: `perft.rs:6` is a correctness gate |
| Split `State` classes + visitors | ~20–25% movegen | n/a (Rust; no vtable in the path) |

Nothing on the thesis checklist is missing here that would be worth
porting: the two techniques where Tesseract kept the weaker variant (PVS
in qsearch territory, MVV-LVA instead of SEE) are present in the stronger
form in ours; the PEXT-vs-magic choice is already the portable one the
thesis recommends.

## Verdict

**Reference material, not a work item. No code change from this review.**

1. **The audit result is "nothing to port."** Tesseract is a strictly
   weaker stack than what `unchessed-core` already ships (hand-tuned HCE
   at ~2400 vs. NNUE v3 + optional Unarchitectured v1 prior here). Its
   value is the technique checklist, and on that checklist this engine is
   at-or-ahead in more modern form on every row.
2. **The negative results are the reusable part — as evidence for the
   gate policy, not a to-do list.** Aspiration, LMR, history, and
   recursive NMP each moved Elo the wrong way on Tesseract while the same
   techniques are load-bearing here (or vice versa, per configuration).
   The thesis's own diagnosis — benefit "heavily dependent on the quality
   and type of evaluation function" — is precisely why project policy
   requires a fresh SPRT per search change rather than porting "known
   good" flags. If one of these is ever A/B'd (e.g. the survey's MTD(f)
   item in `docs/research-survey-arxiv-2026-08-24.md`), the thesis shows
   the failure modes to watch for: fixed-depth Elo drop, multi-thread
   instability, speedup bought with blunders.
3. **The STS depth-7 cumulative ablation is a decent cheap protocol**
   (score + time + EBF per feature, fixed depth for low variance). We do
   pentanomial SPRT plus per-round ablations, which is stronger; a
   STS-style fixed-depth battery remains useful as a pre-SPRT smoke test
   for search changes, not as a gate (STS is positional; most of our
   changes are tactical).
4. **Elo caveat:** 2400–2450 is the author's rough self-estimate from
   win percentages against CCRL-rated engines on their hardware; not
   comparable to any number in this repo, cited only to situate the
   engine's strength.

## Not done

No code change (nothing to port). Rust not compiled (no toolchain in the
sandbox) — the audit is static. The thesis PDF was read through a 30-page
parser: the movegen and search chapters (the relevant ones) are complete;
the evaluation and engine-play chapters plus Table 7.1's EBF column were
recovered from search-engine snippets of the same document and are cited
as such.
