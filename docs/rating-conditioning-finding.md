# The model's rating input does nothing

Prompted by the Maia opponent ladder (Maia 600 through Maia 2600 — a separate
network per rating band) and by Chessformer
([arXiv:2605.19091](https://arxiv.org/abs/2605.19091)), which takes the other
approach: one model *conditioned* on rating, reporting state-of-the-art human
move-matching from a single rating-conditioned network.

Unarchitectured v1 is built for the conditioned approach. It has a `rating`
input projected through `rating_weight`/`rating_bias` into the history vector,
and a `policy_kind` input selecting `POLICY_HUMAN` (0) or `POLICY_GUIDE` (1)
via LoRA-adapted policy heads. Both are live in the shipped engine: `uci.rs`
clamps `UCI_Elo` to 500..3200 and passes it straight through as the rating.

**Neither input had ever been evaluated.** Every calibration, ablation and
analysis tool in this repository hardcodes `--rating 2700` and
`--policy-kind 1`.

## Result: the rating input is inert

`tools/analyse_rating_conditioning.py`, 200 real positions, full Maia-style
sweep:

| rating | top-1 changed | mean \|Δlogit\| | max \|Δlogit\| | teacher agreement |
|---:|---:|---:|---:|---:|
| 600 | — | — | — | 0.2650 |
| 1000 | **0 (0.0%)** | 0.000304 | 0.000633 | 0.2650 |
| 1400 | **0 (0.0%)** | 0.000607 | 0.001266 | 0.2650 |
| 1800 | **0 (0.0%)** | 0.000911 | 0.001900 | 0.2650 |
| 2200 | **0 (0.0%)** | 0.001214 | 0.002533 | 0.2650 |
| 2600 | **0 (0.0%)** | 0.001518 | 0.003166 | 0.2650 |
| 3200 | **0 (0.0%)** | 0.001973 | 0.004115 | 0.2650 |

**Across the entire 2600-point span the chosen move never changes once.** The
largest logit perturbation anywhere is 0.004 — smaller than the gaps between
adjacent candidate moves by two to three orders of magnitude (typical top-2
gaps in this model are 0.1–1.9, see
`benchmarks/unarchitectured-v1/hint-disagreements-2026-08-24.json`). Teacher
agreement is byte-identical at 0.2650 for every rating.

The response is also suspiciously *linear* in rating — deltas scale almost
exactly with the rating value (0.000304 at 1000 → 0.001973 at 3200, a near
perfect ratio). That is the signature of a single scalar term added into a
32-wide vector and then swamped downstream, which is precisely what
`normalized_rating * rating_weight + rating_bias` is.

`POLICY_HUMAN` vs `POLICY_GUIDE` at fixed rating changes the move in **4/200
(2.0%)** of positions. Non-zero, but far too weak to call a persona.

## What this does and does not mean

**It does not mean `UCI_Elo` is broken.** The option genuinely works through
the adaptive move-selection path — that mechanism was itself the subject of an
earlier bug fix (`decide_mode()`/`select_move()` were unreachable when
`Adaptive=false` was combined with `UCI_LimitStrength=true`, found by a
64-level Elo-ladder stress test). Strength limiting works. What is inert is
the rating's effect on the **neural policy hint** specifically.

Since `UnarchitecturedHint` is default-off, no shipping behaviour is affected
today. The finding matters for what it says about the model:

- **A whole conditioning pathway trained into the network is doing nothing
  measurable.** The parameters exist, are loaded, and are multiplied — they
  simply do not influence the output.
- **The Maia-style use case is unavailable.** Anyone hoping to use this model
  to play at a chosen human strength cannot: it produces the same move at 600
  as at 3200.
- **It compounds the GAB finding.** `docs/gab-capacity-finding.md` showed the
  positional-encoding component is provisioned at a quarter of the paper's
  smallest configuration. This shows the rating conditioning is inert. Both
  point the same way: the architecture's *capacity allocation* is where the
  problems are, not the kernel speed that rounds 2–4 optimised.

## If this is ever retrained

Chessformer's approach suggests the conditioning has to be given enough
influence to matter — the rating signal needs a path that is not a single
scalar diluted through a 32-wide projection into a 256-wide model. Their
Maia-3 result is the existence proof that one conditioned model can work.

Concretely, and in priority order alongside the earlier findings:

1. widen GAB to at least the paper's 5M configuration (d1=32, d2=d3=64);
2. give rating a real conditioning path, and verify it with this tool before
   trusting it. `docs/research-notes-maia-levels-reverse-engineering.md`
   reverse-engineered the Maia ladder's three generations and extracts the
   two conditioning designs that demonstrably work at scale (Maia-2's
   discrete self/opponent bucket embeddings; Maia-3's interpolated anchor
   embeddings) — both take TWO inputs (self and opponent skill), not the
   single scalar this doc found inert. `tools/build_level_conditioned_moves.py`
   already emits the dual-elo move labels a retrain would train on;
3. theme-balanced sampling toward quiet/mate/fork positions
   (`docs/unarchitectured-v1-theme-breakdown.md`);
4. weight clipping so the result is quantizable
   (`docs/fishtest-and-quantization-notes.md`).

All four are retrain-only, and any retrained net still needs its own SPRT.

## Honest limits

- **200 positions, not the full 600**, to keep the seven-rating sweep
  affordable. The result is 0/200 with zero variation in teacher agreement, so
  more positions would not change the conclusion — but it is a subset.
- This measures the **exported student checkpoint**. The 58M oracle may
  condition properly; nothing here tests that.
- The corpus is over-the-board tournament play (min Elo 2300), so it is not a
  natural test set for *low*-rating human emulation. That does not affect the
  finding, since the question asked is whether the input changes the output at
  all, and it does not.
- Offline analysis; no games played.

## Syzygy, briefly

The Advanced-topics page adds implementation detail (Syzygy uses DTZ not DTM,
the engine pre-selects TB-good root moves and searches only those, 7-man needs
~17 TB of storage and a raised `ulimit -n`). None of it changes last round's
conclusion in `docs/stockfish-empirical-data-notes.md`: their own measurements
put 6-man Syzygy at +2.7 Elo for SF15 in RAM and −1.26 ± 1.46 from SSD for
SF17.1. Still not worth building here. Worth noting the page states plainly
that "tablebases bring only a limited increase in strength" and links the same
Elo data.
