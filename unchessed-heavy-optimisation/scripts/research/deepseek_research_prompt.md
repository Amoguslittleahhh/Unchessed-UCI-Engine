# Research request: Unchessed AI — triage and deep-dive on the 200-idea backlog

## Context

Unchessed AI is a from-scratch Rust UCI chess engine (repo:
github.com/Amoguslittleahhh/Unchessed-UCI-Engine). Current state, briefly:

- **Search**: alpha-beta with PVS, quiescence, TT, null-move, LMR,
  killers/history, MultiPV, clock-aware time management.
- **Eval**: NNUE v4 — HalfKAv2_hm features (32-bucket horizontal king
  mirroring, own-king as an active feature, factorized training), 256-wide
  accumulator, plain SCReLU output head, 22,528 inputs. SPRT-validated
  +26.1 ± 12.4 Elo over v1 (2,251 games, LOS 100%). A prior architecture
  (v3, using a Stockfish-SFNNv5-style concatenated output head with the
  same feature set) failed SPRT badly: -70.3 ± 22.1 Elo, LOS 0%, in 756
  games — the fix that produced v4 was the feature-set/output-head
  redesign, not a data or hyperparameter change.
- **Confirmed via direct source inspection (not assumption)**: the NNUE
  has **no incremental accumulator updates** — every eval call does a
  full recompute of all 22,528 features for both perspectives. This is
  the single highest-confidence, highest-priority item in the whole
  backlog; treat it as a hard prerequisite for several other ideas below
  (anything about accumulator refresh strategy, wider accumulators, or
  eval-cost-sensitive search heuristics assumes this exists first).
- **Human policy net**: Maia-style, per-rating-bucket (<1300, 1300-1599,
  1600-1899, 1900+), trained on 19.9M positions from 781k Lichess games.
  Top-1 accuracy 29.2%-33.5% depending on bucket; en-passant accuracy is
  notably weak at <1300 (18.8%) vs 1900+ (67.4%) — plausibly real human
  behavior, not necessarily a network deficiency (unverified).
- **Adapter/persona system**: live Bayesian opponent-Elo estimate from
  centipawn loss (converges ~8-12 moves), four personas (MATCH/PUNISH/
  CLINCH/DEFEND) with hysteresis-gated transitions, engine-tell detection
  from move-timing + quality, an opening book with a troll tier gated by
  the live Elo estimate.
- **Reviewer tool** (PGN move classification, accuracy %): not built yet,
  roadmap only.

Two prior research passes already exist and should NOT be re-derived from
scratch:
1. A 19-question deep-dive (NNUE ablation, king buckets, accumulator
   width, dual-net, quantization, incremental updates, WDL exponent,
   feature sets, output head, Syzygy labels, curriculum learning,
   adversarial mining, label noise/bootstrapping, SEE, correction/
   continuation history, time allocation, a pre-SPRT smoke test, time-
   pressure policy input, blunder-rate calibration) — already answered
   and largely trustworthy (spot-checked against real arXiv papers and
   the actual codebase).
2. A second pass covering 14 longer-horizon topics (tablebase probing,
   pondering, MultiPV-weighted move selection, puzzle-rush mode,
   accumulator refresh strategy, style transfer, anti-fingerprinting,
   explainability, self-play league, NN move ordering, NN pruning,
   endgame hybrid eval, opponent-style-conditioned training,
   transformer-as-labeling-oracle) — also largely trustworthy, with one
   recurring, confirmed-wrong claim to avoid repeating: it mislabels the
   -70.3 Elo SPRT failure as "v2 (HalfKA, 45056 features)" — the real
   failure was **v3** (HalfKAv2_hm, 22,528 features); the actual v2 was
   just v1's architecture retrained on more data and was never
   SPRT-gated at all.

Attached/referenced alongside this prompt: `200_research_ideas.md`, a
broad one-line-per-item brainstorm of 200 further research directions
across 14 categories (NNUE architecture, training data, search, movegen,
opening book, persona/adapter, policy net, Reviewer tool, testing,
calibration/deployment, longer-horizon/exploratory, UCI tooling,
documentation, benchmarking/community/ecosystem). None of these 200 have
been researched yet — that's the job here.

## What went wrong with a previous attempt at this — read before starting

A different large research document was generated for this same 200-item
list by another model running as 100 parallel subagents. It produced
~1.67 million words across 2,777 pages. When spot-checked, it contained:
- Fabricated citations (a wrong named author attributed to a real,
  correctly-known technical contribution — HalfKAv2_hm mirroring is
  Tomasz Sobczyk's work, not "Infuehr" as that document claimed twice
  with two different fake PR numbers).
- A self-generated internal audit at the end of the document admitting
  the length existed to clear an artificial page-count target, not
  because the content needed that space, and rating its own work 6.5/10.

**Do not repeat this failure mode.** Depth should come from genuine
uncertainty and genuinely non-obvious tradeoffs, not from padding every
item to a uniform word count. A one-paragraph answer is correct and
complete for an idea that has an obvious, well-established answer. Save
real depth for the items that actually warrant it. Never invent a named
author, PR number, or specific benchmark figure — if you're not certain
a citation is real, say "unverified" or omit it rather than presenting
invented specificity as fact.

## What I want

**Step 1 — Triage all 200 ideas.** For each of the 14 categories in
`200_research_ideas.md`, sort its items into three tiers with a single
sentence of rationale each:
- **Tier A (well-established, do it):** standard technique, known payoff,
  low risk.
- **Tier B (genuinely open, worth a real experiment):** unclear whether
  it helps at this project's scale (200K-23M NNUE params, single
  developer, Rust, from-scratch — no Stockfish codebase inherited).
- **Tier C (low priority / speculative / skip for now):** interesting but
  not worth pursuing yet, with a one-sentence reason why not.

**Step 2 — Deep-dive the top 15-20 items** (your judgment on which matter
most, weighted toward Tier A/B items with the highest expected impact —
the confirmed-missing incremental accumulator updates item and anything
that directly gates it should be at or near the top). For each deep-dive
item, cover:
1. **Prior art**: how established engines/projects have approached it
   (Stockfish, Leela/lc0, Maia, nnue-pytorch, or general ML/search
   literature as relevant) — only cite sources you're actually confident
   exist.
2. **Established vs. open**: is this a known win or a real research
   question at this project's scale?
3. **Concrete recommendation**: worth doing now, worth doing after some
   prerequisite, or not worth doing — and why.
4. **Pitfalls specific to Unchessed**: interactions with the existing
   king-mirroring bucket scheme, the persona/policy-net system, the lack
   of incremental accumulator updates, or the Rust implementation
   context.

**Step 3 — One paragraph of overall synthesis**: given everything above,
what are the 5 highest-leverage next actions, in order, and why.

## Output format

Keep Step 1's triage compact (one line per idea — this should read like a
sorted list, not 200 essays). Spend the real word budget on Step 2's 15-20
deep-dives. No fixed minimum length per item — let genuine complexity
drive length, not a target page count.
