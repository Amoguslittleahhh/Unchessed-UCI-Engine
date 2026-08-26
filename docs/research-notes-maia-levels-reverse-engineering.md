# Reverse engineering the Maia opponent ladder — mechanism, training pipeline, and what it means for this engine

Prompted by the "Play Against Maia" ladder (Maia 600 → 2600+). `docs/rating-
conditioning-finding.md` already flagged the ladder as motivation for fixing
our inert rating input; this note goes into the actual source. All three
generations of Maia are open source, and the mechanism is fully readable —
no inference required.

Sources (pinned):

| Gen | Repo @ commit | Paper |
|---|---|---|
| Maia-1 | `CSSLab/maia-chess` @ `749204c` | KDD '20, arXiv:2006.01855 |
| Maia-2 | `CSSLab/maia2` @ `67bee6c` | NeurIPS '24, arXiv:2409.20553 |
| Maia-3 / Chessformer | `CSSLab/maia3` @ `1e13597` | ICLR '26, arXiv:2605.19091 |

## The three mechanisms, in one paragraph each

**Maia-1: one small model per level.** Nine Leela CNNs (64 filters, 6
residual blocks with SE ratio 8; policy *and* value heads) trained separately
for 1100–1900 in steps of 100. Each level's training set is Lichess
standard-rated games (2017-01→2019-11) where **both players' Elo fall inside
a 100-point window** at the target (`extractELOrange.py`), bullet games
removed, moves played with **< 30 s on the clock removed**
(`remove_low_time`), `pgn-extract` deduped, sampled at 20 blocks of 200 k
games per year (~12 M games/level) with a **time-ordered test split** (last
3 blocks of 2019). Deployment is `lc0` with `go nodes 1` — **no search**:
the policy head outputs the level's human move distribution, so the bot
plays the *average* move of a human in that window (the README notes the
models come out slightly stronger than the nominal rating). A separate
CNN of the same backbone is trained on **winrate-based blunder labels**
(a move is a blunder if it drops the mover's Stockfish-CP-derived winrate
below 0.1; CP comes from Lichess's Stockfish analysis, winrate via a
cp→winrate lookup table). The data pipeline emits a per-move CSV with the
schema: `game_id, result, white_elo, black_elo, time_control, move_ply,
move, cp, cp_rel, cp_loss, is_blunder, winrate, winrate_loss, blunder_wr,
opp_winrate, active_elo, opponent_elo, clock, opp_clock, clock_percent,
low_time, board...` — i.e. a move-level, level-labeled dataset.

**Maia-2: one unified model, discrete dual skill tokens.** A single
CNN(5 blocks, 256 ch) → 8×8 patch tokens → **ViT** (2 blocks, 1024 d,
16 heads) model predicts the human move *and* outcome for any skill pair.
Skill conditioning is **two discrete elo categories** — one for self, one
for the opponent — each mapped by `map_to_category` into **10 buckets**:
`<1100, 1100-1199, 1200-1299, …, 1900-1999, >=2000`, embedded (128 d each,
concatenated 256 d) and fed to the transformer. Three heads: policy
(all-moves logits, masked softmax over legal moves), a "side_info" feature
head, and a scalar value. Training data: Lichess 2018-05→2023-11, same 30 s
clock threshold, **first and last 10 moves of each game trimmed**,
`max_games_per_elo_range: 20` (same level-balanced sampling). Two
checkpoints: **rapid and blitz** (time-control-specific models). Inference
is `(fen, elo_self, elo_oppo) → move distribution`.

**Maia-3 / Chessformer: unified transformer, interpolated elo anchors,
UCI-native levels.** A family of pure-transformer engines (5M / 23M / 79M
params, HuggingFace `UofTCSSLab/Maia3-*`) with a **UCI level interface**:
`setoption name Elo value N` (plus `SelfElo` / `OppoElo`, all 0–5000) —
**this is the mechanism behind the platform's Maia-600…2600+ ladder**.
Elo conditioning is **continuous**: `interpolate_elo()` clamps the raw
elo to [0, upper] and takes a convex combination of **two learned anchor
embeddings** (a "low" and a "high" anchor), weighted by `elo/upper` and
`1 - elo/upper`, then broadcasts the resulting embedding to all 64 board-
square tokens and concatenates it with the board features (self and
opponent get separate anchors). Move selection is human-like: **sample**
from the policy logits (temperature + top-p, not argmax), then a **one-ply
opponent-response lookahead** — for each top candidate the model is queried
from the *opponent's* perspective with the elos swapped, and the candidate's
WDL is the inverted opponent value. So "play at level N" = sample like an N-
level human, keep what survives against an opponent at *their* level.

## The level ladder, decoded

The UI ladder is not 16 networks. It is one unified model (Maia-2/3) with a
skill-conditioning input, exposed through `Elo`/`SelfElo`/`OppoElo`. The
spacing (600, 800, then 100-steps to 2000, then 200-steps) is a product
choice for the dropdown; the model buckets/interpolates internally (10
discrete buckets in Maia-2; continuous anchors in Maia-3). The three
generations are three answers to the same question — *how do you make one
model play at a chosen human strength*:

1. Maia-1: **don't** — train one model per level (a model family).
2. Maia-2: one model + **discrete self/opponent bucket embeddings** (10 buckets).
3. Maia-3: one model + **continuous interpolated self/opponent anchor embeddings**.

All three condition on **both** elos (self *and* opponent), and all three
keep the clock: 30 s low-time filter in Maia-1/2 (plus first/last-move
trimming in Maia-2). None of them search at deployment time; strength
comes from the *distribution*, not from a depth knob.

## What it means for this engine

### Training

1. **Level windows are a both-players rule, not a mean.** Maia keeps a game
   for level L only if *both* players are in [L, L+100). Our training
   blocks (`data/training/`) are banded by the *mean* of the two ratings —
   a 2400-vs-1600 game lands in the same mean-band as 2000-vs-2000. The
   new tool `tools/build_level_conditioned_moves.py` applies the both-
   players rule over the committed blocks and emits the move-level labels.
2. **The move-level label schema is the retraining input.** Maia's CSV
   (per move: position, move, `active_elo`, `opponent_elo`, result, clock,
   `low_time`, plus CP-derived `cp_loss`/`winrate`/`is_blunder`) is the
   format a level-conditioned network trains on. From our blocks we can
   generate everything except the CP columns (they require a Stockfish pass
   or Lichess's `%eval`/`%clk` tags — the mirror stripped them, and our
   blocks are multi-source, not raw Lichess). The tool emits the full
   row with `clock: null` and documents the CP columns as the deferred
   half.
3. **Adopt the split discipline.** Maia-1/2 use a *time-ordered* holdout
   (latest months = test) and level-balanced sampling (20 blocks per range).
   Our blocks are source-disjoint but not time-ordered; a retrain should
   split within-band by date.
4. **The 30 s clock filter + first/last-10 trimming are the "clean human
   move" definition.** We can apply the move trimming now; the clock filter
   is deferred with the clock tags (needs raw Lichess files — the
   `available_extensions_not_committed` list in the block manifest).

### The network (the rating-conditioning fix)

`docs/rating-conditioning-finding.md` measured our shipped checkpoint's
rating input as inert: **0/200 moves change from 600 to 3200**, with a
suspiciously *linear* response — the signature of a single scalar
(`normalized_rating * rating_weight + rating_bias`) diluted through a 32-
wide projection into a 256-wide model. Maia's three generations are three
empirically-validated alternatives, in ascending order of ambition:

1. **Discrete dual-bucket conditioning (Maia-2, simplest real fix).**
   Replace the scalar with two integer inputs — `elo_self_bucket`,
   `elo_oppo_bucket` — each over Maia-2's 10 buckets (`<1100`, 100-wide
   1100–2000, `>=2000`), each through its own learned embedding that is
   concatenated into the model input (not projected through the 32-wide
   history vector). The labels for this exist now:
   `tools/build_level_conditioned_moves.py` emits `active_elo` and
   `opponent_elo` per move, so the buckets are a pure labeling detail.
2. **Continuous dual-anchor conditioning (Maia-3).** Two learned anchor
   embeddings per side (low/high), linearly interpolated by the clamped
   elo, broadcast across the input. Strictly more parameters; the
   interpolation keeps the signal at *every* rating value instead of
   bucket edges. Adopt only if (1) verifies and bucket-boundary artifacts
   matter.
3. **Opponent-response conditioning (Maia-3's lookahead).** Querying the
   prior from the opponent's perspective with swapped elos to rank root
   candidates is search-adjacent and would change the tree — it belongs
   behind the usual SPRT gate, not in this note's scope.

Whatever is retrained must be re-verified with
`tools/analyse_rating_conditioning.py` — the 0/200 sweep is the acceptance
test for "the conditioning path now matters", and the linear-response
diagnostic in that doc is the thing a broken retrain would reproduce.

What we cannot do from this sandbox: run the Maia models themselves
(weights are on Google Drive / HuggingFace — both hosts blocked; only
PyPI and GitHub are reachable), and generate the CP/CP-loss/blunder label
columns (need a Stockfish pass or raw Lichess files). Both are recorded as
next steps, not gaps in the mechanism.

## Honest limits

- This is a source reading of three public repos at the pinned commits; no
  Maia model was executed here. Where the README and code disagreed the
  code wins (all citations above are code-level).
- Labeling run on the full blocks (this commit): 71,961 games → **800,971
  rows** (profile in `benchmarks/unarchitectured-v1/`). The both-players
  window rule excludes 57,994 games (81%) whose two ratings span more than
  100 points — the strict cost of Maia's rule, which is exactly why it
  isolates "the average move at level L". The 600–900 windows are empty:
  the committed blocks contain almost no rated games below 1000 (a
  property of the source, documented in the block README). The labeling
  run also surfaced a data correction: 26 Carlsen-block games whose SAN
  text desyncs the parser (the round-14 manifest's `illegal_games_dropped:
  0` for that block was never measured) — re-cleaned and documented in
  `data/training/README.md`.
- Maia-1's level models were trained on 2017–2019 Lichess; Maia-2 on
  2018–2023; the platform ladder in the screenshot is a Maia-2/3-era
  product. The *mechanism* citations are generation-specific as labelled.
- Maia-1's repo is GPL; nothing was copied into this repo — only the
  mechanism is documented, and our data/tools are independent
  implementations of the described rules.

## Files

- `tools/build_level_conditioned_moves.py` — Maia-style move-level labeling
  over `data/training/` (both-players windows, bullet skip, first/last-
  move trim, `active_elo`/`opponent_elo` per move). Streams (the full
  output is ~2x10^6 rows / ~700 MB, never committed); per-game parsing on
  the same path the blocks were validated with.
- `tools/test_build_level_conditioned_moves.py` — fixture + real-block
  tests.
- `benchmarks/unarchitectured-v1/level-conditioned-moves-profile.json` —
  full-run profile: per-window row counts, skip counts, games per source,
  sha256 of the full (uncommitted) row stream, exact parameters.
- `benchmarks/unarchitectured-v1/level-conditioned-moves-sample.jsonl` —
  deterministic 200-rows-per-window sample of the full output (the
  retraining pipeline's inspectable slice; the full output regenerates
  bit-for-bit from the pinned blocks).
- `data/training/` (previous commit) — the blocks this tool labels.
