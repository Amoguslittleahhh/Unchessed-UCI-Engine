# Unchessed AI

A UCI chess engine family built from scratch in Rust:

- **Unchessed Game Adapter** (`unchessed-adapter.exe`) — an adaptive engine that
  estimates its opponent's strength *live from their moves* (fully offline, no
  rating lookups) and shapes its play to match, punish, or clinch.
- **Unchessed Game Reviewer** (`unchessed-reviewer.exe`) — a full-strength
  analysis engine: same `unchessed-core` eval/search as the adapter, but no
  persona/opponent-modeling layer — always plays at its raw strength.

Both binaries share the same `unchessed-core` crate (eval, search, movegen)
and are always rebuilt/redeployed together after any change; they differ only
in the UCI options and default behavior their own `main.rs` exposes.

## Status: barebones milestone (this build)

| Component | State |
|---|---|
| Bitboard movegen | ✅ perft-verified (startpos d6 = 119,060,324; Kiwipete d5 = 193,690,690, exact) |
| Search | ✅ iterative deepening alpha-beta, quiescence, TT, null-move, LMR, killers/history, MultiPV, time management |
| Eval | ✅ NNUE (768-input/256-accumulator/SCReLU), SPRT-validated +107.1 Elo over the hand-crafted eval, auto-loaded from `unchessed-nnue.bin`; HCE (material + PSTs + bishop pair + mobility/passed-pawns/rook/knight-outpost terms) remains as a fallback when the file is absent |
| UCI protocol | ✅ full loop, worker-thread search, `stop` safe, 9/9 smoke tests |
| Time management | ✅ clock-aware: deep searches on a full clock, urgency tiers as time drains (near-instant on the increment in panic mode), situation-scaled budgets (sharp/wide positions get more), easy-move early stop, score-drop extensions; verified flag-free in 10s+0.1s blitz |
| Opening book | ✅ ~45 embedded main lines with ECO names + troll tier + external Polyglot `.bin` support (key computation verified against the format spec test vectors) |
| Adapter brain | ✅ live opponent-Elo model, MATCH/PUNISH/CLINCH/DEFEND personas, human-plausible move selection, `UCI_Opponent` engine identification |
| Human policy net | ✅ Maia-style per-rating policy nets (v2: castling-rights + en-passant aware inputs) trained on 19.9M positions from 781k human Lichess games (CC0 data), pure-Rust inference, auto-loaded from `unchessed-maia.bin` |

**Honest goals note:** beating full-strength Stockfish is not a realistic outcome
for any hand-built engine — Stockfish is 15+ years of distributed testing. The
achievable target, which this architecture is built for: play convincingly at
*any* level from ~600 Elo to master+ strength and adapt between them
automatically. NNUE + search work later raises the ceiling.

## Building

```
cargo build --release
```

Produces `target\release\unchessed-adapter.exe` (standalone, no dependencies).

Tests (perft suite, search, book, model): `cargo test`. Deep perft:
`cargo test --release -- --ignored`.

## Repository layout

- `unchessed-core/` — shared eval, search, movegen, UCI protocol logic used
  by both binaries.
- `unchessed-adapter/`, `unchessed-reviewer/` — the two binary crates
  (thin `main.rs` wrappers around `unchessed-core`).
- `unchessed-datagen/` — training-data generation tooling.
- `tools/` — auxiliary scripts (NNUE trainer, etc.).
- `scripts/exhibition/` — the game-runner scripts used for exhibition
  matches against reference engines (RubiChess).
- `scripts/sprt-history/` — smoke-test/SPRT-launch scripts committed as a
  record of specific past eval/search experiments (mix of passed and
  reverted techniques — check project memory or `scripts/sprt-history/`
  filenames for which).

SPRT gate logs/PGNs and exhibition-series results themselves are not
committed to git (they're large and regenerated per-run) — see the WSL
build environment's `~/unchessed-ai/results/README.md` for that data.

## En Croissant setup

The engine has been registered in En Croissant's engine list automatically
(`%APPDATA%\org.encroissant.app\engines\engines.json`). If you need to do it
manually: **Engines → Add new → Local** and point it at
`target\release\unchessed-adapter.exe`.

- **Play vs it:** Board → New game → pick *Unchessed Game Adapter* as one side.
- **Watch it think:** open the engine log panel — every adapter decision is
  narrated via `info string [Unchessed] ...` lines (opponent estimate, persona
  switches, book/troll choices).

> If Windows **Smart App Control** blocks a freshly rebuilt exe (error 4551),
> that's an OS policy on unsigned new binaries — the currently built release
> exe passed it. If it ever triggers after a rebuild, re-run the build once
> (reputation re-check) or run the test suite via WSL.

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 128 | transposition table MB |
| `MultiPV` | 1 | analysis lines shown |
| `Adaptive` | true | the whole adapter brain; off = always best move |
| `UCI_LimitStrength` / `UCI_Elo` | false / 2400 | hard cap on playing strength |
| `Contempt` | 25 | drive to win drawish games (fuels CLINCH mode) |
| `Troll` | Auto | `Off` / `Auto` (model-gated) / `On` (forced clowning) |
| `OwnBook` | true | use the opening book in games |
| `BookFile` | — | path to any Polyglot `.bin` book (e.g. built from Lichess masters) |
| `BookDepth` | 16 | max plies to stay in book |
| `PolicyFile` | auto | path to a policy weights file; default: `unchessed-maia.bin` next to the exe |
| `UCI_Opponent` | — | standard GUI-supplied opponent info; seeds the model for engines |

## How the adapter thinks

1. **Pre-game:** if the GUI sends `UCI_Opponent`, known engines (Stockfish,
   Leela, Komodo, …) seed the model at their real strength — trolling is
   hard-locked off against strong engines. Humans always start neutral:
   declared ratings are never trusted as truth.
2. **Live model:** every opponent move is compared against the engine's own
   analysis; centipawn loss (weighted by position difficulty, book moves
   discounted) feeds a Bayesian running Elo estimate that converges in ~8–12
   moves and keeps tracking.
3. **Personas** (selection only — the search underneath always runs full
   strength): **MATCH** blends to the opponent's level with human-plausible
   moves; **PUNISH** snaps to forcing best moves the moment they blunder
   (plays found mates immediately, prefers simplifying captures when far
   ahead); **CLINCH** picks venomous, trap-laden lines in drawish late games
   (narrow-safe-path metric, keeps queens on, and wires *contempt into the
   search itself* so drawn lines score negative while chasing a win — but
   neutral again when DEFENDing, because then a draw is a rescue); **DEFEND**
   digs in when worse. Transitions have hysteresis (enter/exit thresholds)
   so the engine commits to a plan instead of flapping, and every persona
   change is logged: `persona MATCH -> PUNISH (eval 990 cp, opponent ~1011)`.
4. **Engine-tell detection**: near-instant, near-perfect replies in positions
   with real choice raise a suspicion score (fed by the opponent's clock usage
   between moves). A suspected engine gets full-strength chess and zero
   trolling, whatever the rating estimate says. Erratic play (brilliancies
   mixed with blunders) widens the model's uncertainty instead of narrowing
   it — the sandbagger pattern.
5. **UCI_Elo semantics**: with `UCI_LimitStrength` on, the engine plays *at*
   `UCI_Elo` in every mode, matching standard UCI behavior.
6. **Book:** popularity-weighted theory with ECO names; a separately-tagged
   troll tier (Bongcloud, Scholar's mate attempts, Stafford, Fried Liver, …)
   gated by the live Elo model — big game detected → mainlines only, and a
   bail-out guard eval-checks the position before continuing any troll line
   (`troll line refuted — back to real chess`).

## The human policy net

MATCH-mode move selection is driven by Maia-style policy networks: one net per
rating bucket (<1300, 1300–1599, 1600–1899, 1900+), each trained to predict
*what a human of that rating actually plays*, on non-bullet rated games from
the [Lichess open database](https://database.lichess.org) (CC0). At play time
the engine blends the two buckets nearest its current target rating, so "play
like a 1450" uses genuinely 1450-flavoured move preferences, not random noise
on top of engine moves. If the weights file is missing, the engine falls back
to the built-in heuristic priors and says so in the log.

The v2 nets see the full special-rule state (castling rights + en-passant
square as explicit inputs; en-passant and promotion samples oversampled in
training). Validation top-1 accuracy predicting real human moves, per bucket:

| Bucket | overall | castling | en passant | promotion |
|---|---|---|---|---|
| <1300 | 29.2% | 53.4% | 18.8% | 76.7% |
| 1300–1599 | 31.0% | 60.7% | 44.7% | 79.3% |
| 1600–1899 | 32.8% | 74.1% | 61.4% | 79.5% |
| 1900+ | 33.5% | 81.5% | 67.4% | 75.8% |

(That en-passant accuracy *rising with rating* is real human behavior — weaker
players genuinely miss or decline en passant, and the per-bucket nets
reproduce that.)

Pipeline (reproducible):

```
cargo run --release -p unchessed-datagen -- samples 4000000 0.5 games1.pgn ...
python tools/train_policy.py samples unchessed-maia.bin 3 256 4000000
# put unchessed-maia.bin next to unchessed-adapter.exe
```

Debug helper: type `policy 1200` (or any rating) into the engine's stdin to
see the net's top human moves for the current position.

## The NNUE evaluator

`unchessed-nnue.bin` (auto-loaded next to the exe if present, `EvalFile` UCI
option to point elsewhere, falls back to HCE if absent/unreadable) is trained
on 108M HCE-labeled self-play positions via `tools/train_nnue.py` — a small
768-input/256-accumulator/SCReLU net (~197K params), 15 epochs, WDL-space loss
with exponent 2.5 matching Stockfish's nnue-pytorch recipe. SPRT-validated
+107.1 ± 27.0 Elo over the hand-crafted eval (532 games, LOS 100%) — the
biggest single gain in this project's history, more than double the previous
best (mobility, +52.3 Elo).

```
python tools/train_nnue.py selfcheck              # format/gradient sanity check
python tools/train_nnue.py unchessed-nnue.bin 15 shard0.bin shard1.bin ...
```

## Roadmap (next rounds)

- **Reviewer**: full-strength UCI engine + PGN review CLI (move classification,
  accuracy %).
- **NNUE**: quantization-aware training (int8/int16, ~5000x faster inference),
  incremental accumulator updates, training on Lc0-distilled labels instead of
  the current HCE-labeled data (shadow-streaming Lc0's training-data repository
  directly, deliberately deferred pending this first result).
- Deeper policy nets (conv/resnet) once GPU training is available; more data,
  more buckets.
- Lazy SMP threads, pondering, tablebases, adaptation tuning.

## Development verification tools

- `python tools/uci_smoke.py <engine>` — 9-step UCI protocol conformance test.
- `python tools/selfplay.py <engine> [games] [movetime]` — self-play sanity run.
