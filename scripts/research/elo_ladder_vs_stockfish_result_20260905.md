# Absolute Elo estimate vs. real Stockfish (2026-09-05)

First absolute (externally-anchored) Elo estimate for the adapter in
this project's history -- everything else in `scripts/sprt-history/`
and the `wsl_sprt_*` scripts measures relative strength (engine A vs
engine B on this repo). This anchors against Stockfish 18's own
UCI_Elo calibration instead.

## Setup

- Engine: `target/release/unchessed-adapter.exe` at main HEAD (includes
  the `known_full` port, commit `0df9a86`), `Adaptive=false` (pure
  engine strength, no persona behavior), `OwnBook=false`, `Threads=1`,
  `Hash=64`.
- Opponent: Stockfish 18 (the binary bundled with the En Croissant
  chess GUI, `org.encroissant.app/engines/stockfish/`), `UCI_LimitStrength=true`,
  `UCI_Elo` set per level.
- `movetime=1000ms` for both engines every move, no cutechess-cli
  (driven directly over UCI via `elo_ladder_vs_stockfish.py`, on this
  reviewer's Windows machine, native binaries -- not WSL).
- 8 games per level, alternating colors, 8 different fixed openings.
- An initial wider ladder (1500/1800/2000/2200/2500/2800) was cut short
  after 1500 swept 6/6 and 2200 stayed pace -- not worth the wall-clock
  time once the trend was obvious. Re-run focused on 2200/2500/2800.

## Result

| Stockfish UCI_Elo | Score | Record |
|---|---:|---|
| 2200 | 8.0/8 (100%) | perfect sweep |
| 2500 | 8.0/8 (100%) | perfect sweep |
| 2800 | 5.5/8 (68.8%) | 3W 3D 2L -- first competitive level |

**Implied absolute Elo: ~2940** at this time control, from the 2800
crossover point (`elo_from_score(0.688, 2800) ≈ 2937`).

The "implied ~3400" / "implied ~3700" figures printed for the 2200 and
2500 levels are an artifact of the logistic Elo formula blowing up
toward infinity as score approaches 1.0 on a perfect sweep -- they only
mean "clearly well above that level," not literal point estimates. Only
the 2800 result is a real crossover point.

## Caveats

- **n=8 at the one informative level (2800) is a small sample.** The
  real 95% confidence interval on that single data point is wide,
  easily +/-150-200 Elo. This is a rough estimate, not a precise rating.
- **Fast time control.** `movetime=1000ms` is closer to bullet/hyperbullet
  than a reference control (CCRL/CEGT lists typically use much slower
  controls). This number is specific to this time control and should
  not be read as "the" engine rating.
- **Adaptive=false only.** This measures raw engine strength with the
  persona/adaptive system off, not what a real opponent would face by
  default (`Adaptive=true`), which deliberately varies playing strength.
- Stockfish's `UCI_Elo` calibration is itself an approximation
  (Stockfish's own docs note it isn't perfectly linear or precisely
  matched to any single external rating pool), so "anchored" here means
  "anchored to Stockfish's own estimate of its restricted strength,"
  not to a human-rated pool directly.

A more precise number would need many more games at 2600-3000 (finer
Elo steps, larger n) and/or a slower, more standard time control.
