# Unchessed AI results — WSL data layout

All data below is from **unchessed-adapter**, tested with `Adaptive=false` in
SPRT/exhibition contexts (which disables persona blending, making it play at
its full raw eval/search strength — not the same as `unchessed-reviewer`).
**unchessed-reviewer** (the dedicated full-strength analysis binary) has not
been separately benchmarked in any of this data; its results folder is
present but currently empty.

## Layout

```
results/
  adapter/
    sprt_gates/            <- every SPRT gate's .log (full game-by-game
                               transcript + final Elo/LLR verdict) and .pgn
                               (every game played), one pair per technique
                               tested this project. Filename indicates what
                               was being isolated, e.g.
                               sprt_tier2.log/.pgn = RookOnKingRing +
                               passed-pawn king-distance + pin-aware SEE
                               vs the pre-tier2 baseline.
    exhibitions/            <- tenmin_*_series/ = 3-game, 10-min-per-side
                               matches vs RubiChess (full-strength unless
                               named "matched_level", which used a
                               calibrated RubiChess LimitNps to play at
                               ~50% vs a fixed-strength adapter). Each
                               game has its own subfolder with the full
                               UCI transcript (adapter_log.txt /
                               rubichess_log.txt), move list, and PGN.
    calibration/            <- ladders run to find a RubiChess LimitNps
                               value matching a target adapter strength.
    archive_early_exploration/
                            <- earlier-session exploratory analysis
                               (elo ladders/scaling/SPSA dry runs) largely
                               superseded by the SPRT-gate discipline used
                               for everything since — kept for reference,
                               not actively maintained.
  reviewer/                 <- empty; no dedicated reviewer benchmarks yet.

data/maia-data/
  l20*.pgn(.zst)            <- raw Lichess human-game corpus, source data
                               for move-prior/opponent-modeling training
                               AND the planned future NNUE training run.
  nnue/                     <- ~11GB, ~100M labeled positions from the
                               2026-07-21 labeling run — the NNUE training
                               data, ready but training itself deliberately
                               deferred (finish classical eval work first,
                               per project decision).
  sprt_book.pgn             <- the opening book used by every SPRT gate.
  unchessed-maia*.bin       <- trained move-prior policy net(s) used by the
                               adapter's persona/opponent-modeling system.
```

## Binaries

- `~/unchessed-kingsafety-src/` — the active WSL build/test directory,
  kept in sync with the Windows repo's `unchessed-core`/`unchessed-adapter`/
  `unchessed-reviewer` crates. Name is historical (dates to an old king-safety
  attempt) but this is the current, live build dir — don't be misled by the
  name.
- Windows repo root: `unchessed-adapter.exe` / `unchessed-reviewer.exe` —
  the deployed binaries, always kept in sync after each SPRT-gate resolution.
  `unchessed-adapter.exe` is the persona/adaptive engine; `unchessed-reviewer.exe`
  is the full-strength, no-persona analysis engine. Both share the same
  underlying `unchessed-core` eval/search — they differ only in which UCI
  options and default behavior (Adaptive mode, opponent modeling, etc.) each
  binary's `main.rs` exposes.

## For GLM research

If feeding this to an external tool for analysis: the `sprt_gates/` logs are
the highest-signal data (each documents a controlled A/B test with a real
statistical verdict). The `exhibitions/` PGNs are real games but are N=1-3
per configuration — anecdotal, not statistically powered; treat them as
qualitative color, not evidence, the same way this project's own memory
does.
