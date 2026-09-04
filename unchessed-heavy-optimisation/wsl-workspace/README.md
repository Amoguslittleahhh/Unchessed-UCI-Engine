# WSL workspace — kept separate from the Windows source tree

Everything under this directory came from the WSL build/test environment
(`~/unchessed-ai/` there), **not** from the Windows repo. It is deliberately
kept in its own top-level folder rather than merged into the main tree, so
it's always clear which side of the Windows/WSL split a file came from.

## What's here

- `results/` — SPRT gate logs/PGNs and exhibition-series game data. See
  `results/README.md` for the full layout and how to weigh SPRT data (real
  statistical evidence) vs exhibition PGNs (anecdotal, N=1-3 per matchup).
- `scripts/` — the exhibition game-runner scripts as they exist on the WSL
  side (largely mirrors `scripts/exhibition/` in the Windows repo root).
- `logs/` — misc run logs from the WSL side.

## What's deliberately NOT here

- The WSL build directory (`~/unchessed-kingsafety-src`) is a synced mirror
  of this repo's `unchessed-core`/`unchessed-adapter`/`unchessed-reviewer`
  source — not duplicated here since it'd just be a stale copy of what's
  already at the repo root. Any script unique to that directory (not already
  in `scripts/`) has been copied into `scripts/sprt-history/` instead.
- The NNUE/Lichess training data corpora (~140GB), Python venvs (~1GB), and
  third-party engine binaries (Stockfish, RubiChess, cutechess-cli) used on
  the WSL side are not tracked here — too large for git and not source
  material; they're downloaded/built locally as needed.
