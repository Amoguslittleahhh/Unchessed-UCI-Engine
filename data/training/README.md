# Training data blocks

Game data spanning all player levels for the engine's training pipelines
(NNUE, Unarchitectured v1). Selected, banded, and **fully
move-legality-validated** on 2026-08-26 from the pinned PGN mirror
`rozim/ChessData` @ `ed88abd` (fetched by partial git clone — the sandbox
can reach GitHub but not lichess.org/archive.org/twic.sesse.net; see the
egress audit in `docs/dev-environment.md`).

**What "reliable" means here, precisely** (see `manifest.json` for the
per-block record):

- **Hash-pinned.** Every file has its SHA-256 in the manifest;
  `python3 tools/training_blocks.py verify` re-checks them.
- **Fully move-legal.** Every game in every block was parsed and replayed
  with python-chess; illegal games dropped (160 total, all from two
  lichess bands). Committed blocks are 100% legal — this is what the
  upstream mirror's "dirty data" warning is a warning *against*.
- **Rating-banded where rated.** Lichess blocks are banded by the mean of
  `WhiteElo`/`BlackElo` (both required, both in [100, 3500], standard
  results only). Games with unknown ratings (tag `0` or missing — ~25% of
  the source chunks, mostly open-tournament games) are quarantined, never
  banded.
- **Provenance per file.** Source repo commit + in-repo path per block;
  the upstream caveat ("dups, dirty data, errors") is recorded.
- **Honest gaps.** World Championship files carry no rating tags (kept as
  a separate top-tier block); 219 Carlsen–Nakamura head-to-heads appear in
  both player archives (0.30%, documented, not deduped); ratings are as
  tagged by the source (lichess = platform-calibrated, self-registered
  accounts).
- **Correction (2026-08-26).** Re-running `clean` on the round-14 Carlsen
  block found 26 games (0.6% of that block) whose SAN text makes
  python-chess 1.11.2 drop tokens — the manifest's
  `illegal_games_dropped: 0` for that one block was never actually
  measured. The block was re-cleaned in place (4,314 → 4,288 games) and
  the manifest/counts above updated; all 14 other blocks re-verified with
  zero dropped-token games.

## Coverage (71,961 games)

| Level | Games | Source |
|---|---:|---|
| ≤1400 | 695 | lichess (merged 3 chunks) |
| 1400–1700 | 4,887 | lichess |
| 1700–2000 | 4,465 | lichess |
| 2000–2300 | 16,313 | lichess |
| 2300–2600 | 22,099 | lichess |
| ≥2600 | 1,926 | lichess |
| GM player archives (2687–2735 mean) | 9,867 | Carlsen, Nakamura (PgnMentor) |
| Master leagues (2209–2647 mean) | 1,421 | Bundesliga 2006-07, British Champ 2017 |
| TWIC weekly issues (1528–2786) | 9,666 | issues 400 / 1000 / 1649 |
| World Championship (unrated tags, top-tier) | 622 | 1990, 1993 |

The low bands are the thin part — that's a property of the source (the
mega-clean set is mostly strong games), not of the selection: the mirror
holds ~10M games and the full TWIC series, and `manifest.json` lists the
exact paths to extend from (Kingbase 2017 2200+ chunks, all 1440 TWIC
issues, the remaining 62 mega chunks, 1000 per-player GM archives).

## Tooling

```sh
python3 tools/training_blocks.py verify              # re-check hashes vs manifest
python3 tools/training_blocks.py validate F --sample 200   # fast header stats + sampled parse
python3 tools/training_blocks.py clean F             # full parse, drop illegal games
python3 tools/training_blocks.py split SRC --prefix OUT  # band a new source PGN (text-level, lossless)
python3 tools/training_blocks.py fetch --out DIR     # partial-clone the pinned source (needs GitHub egress)
```
