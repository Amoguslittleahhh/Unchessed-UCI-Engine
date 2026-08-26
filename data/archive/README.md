# Chess history archive (1834 → 2022)

The finest breadth of human play collectable from sources reachable from
this sandbox (GitHub egress only; the classic chess-data hosts and
HuggingFace are not reachable). Complements `data/training/` (which is
banded by rating for engine training): this archive is curated for
**era, theme and quality**, not rating.

**123,385 games, 294 files, 84.8 MB**, spanning 1834 (De la Bourdonnais–
McDonnell, Paris) to 2022 (Reykjavik open), plus two pre-1800 legend
entries. Every game was move-legality-validated by full per-game parse
(52 dirty games from the mirrors dropped and counted); games are
byte-verbatim from the pinned sources.

## Contents

| Directory | Files | Games | What it is |
|---|---:|---:|---|
| `classics-1834-1899/` | 18 | 3,485 | De la Bourdonnais–McDonnell 1834, Saint-Amant 1843, London 1851 (two sources), Anderssen/Bird/Blackburne/Horwitz/Morphy/Staunton/Zukertort archives, 1800–1900 era archive, GM-annotated classics (immortals, Hayes, Schiller, great masters), early Italian championships |
| `1900-1945/` | 15 | 20,715 | Lasker/Capablanca/Alekhine/Réti/Nimzowitsch/Tartakower archives, era archives 1901–1940, New York 1924, British championships 1946–55, Hastings 1920s–49 |
| `1946-1970/` | 37 | 47,734 | era archives 1941–1970, WCC matches 1957–61, 1964 Moscow candidates, 1973/1976 interzonal cycles, British championships 1965–70, Hastings 1950s–70s |
| `1971-1999/` | 74 | 28,590 | the complete Karpov–Kasparov era: every candidates cycle and interzonal 1979–1993 on file, WCC 1984/85/86, PCA years, 1998 WCC playoffs, "Kasparov vs. The World" 1999, British championships 1971–90, Hastings 1980s–90s, national championships (Ireland, Denmark, Canada, Croatia, Sweden), and a **9,078-game all-2600+ rated slice (1970–1998)** — the strength anchor |
| `2000-plus/` | 103 | 13,499 | 2002 Sparkassen candidates, 2000/04/06 WCC, every PgnDownloads top tournament 2007–11 (Amber blindfold & rapid, FIDE Grand Prix, World Cup, World Blitz, Tata Steel, US championships, …), Tata Steel 2003–10, British championships 2004–17, Hastings 2000s/2015, Reykjavik open 2022 |
| `womens/` | 35 | 8,976 | Polgar (all three sisters), Kosteniuk archives; women's WCC matches 1987–1996 (FIDE + PCA), women's interzonals; FIDE Women's Grand Prix & Women's World 2008–11; women's invitational 2009–17 |
| `world-championships/` | 2 | 335 | London 1851 — the first world championship (both available versions) |
| `correspondence/` | 10 | 51 | ICCF correspondence chess 2017–21 (see honest limits — the mirror is mostly unparseable) |

The WCHAMP world-championship file (1886–1994) and the Lasker/Blackburne/
Bird/Tartakower/Italy/Sweden archives span several eras, so they are
**partitioned per game by Date year** into the era directories above
(e.g. `world-championship-matches-1900-1945.pgn`) — games are copied
verbatim, just re-bucketed.

## Sources (pinned commits)

| Key | Repo | Commit | Used for |
|---|---|---|---|
| chessdata | `rozim/ChessData` | `ed88abd2716da58ee55d42b662455c1c8ebe0776` | all of the above except the annotated classics |
| scoutfish | `mcostalba/scoutfish` | `00cec1339f97114a32c30080dbad5e3a500634f2` | GM-annotated classic collections, New York 1924, Moscow 1964, Kramnik, Polgar |
| annotated | `hegde10122/CHESS_ANNOTATED_GAMES` | `550bdd161514b48abd5008ee4a7daa0db7718f66` | ICCF correspondence |
| pepper | `saikrishna-1996/deep_pepper_chess` | `b05bfe2e6defad7a85d6099ad5d69e1b46888eb5` | women's invitational 2009–17 |

The file-by-file layout (source path → destination, with notes) is
`tools/archive_layout.json`.

## Tooling

```sh
python3 tools/archive_blocks.py fetch --stage /tmp/arc/src   # git egress needed
python3 tools/archive_blocks.py build --stage /tmp/arc/src   # verbatim copy + era split + legality clean + manifest
python3 tools/archive_blocks.py verify                       # re-check sha256 + game counts
```

The clean is the same rule as `tools/training_blocks.py clean`: a game that
makes python-chess 1.11 drop even one SAN token is dropped whole (its board
state is desynced from then on); parser warnings are captured and counted,
never printed.

`manifest.json` records per file: source repo/path/commit, sha256, game
count (before/after clean), drop count, rated-game count, year span.

## Honest limits

- **52 games dropped** by the legality clean (0.04%): old/annotated files
  (Lasker 1834–99 part, immortals, New York 1924, 19th-century misc) and
  the ICCF mirror.
- **Correspondence chess is thin by accident, not choice.** The only
  reachable ICCF source is an amateur annotated-games mirror whose PGNs are
  mostly unparseable by python-chess (31 of 82 games dropped; 51 kept).
- **Two legend entries** in `classics-1834-1899/misc-19th-century.pgn`:
  "London 1475" (Castellvi vs Vinyoles — the date is a source-side data
  error; the pair are associated with 15th-century Madrid) and "London
  1790" (the Philidor vs "Andrew Smith" legend). Kept as historical
  material, flagged here.
- **681 games appear in more than one file** (London 1851 in three files,
  Bourdonnais in two, WCHAMP games also inside era archives, …). No dedupe
  was applied — pipelines that need it should dedupe by a
  players+date+moves key, as `data/training/` documents.
- **40,950 of 123,385 games carry Elo rating tags.** Pre-1990 over-the-
  board play is mostly unrated in these sources; that is why the
  rating-banded training set and this archive are separate.
- Upstream caveat from the primary mirror applies: "dups, dirty data,
  errors". The legality clean addresses the move-text side; header noise
  (odd event names, missing dates) remains.
- The 2000s coverage is top-events-only (2007–11 tournament set + Tata
  Steel + British); 2012–2021 is not represented except by the Polgar
  archive (to 2021) and Reykjavik 2022 — the modern mass is already in
  `data/training/` (lichess 2022 bands).
