# Commercial and real-service validation

## What is actually benchmarked

The timing feature has now been exercised on actual exports from three online
services. Aggregate results are committed in
`data/service-timing-benchmarks/result.md`.

| Service | Perspectives | Accounts | Label quality | Result |
|---|---:|---:|---|---|
| Lichess BOT | 390 | 39 | affirmative `BOT` title | 20.5% at ACF1 >= 0.45 |
| Lichess unmarked | 2,204 | 2,141 | noisy comparison class | 30.4% at threshold |
| Chess.com unmarked | 3,637 | 1,733 | no affirmative BOT field | 14.4% at threshold |
| FICS unmarked | 3,254 | 1,225 | `IsComp` perspectives excluded | 13.7% at threshold |

The primary rating/time-matched Lichess result remains AUC 0.413 with an
account-bootstrap 95% interval of 0.260–0.575. Real Chess.com blitz traffic
crossed the timing threshold for 29.0% of account aggregates; FICS blitz crossed
it for 22.7%. This is why timing cannot be a standalone classifier.

The service runner understands:

- Lichess `%clk` and the committed pseudonymous JSONL;
- Chess.com `%clk`, including decimal seconds and zero-increment controls;
- FICS fractional-second `%emt` plus affirmative `WhiteIsComp` and
  `BlackIsComp` tags.

Only aggregate output is written. Extracted PGN player names and moves never enter the report; source repository URLs remain for provenance.

## Commercial platform coverage

| Platform | Public clock export and labels | Status |
|---|---|---|
| Chess.com | Public player game archives and PGN clocks; no affirmative computer-account field in archive games | benchmarked as an unmarked threshold stress test |
| Internet Chess Club | No documented public bulk clock archive with computer labels found | importer-ready only; requires a lawful user export |
| Playchess / ChessBase | No public bulk clock archive found | importer-ready only; requires a licensed user export |
| FIDE Online Arena | No public bulk move-clock archive found | importer-ready only; requires a lawful user export |
| chess24 | Playing service closed in January 2024 | no current service to test |
| Lichess | CC0 clocks and affirmative BOT titles | fully benchmarked as the open reference |
| FICS | Fractional elapsed-move times and computer tags | benchmarked as a public secondary stress test; only aggregates are committed |

“No public archive found” is not replaced with scraped, private, or fabricated
data. A PGN exported by the account owner can be measured locally without being
committed.

## Reproducing the service aggregate

With lawful local PGNs at the pinned revisions recorded in `report.json`:

```bash
python3 tools/service_timing_bench.py \
  --config config/timing_validation.json \
  --lichess-records data/timing-validation/records.jsonl \
  --lichess-matched-report data/timing-validation/report.json \
  --chesscom-pgn /licensed/path/chesscom/*.pgn \
  --fics-pgn /licensed/path/fics/*.pgn \
  --json service-report.json \
  --markdown service-report.md
```

## Commercial engine test suites

The repository cannot redistribute or silently acquire proprietary ChessBase,
Fritz, Convekta, Chess King, HIARCS, or other paid test positions. It now
provides two lawful paths:

1. **Use the vendor GUI directly.** Install `unchessed-reviewer` as the UCI
   engine, place `unchessed-nnue.bin` beside it, set `Threads=1`, `Hash=128`,
   `MultiPV=1`, disable books, and run the licensed product's test-set command.
   Fritz/ChessBase includes a “Process test set” workflow and records solved
   count and average solution time.
2. **Export a licensed suite to coordinate-move EPD** and use
   `tools/uci_epd_suite.py`. The runner accepts `bm`/`am` answers such as
   `e2e4`, clears game state between positions, pins UCI options, records binary
   and suite SHA-256 values, and emits per-position JSON.

Example:

```bash
python3 tools/uci_epd_suite.py \
  --engine target/release/unchessed-reviewer \
  --epd /licensed/path/vendor-suite-uci.epd \
  --movetime 10000 --threads 1 --hash 128 \
  --json vendor-result.json --markdown vendor-result.md
```

SAN-only answers are deliberately reported as unscored rather than guessed.
Convert them to UCI coordinates with the licensed database/GUI. The runner
refuses the adaptive player-facing binary by default so persona behavior cannot
contaminate a fixed-strength engine test.

## What remains externally required

No commercial-suite score is claimed because no licensed proprietary suite or
commercial Windows GUI is present in this Linux checkout. To obtain those
results, provide the legally acquired suite export or run the packaged reviewer
inside the licensed GUI. The harness and checksum/provenance output are ready;
the proprietary assets remain outside Git.

Position-suite solved percentages are regression diagnostics, not Elo. They do
not replace game matches, family-diverse opponents, confidence intervals, or
SPRT.
