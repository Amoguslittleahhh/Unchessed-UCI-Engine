# Real-service timing benchmarks

> Aggregate results only; no game-level service account identifiers, game IDs, moves, or raw commercial-service PGNs are committed.

**Decision: timing-only classification rejected across real-service benchmarks.**

## Cross-service account-level stress test

| Service / label | Perspectives | Accounts | Median ACF1 | Accounts at ACF1 >= 0.45 |
|---|---:|---:|---:|---:|
| lichess / bot | 390 | 39 | 0.212 | 20.5% |
| lichess / unmarked | 2204 | 2141 | 0.310 | 30.4% |
| chess.com / unmarked | 3637 | 1733 | 0.169 | 14.4% |
| fics / bot (insufficient; descriptive only) | 1 | 1 | 1.000 | 100.0% |
| fics / unmarked | 3254 | 1225 | 0.159 | 13.7% |

## Primary matched Lichess classification result

| Metric | Result |
|---|---:|
| Account AUC (higher = BOT) | 0.413 |
| Account-bootstrap 95% CI | [0.260, 0.575] |
| Sensitivity at 0.45 | 20.0% |
| Unmarked threshold share at 0.45 | 33.3% |

## Commercial platform availability

| Platform | Status |
|---|---|
| chess.com | benchmarked from public archive PGNs; no affirmative BOT field, so threshold stress test only |
| internet chess club | not benchmarked: no public bulk clock archive and affirmative computer labels available locally |
| playchess chessbase | not benchmarked: no public bulk clock archive available locally |
| fide online arena | not benchmarked: no public bulk move-clock archive available locally |
| chess24 | not benchmarked: playing service closed in 2024 |

## Limits

- Only Lichess supplies enough affirmative BOT labels for an AUC benchmark in this snapshot.
- Chess.com and FICS unmarked traffic can include automation; threshold shares are stress tests, not verified-human false-positive rates.
- Chess.com and FICS archives are account-centered samples, not natural-traffic samples.
- No result is fabricated for a commercial service without a lawful public export or user-supplied licensed data.
