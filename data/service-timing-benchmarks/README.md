# Real-service timing benchmark aggregates

This directory contains aggregate benchmark results only. It intentionally does
not contain game-level Chess.com or FICS account identifiers, game identifiers,
moves, or raw PGNs.
The JSON report records source repository commits, byte counts, and content
checksums so a lawful local copy can reproduce the result with
`tools/service_timing_bench.py`.

Only Lichess supplies enough affirmative BOT metadata in the current snapshot
to support a classifier AUC. Chess.com and FICS results are unmarked-traffic
threshold stress tests, not verified-human false-positive estimates.
