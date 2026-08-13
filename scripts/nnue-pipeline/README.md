# NNUE data pipeline scripts

- `worker_bench.sh` — benchmarks `unchessed-datagen nnue` worker-count
  throughput (used to pick 28 workers / 2 per core over both 14 and 56 on a
  14-core WSL box; see project memory for the numbers).
- `full_pipeline.sh` — end-to-end unattended pipeline: waits for Lichess
  monthly PGN downloads to finish, verifies each `.zst`'s integrity (a curl
  process exiting is not proof of a complete download -- a dropped
  connection looks identical), decompresses one month at a time, labels it
  with `unchessed-datagen nnue`, deletes the decompressed intermediate to
  control disk usage, then retrains the NNUE on the combined dataset --
  with a hard record-count safety ceiling before training starts, so an
  unexpectedly large combined dataset can't repeat this project's earlier
  near-OOM incident with nobody around to catch it.

Both assume `unchessed-datagen`/`unchessed-adapter` are already built at
`~/unchessed-kingsafety-src/target/release/` and the NNUE training venv is
at `~/unchessed-ai/data/maia-venv/` -- adjust paths for a different
environment.
