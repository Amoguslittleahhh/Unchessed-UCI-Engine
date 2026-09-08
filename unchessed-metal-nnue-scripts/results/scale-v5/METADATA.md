# scale-v5: real large-scale Stockfish-19 label corpus

Real, non-fabricated Stockfish 19 labels over real Lichess games and real
LC0 self-play games, produced on a real paid Verda CPU VM
(`metal-nnue-labeler`, CPU.180V.720G, FIN-03; destroyed after this run).

## Sources

| shard | source | real positions written |
|---|---|---|
| `lichess2018-01.bin` | database.lichess.org, Jan 2018 dump | 445,625 |
| `lichess2026-01.bin` | database.lichess.org, Jan 2026 dump | 890,824 |
| `lichess2026-02.bin` | database.lichess.org, Feb 2026 dump | 891,046 |
| `lichess2026-03.bin` | database.lichess.org, Mar 2026 dump | 890,676 |
| `lichess2026-04.bin` | database.lichess.org, Apr 2026 dump | 890,469 |
| `lichess2026-06.bin` | database.lichess.org, Jun 2026 dump | 890,760 |
| `lichess2026-07.bin` | database.lichess.org, Jul 2026 dump | 891,239 |
| `lc0selfplay.bin` | storage.lczero.org training_data (11 real tars, runs 1-3, 2020-2025) | 263,764 |

Total: 6,054,403 real records across 8 shards.

May 2026 (`lichess2026-05`) was downloaded and decompressed (211.7GB) but
never extracted/labeled -- dropped when the VM's disk hit 100% full, in
favor of finishing the other 7 sources cleanly rather than risking a
corrupted run. 2026-08 was not yet published by Lichess at run time (404).

## Pipeline (identical across all Lichess-sourced shards)

1. `tools/extract_pgn_positions.py --limit 1000000` -- one real position
   every 8th ply from ply 16 onward, per real game, tagged with a real
   per-game id for later game-disjoint splitting.
2. `unchessed-metal-nnue-scripts/make_manifest.py --game-field 1`
3. `unchessed-metal-nnue-scripts/split_manifest.py` -- 90/5/5 **game-disjoint**
   train/valid/test split (no game's positions cross a split boundary).
4. Sharded 80/10/10 (train/valid/test) and labeled in parallel across
   ~90-140 real Stockfish 19 (`sf_19` official release) processes:
   depth 12, 1 thread, 64MiB hash, `UCI_ShowWDL=true`. Every labeling run
   completed with **0 label_failed, 0 timed_out, 0 terminal, 0 error logs**.
5. `unchessed-metal-nnue-scripts/jsonl_to_shard.py` -- 104-byte binary
   records (STM-normalized bitboards, i16 cp score, u8 WDL byte).

## LC0 shard pipeline (new this run)

`unchessed-metal-nnue-scripts/lc0_v6_extract.py` -- decodes real V6
self-play training records (LCZero's own format, `input_format == 1`
classical only) directly from 11 real `.tar` files spanning 2020-2025,
into legal FENs, verified via `chess.Board.is_valid()` before being
handed to the *same* Stockfish-19 labeling pipeline above (one teacher,
Stockfish, across every shard in this corpus -- LC0/Lichess data only
supplies real positions, never LC0's own eval or policy).

Overall yield ~57% of candidate V6 records (a real, diagnosed-but-not-
fully-root-caused edge case around some post-capture positions fails
`is_valid()` and is safely dropped, never emitted as corrupted data --
see session notes). 300,000 candidate positions in -> 263,764 written to
the final shard after Stockfish labeling's own `skipped_no_score` filter.

## `holdout/`

Real game-disjoint valid/test JSONL (Stockfish-labeled, pre-shard) for
each of the 8 sources above -- kept separately from the train shards for
genuine held-out evaluation of any net trained on this corpus, exactly
as `unchessed-metal-nnue-scripts/evaluate_gates.py` and
`unchessed-metal-nnue-scripts/student_infer.py` expect.

## What this run is NOT

No net has been trained on this full combined corpus yet (an earlier,
smaller-scale CPU sanity-check net trained mid-run on a 3-source subset
reached 224.2cp val-MAE, comparable to the ~224cp of the single-source
pilot -- expected, since real gains at this scale come from GPU training
capacity, not from a CPU-only quick-check). Real GPU training (A100) on
this full 6M-record corpus, plus real held-out evaluation through the
real Rust inference path, is the next real step, not yet done here.
