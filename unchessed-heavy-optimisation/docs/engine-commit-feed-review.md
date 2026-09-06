# Reviewing three engine commit feeds

Scan of recent `master` activity in Stockfish, Lc0 and RubiChess, checking each
theme against this engine. One gap was worth closing; the rest are recorded so
they are not re-examined from scratch.

## Closed: no mate-finding regression suite

Stockfish's recent `seekMate` work (2026-08-24, "Make extensions and futility
pruning depend on new seekMate variable") is validated with **matetrack** — a
fixed EPD suite of positions with known forced mates, run deterministically at
fixed nodes, **with no games at all**. Their PR text reports "61 FENs, 61 found
mates, 61 best mates."

That is a regression harness, not a strength test, and it catches a failure
mode SPRT is bad at: an engine that is fine *on average* but has lost the
ability to see a specific forced win. Two of their last twenty commits touch
mate finding ("Improve mate finding with a dynamic futility pruning depth
cutoff"), which is only safe to iterate on because the suite exists.

**We had nothing equivalent** — four hand-written mate assertions scattered
through `search.rs` unit tests, no EPD file, no runner, no coverage record.

It matters here more than it might elsewhere, because mate handling is this
engine's weakest *measured* area:

- `docs/unarchitectured-metal-theme-breakdown.md` put `mate_available` last of
  nine categories: top-1 **0.2105**, mean regret **408.7cp**, against
  0.7583 / 38.9cp for captures.
- The round-8 disagreement work found the real checkpoint ranking a forced
  back-rank mate **10th of 17**.

Those are policy findings rather than search findings, but they are exactly
why the search's own mate finding deserves a standing gate.

`benchmarks/matetrack.epd` now holds seven positions across five distinct
mating patterns (back-rank, smothered, queen-and-king, corner, two-rook
ladder), both colours represented, in standard EPD with `bm`/`dm`/`id`/`c0`
opcodes so any EPD-aware runner can consume it.
`tools/build_matetrack_suite.py` generates and verifies it;
`unchessed-core/src/search.rs` gains `matetrack_suite_finds_every_forced_mate`;
`tools/test_matetrack_suite.py` keeps the two in sync.

**The generator caught a bad fixture of mine during construction.** A
"two-rook ladder mate" I wrote was not mate at all — the king simply walks to
the seventh rank. It was excluded automatically, and I replaced it with a
verified position. Every `bm` is checked to be legal, mating, and the *unique*
mate, so the expected move is unambiguous.

## Reviewed, no action

**Stockfish — memory-safety hardening.** A visible cluster on 2026-08-19:
"Harden the tablebase reader against corrupted files", "Avoid UB decoding
LEB128, and guard against corrupted files", "Read hash_bytes tail bytes as
unsigned", "Refuse a failed large-page allocation". Our equivalent surface is
the two net loaders, and they are already guarded — `nnue.rs` rejects a bad
magic, an unsupported version, a wrong `ft_in`/`acc`, and any truncation
(`read_f32s` bounds-checks every read); `unarchitectured_metal.rs` has 18 distinct
error returns. Rust also removes the specific UB class most of those commits
address. Nothing to port.

**RubiChess — Valgrind/ASan pass (#522).** "index out of bounds fix 1", "index
out of bounds fix 2", uninitialised-memory fixes. This is the strongest
argument in the three feeds for a tool we cannot run: safe Rust prevents the
out-of-bounds and uninitialised reads they were fixing. Worth noting that
RubiChess is the closest of the three to this project in scale and team size,
and their answer to correctness was a sanitiser pass — the equivalent here is
`cargo test` under `-Zsanitizer` or just Miri, neither of which is reachable
without a toolchain.

**RubiChess — threat features (#517)** and **Stockfish — `Simplify
Position::update_piece_threats()`**. Both engines are investing in
threat-based NNUE inputs. Already reviewed and rejected in
`docs/nnue-architecture-audit.md`: a feature-set change is meaningful only
with a net trained on those features, so it needs new indexing, a new format
and a full retrain.

**Stockfish — LMR and futility tuning** ("Reduce LMR less aggressively in loose
alpha windows", "Simplify Away Second Razoring Number", "Simplify optimism
scaling formula"). These are exactly the parameters in our `SearchParams`, and
exactly the changes that need SPRT. Not portable as constants: their values are
tuned against their evaluation and their search, and transplanting a magic
number tuned elsewhere is the sort of change this project's SPRT discipline
exists to reject.

**Stockfish — NUMA and large pages** ("Initialize shared continuation history
once per NUMA node", "Place continuation history on large pages"). Note
RubiChess went the other way — "Disable NUMA support. Works like sh... at
TCEC." Two serious engines disagreeing is a good reason not to speculate; the
target here is a single-socket laptop CPU where NUMA is moot, and Linux
transparent huge pages already apply without code.

**Lc0.** Its recent work is overwhelmingly backend and build infrastructure —
CUDA arch detection, CUTLASS, ONNX/CoreML/MIGraphX backends, TensorRT typing.
All GPU-inference concerns for an MCTS engine. Already covered by
`docs/policy-prior-calibration.md`: PUCT needs a network evaluation per
expanded node, and at 9.72ms per forward pass a few hundred nodes would consume
an entire move budget.

## Status

The only behaviour change is a new test plus a new benchmark asset — no engine
code path is modified.

**Not compiled** — no cargo or rustc in this sandbox. The new Rust test was
checked by symbol review (it reuses the existing `best_move` helper,
`is_mate_score` and `mate_in`, all already used by the neighbouring test),
bracket balance across all 21 tracked `.rs` files, and by verifying every
position independently in Python. It has **not** been run by `cargo test`.

Worth flagging for whoever does compile it: if any suite entry fails, that is
a genuine finding about the search rather than a broken fixture — each mate is
verified unique by `python-chess`, so the expected move is not in doubt.
