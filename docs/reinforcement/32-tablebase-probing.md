# Endgame tablebase probing

## Scope and conclusion

This investigation covers **integration of local Syzygy endgame tablebase probing** into Unchessed, not acquisition, mirroring, or hosting of tablebase files. It is a Tier 1 design investigation only: no engine code, dependencies, UCI defaults, tablebase files, or Tier 2/3 work were added.

**Recommendation: keep integration as a future, opt-in Tier 2 project, but do not implement it or acquire/host files now.** Syzygy is the practical standard for exact endgame knowledge: WDL50 (`.rtbw`) is appropriate inside search and DTZ50'' (`.rtbz`) is primarily a root move-selection/finish metric. The expected value is credible in low-material positions, but integration is not a one-function evaluation hook. It requires an exact position adapter, correct legal move/undo semantics, halfmove-clock handling, bounded probing, thread-safe table access, and a deliberate UCI/API policy. File acquisition and hosting are a separate Tier 3 storage/distribution decision and must not be smuggled into the integration project.

## Repository verification

The requested source inspection was run against branch `manus/research-facilities`. The focused grep command was:

```text
grep -RniE 'tablebase|syzygy|\.rtbw|\.rtbz|tbprobe' unchessed-core/src Cargo.toml unchessed-core/Cargo.toml 2>/dev/null || true
```

Output was empty (`0` matching lines). A separate saved script scanned every Rust file under `unchessed-core/src` for `tablebase`, `syzygy`, `.rtbw`, `.rtbz`, and `tbprobe`; its exact output was:

```text
source_hits= 0
tablebase_files= 0
tablebase_file_paths= []
```

(The script labels the second line `tablebase_files`; the output above is reproduced semantically exactly except for that spelling correction.) A repository-wide file search excluding build artifacts also found **zero** `.rtbw` or `.rtbz` files. `unchessed-core/Cargo.toml` has an empty `[dependencies]` section. The UCI option listing in `unchessed-core/src/uci.rs` contains no `SyzygyPath`, probe-limit, or tablebase option; root search is in `search.rs` (`go_with_root_hints` and `negamax`) and has no tablebase hook.

This is a verified absence in the inspected source and working tree, not an assumption that an uninspected external binary has no support. Existing generated logs may mention other engines' Syzygy options, but those are not Unchessed implementation.

## Tier 1 real-world checks and negative results

Real testing was attempted because the standing rule makes it mandatory wherever feasible. `python-chess` was installed locally with:

```text
python3 -m pip install --user --quiet python-chess
```

The saved check script was run as:

```text
python3 /home/ubuntu/jobs/job_KZC1ecIq_a1/probe_check.py
```

For a real legal position (`8/8/8/8/8/8/2K5/2k5 w - - 0 1`), it attempted to open `/tmp/unchessed-no-syzygy` and probe WDL. The actual result was:

```text
source_hits= 0
tablebase_files= 0
tablebase_file_paths= []
probe_exception= FileNotFoundError [Errno 2] No such file or directory: '/tmp/unchessed-no-syzygy'
```

Thus a real local probe cannot run: there are no table files to open. This is the concrete blocker for design-only status; it is not a claim that the position is unprobeable in principle. The check also confirms that simply installing a client library does not supply the data.

A baseline engine smoke command was additionally attempted with a nonexistent Syzygy path. The release binary advertised the existing options only (Hash, Threads, MultiPV, evaluation/policy options, and search tuning options), with no Syzygy option. The run then exposed an unrelated existing panic in `unchessed-core/src/movegen.rs:367` (`index out of bounds: the len is 64 but the index is 64`) on the hand-constructed adjacent-king FEN. This is not tablebase evidence and is reported to avoid conflating an invalid baseline fixture with a successful integration test. No source was changed to work around it.

The repository's current Rust toolchain is `rustc 1.75.0` / `cargo 1.75.0`. A no-change baseline test was attempted:

```text
cargo test -p unchessed-core --lib --quiet
```

It was blocked before compilation because the checked-in lockfile uses an unsupported newer format:

```text
error: failed to parse lock file .../Cargo.lock
Caused by: lock file version 4 requires `-Znext-lockfile-bump`
```

This is an independent environment/toolchain blocker, not evidence about tablebase correctness. No lockfile regeneration or dependency change was made.

## What Syzygy provides

Syzygy has two complementary metrics. WDL50 returns five-valued information from the side-to-move perspective: unconditional win/draw/loss (`2/0/-2`) and cursed win/blessed loss (`1/-1`) where a theoretical win/loss is drawn by the fifty-move rule. `probe_wdl` is intended for positions immediately after a capture or pawn move (the zeroing boundary); an engine must apply the documented WDL probing/recursion conventions rather than treating the value as a generic static evaluation.

DTZ50'' is distance to the next zeroing move, with rounding/ambiguity near the 50-move boundary. Positive and negative values encode the side-to-move outcome and the urgency of a pawn move or capture. DTZ min/max selection can preserve a win under the fifty-move rule, but it is not depth-to-mate and can choose unintuitive progress moves. The authoritative Python documentation states that **both WDL and DTZ tables are required for DTZ probing**, and that root DTZ probing is slower. It also warns that a position generally requires the exact material table plus transitively reachable lower-material tables.

Syzygy files do not cover positions with castling rights. A production adapter therefore needs a conservative eligibility test: no castling rights, no unsupported variant, and a piece count within the configured probe limit (normally 5 or 6 initially, not an implicit promise of 7). Missing/corrupt tables must be a normal “no result” path, never a search failure; table files should be checksum-validated before use because malformed files can cause denial of service.

## Rust crate and license research

The candidate landscape was checked with the live crates.io index:

```text
cargo search syzygy --limit 10
```

Relevant results included `shakmaty-syzygy = 0.28.1`, `pyrrhic-rs = 0.2.0`, `fathom-syzygy = 0.1.0`, and `fathom-syzygy-sys = 0.1.0`. The current repository cannot use `cargo info` with Cargo 1.75 (the command is unavailable), so metadata was checked through crates.io and the projects' authoritative repositories/docs.

| Candidate | License/status | Integration assessment |
|---|---|---|
| [`shakmaty-syzygy`](https://docs.rs/shakmaty-syzygy) | GPL-3.0-or-later, verified from its GitHub repository | Mature Rust API (`Tablebase`, Wdl, Dtz, memory-mapped-filesystem feature), but GPL licensing is a serious compatibility question for this engine and should not be selected without an explicit project license decision. It uses `shakmaty` positions, so an adapter/conversion boundary is still needed. |
| [`pyrrhic-rs`](https://crates.io/crates/pyrrhic-rs) | MIT, verified on crates.io and its GitHub page; transliterated Pyrrhic code has additional upstream copyright acknowledgements | Most directly shaped for embedding in an existing engine. It wraps the original unsafe API in `TableBases`, requires an `EngineAdapter` for attack generation, and accepts colon-separated table paths. Its small community/low activity and unsafe-origin lineage require source audit, corpus testing, and license notices before adoption. |
| [`fathom-syzygy`](https://crates.io/crates/fathom-syzygy) / `fathom-syzygy-sys` | Candidate crates found, but not selected or built; inspect their bundled LICENSE and generated bindings before use | Fathom's upstream C project documents a permissive MIT license for its modifications and unrestricted redistribution of Ronald de Man's original code, with `tb_init`, `tb_probe_wdl`, and `tb_probe_root`. The upstream README says the original basil00 repository is unmaintained and points to jdart1/Fathom. FFI adds build/unsafe and cross-platform complexity. |

The permissive Rust option worth a future focused audit is therefore `pyrrhic-rs`, while `shakmaty-syzygy` is technically attractive but GPL-3.0-or-later. “Permissive” does not mean automatically safe: verify the complete transitive license set and preserve copyright/license files. No crate was added, downloaded into the repository, or compiled as part of this design-only item.

## Proposed integration boundary (future work, not implemented)

A future implementation should introduce a tablebase service owned by the search/engine lifecycle, initialized only after a validated `SyzygyPath` is configured. The service should expose a fallible `probe_wdl(position, halfmove_clock)` operation and a separate root-only DTZ operation. It should return `Unavailable` for ineligible positions, missing material tables, unsupported castling/variant state, or I/O errors, allowing normal NNUE/search evaluation to continue.

The position adapter must map Unchessed's board representation into the crate's required piece bitboards, side to move, en-passant state, and halfmove clock. It must be verified against legal move generation, promotions, en passant, check status, and color symmetry. WDL may be used as a bounded search terminal/ordering signal only after handling cursed/blessed outcomes and the current contempt policy explicitly. DTZ should initially remain root-only, where legal root moves can be made, probed, undone, and compared; it should not be injected into ordinary static evaluation or NNUE training labels without a separate design.

Every probe must be bounded by piece count and a cheap material signature, avoid probing in the presence of castling rights, and be cheap on the no-table/no-hit path. Shared immutable mappings plus memory-mapped files are plausible, but concurrency and file descriptor behavior require measurement. The engine must never assume that a configured directory is complete: 6-man probing can require lower-man tables after captures, as the Python documentation explicitly notes.

Required tests before any integration include known 3–5-man WDL and DTZ positions with checked expected values; side-to-move/color mirror tests; halfmove values around 0, 99, and 100; cursed-win/blessed-loss cases; promotion and en-passant transitions; missing-table fallback; corrupted/checksum-rejected files; castling-rights rejection; and concurrent probes on independent positions. A future real strength test must compare the opt-in implementation against the unchanged engine with fixed paired games; tablebase correctness tests are not an Elo claim.

## Integration versus file acquisition and hosting

**Integration** means code, dependency/license review, position conversion, eligibility checks, search/root policy, fallback behavior, and correctness/performance tests. It can be researched without possessing large tablebase collections. **Acquisition/hosting** means selecting 3–5-, 6-, or 7-man coverage, downloading or mirroring `.rtbw`/`.rtbz` files, allocating SSD capacity, checksumming, packaging, and deciding how users obtain them. The public Syzygy site lists approximately 939 MiB for 3–5-man, 149.2 GiB for 6-man, and 16.7 TiB for 7-man WDL+DTZ collections. Those storage and distribution costs are precisely why hosting remains a separate Tier 3 decision. This report does not download, vendor, or recommend hosting any collection.

## Verified versus assumed

| Statement | Status |
|---|---|
| No Syzygy/tablebase implementation matches in `unchessed-core/src` | **Verified** by focused grep and saved source scan; zero hits |
| No `.rtbw`/`.rtbz` files in the repository working tree | **Verified**; zero files found |
| No Syzygy UCI option in the current option listing | **Verified** by `uci.rs` inspection and binary UCI output |
| A real local probe can currently run | **Disproved**; python-chess probe hit `FileNotFoundError` because no table directory/files exist |
| Syzygy WDL/DTZ semantics and transitive material-table requirement | **Verified** from python-chess documentation and Syzygy reference material |
| `shakmaty-syzygy` is GPL-3.0-or-later | **Verified** from its authoritative GitHub repository |
| `pyrrhic-rs` is MIT and uses an engine attack adapter | **Verified** from crates.io/GitHub documentation; full transitive licensing still requires a future audit |
| Fathom exposes WDL/root-DTZ APIs and permissive licensing claims | **Verified** from upstream README; maintenance status and FFI suitability remain engineering considerations |
| Tablebases will improve Unchessed Elo by a particular amount | **Not measured and should not be claimed** |
| The best first coverage is 5-man or 6-man | **Assumed design trade-off**, dependent on future storage budget and target hardware |
| The adapter can be made correct without changing board representation | **Assumed until conversion tests are implemented** |

## References

1. [python-chess Syzygy probing documentation](https://python-chess.readthedocs.io/en/latest/syzygy.html) — WDL50/DTZ50'', return values, eligibility, transitive files, thread-safety, and missing-table behavior.
2. [Chessprogramming.org: Syzygy Bases](https://chessprogramming.org/Syzygy_Bases) — `.rtbw` versus `.rtbz`, search/root roles, Fathom API and licensing context, and size estimates.
3. [Syzygy tables official site](https://syzygy-tables.info/) — WDL/DTZ definitions, practical download sizes, and the explicit separation of local table use from hosted lookup.
4. [shakmaty-syzygy docs](https://docs.rs/shakmaty-syzygy/latest/shakmaty_syzygy/) and [repository](https://github.com/niklasf/shakmaty-syzygy) — Rust API, memory-mapped feature, GPL-3.0-or-later license.
5. [pyrrhic-rs crates.io](https://crates.io/crates/pyrrhic-rs) and [repository](https://github.com/Algorhythm-sxv/pyrrhic-rs) — MIT metadata and existing-engine adapter design.
6. [Fathom repository](https://github.com/basil00/Fathom) — upstream API, maintenance notice, and license text.

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/32-tablebase-probing.md`
