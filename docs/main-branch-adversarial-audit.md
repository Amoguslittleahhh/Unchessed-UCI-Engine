# Adversarial verification audit of `main`

**Audit target:** `main` at commit `818ef9dd5bb7be64fd6085f7c1910b953390da6e`
**Tree:** `062aa187b4d4eb3e4ff69681878f9a4acd946044`
**Audit snapshot:** `/home/ubuntu/unchessed-main-audit.tar.gz`
**Snapshot SHA-256:** `3a4a3586cd522f7305d16985308af4913de75ff19c8051d3bbcf42fc5c017618`
**Audit branch:** `manus/research-facilities`
**Scope rule:** The `main` branch and the immutable audit archive were not modified. Only this report is added to `manus/research-facilities`.

## Executive verdict

The audit found **twelve confirmed defects or material verification gaps**, including **five high-severity correctness/runtime defects** in the engine core and UCI path, plus several high-severity data-integrity and security defects in Python tooling. The most consequential findings are:

1. **Malformed but parse-accepted FEN can generate illegal castling or en-passant moves, corrupt incremental hashes, and panic the process.**
2. **A checkmate reached with halfmove clock 100 is scored as a draw.**
3. **Repetition is declared after a second occurrence rather than the required third occurrence.**
4. **Quiescence scores stalemate as static evaluation rather than a draw.**
5. **`go nodes N` and very short `go movetime N` limits are materially exceeded.**
6. **The adapter performs uncharged opponent-analysis searches before its timed move search.**
7. **`go searchmoves` is silently ignored and can return an unauthorized move.**
8. **Ponder commands are ignored while the engine emits `bestmove` immediately.**
9. **Duplicate PGNs can cross the v5 train/validation boundary despite supposedly game-disjoint splitting.**
10. **Malformed move-label rows can leave corrupt partial datasets and no manifest.**
11. **Several caller-selected PyTorch checkpoints use unsafe pickle deserialization.**
12. **An environment-derived cloud build template reaches `shell=True`, permitting command injection when the environment is attacker-controlled.**

The audit also found confirmed medium- and low-severity issues: promotion material is omitted from SEE; irrelevant en-passant state fragments hashes; the root and interior 50-move policies are inconsistent; final-named dataset shards are published non-atomically; teacher manifests accept caller-supplied false input hashes; depth/time calibration can hang on a silent engine; NNUE relabeling retains an entire input in memory; v5 manifests omit source content hashes; the documented build gate omits native-linker preflight; release mode does not run release tests; deep perft is ignored by the default gate; there is no CI configuration; formatting and strict Clippy are not enforced; Rust/Python dependency versions are not fully pinned; and OpenVINO is an undeclared optional tool dependency.

These findings are **not fixes**. They are verified descriptions of the `main` branch as audited. No edit to engine or tooling code was made.

## Audit method and evidence ladder

The audit used five independent read-only lanes: build/tests, core correctness, search/evaluation, UCI/runtime, and tools/data/security. Each lane extracted the exact `main` archive into a temporary directory and used temporary external harnesses where dynamic reproduction was needed. The source archive and both Git branches were left unchanged.

Evidence levels used in this report are:

- **Source inspection:** exact source paths and line ranges establish what the code says.
- **Automated test:** a relevant test or checker was personally run to completion.
- **Runtime execution:** a current binary or compiled extracted snapshot processed real input.
- **End-to-end/outcome evidence:** a realistic process ran with sufficient repetitions or scale. Most bug reproductions below are runtime evidence, not strength evidence.

A confirmed defect means the behavior was reproduced or directly established from executable control flow. A source-level security or resource finding may be confirmed as code behavior even when exploit payloads or production-scale stress were intentionally not executed.

## Baseline test and build results

### Claim: the core test suite passes

**Evidence level achieved:** **Automated test** and **runtime compilation**. The result establishes the tested suite on the extracted snapshot, not complete correctness.

**Exact method:**

```text
cd /tmp/main
PATH=/home/ubuntu/.cargo/bin:$PATH cargo test -p unchessed-core -- --test-threads=1
```

Result:

```text
118 passed; 0 failed; 6 ignored
```

The ignored deep perft gate was also run independently:

```text
cargo test -p unchessed-core perft::tests::deep -- --ignored
```

Result:

```text
1 passed; 0 failed; 13.49 seconds
```

An independent external harness also performed 1,000 randomized legal copy/make/hash checks.

**Verdict:** **partially confirmed.** The exercised core tests and deep perft assertion passed. The default gate does not cover all available correctness checks, and the passing tests did not catch the malformed-FEN, terminal-rule, quiescence, or SEE defects below.

**Anomalies found:** The deep perft test is ignored by default. The release build path and the normal build script have separate coverage limitations described next. The repository contains no CI workflow enforcing these checks.

**What this does not establish:** It does not establish correctness for malformed but accepted inputs, every legal position, 50/75-move and repetition semantics, exact limits, concurrent execution, cross-platform builds, sanitizer cleanliness, or playing strength.

**Confidence and what would raise it:** Confidence is high for the reported counts and medium for untested state-space behavior. A differential legal-move/fen fuzz campaign, sanitizer/Miri runs, and enforced CI would raise confidence.

### Claim: the documented build gate is self-contained and validates release behavior

**Evidence level achieved:** **Automated test** and **source inspection**.

**Exact method:** `scripts/build-and-test.sh` was inspected at lines 44--49, 57--70, and 87--90 in the `main` snapshot. The script checks for Cargo but not a native C linker. With the documented offline Rust toolchain and no `cc`, `gcc`, or `clang`, the command exited 101 with `error: linker cc not found`. Installing the audit VM's `build-essential` package allowed the same gate to exit 0.

The `--release` branch invokes release compilation but does not invoke `cargo test --release`. A separate direct command was run:

```text
cargo test --workspace --release --lib --bins
```

It passed 118 tests with 6 ignored. A full release workspace command using one bundled toolchain later failed at doc-tests because that bundle lacked `rustdoc`; the unit targets had passed before that failure.

The default `perft::tests::deep` test is marked ignored at `unchessed-core/src/perft.rs:101-109`, and no default build-gate command supplies `--ignored`. No CI directories or workflow files were found in `.github`, `.gitlab-ci.yml`, `.circleci`, `.buildkite`, `azure-pipelines.yml`, or `.azure-pipelines`.

`cargo fmt --all -- --check` exited 1 with 146 formatting hunks across 16 files. `cargo clippy --workspace --all-targets -- -D warnings` exited 101 with 37 diagnostics for the core library and 52 for core library tests under Rust 1.97.0. These diagnostics are version-sensitive quality findings, not automatically runtime defects.

**Verdict:** **partially confirmed.** The build gate can pass after native dependencies are installed and the core tests can pass. It is not self-contained, does not test optimized artifacts in its release branch, omits the deepest functional perft test by default, and is not automatically enforced by CI.

**Anomalies found:** The source archive has no `.git` directory. Consequently the Python suite's Git-dependent bracket test failed in that distribution form: `1 failed, 363 passed, 22 skipped, 326 subtests passed`; explicitly passing 21 Rust files to the checker passed. The moving `stable` toolchain and lower-bound-only Python requirements reduce reproducibility. OpenVINO is imported by `tools/npu_dispatch_benchmark.py` but appears in neither requirements file nor the development-environment documentation.

**What this does not establish:** The formatting and Clippy diagnostics do not prove engine misbehavior. The archive-specific Python failure does not prove failure inside a normal Git checkout. Absence of CI files was searched across the supplied repository conventions; it does not prove an external CI system could not exist elsewhere.

**Confidence and what would raise it:** High for the script and repository-configuration observations. Confidence in portability impact is medium. Pinning toolchains/dependencies, preflighting native tools, running release tests, and adding CI would resolve the gaps.

## Core correctness findings

### Claim: accepted FEN positions are safe and coherent

**Evidence level achieved:** **Runtime execution** backed by **source inspection**.

**Exact method:** `unchessed-core/src/fen.rs:11-55` in `main` parses board text without enforcing eight ranks and eight files per rank. Lines 63--93 accept castling and en-passant fields based primarily on syntax; only king-count validation is present. `movegen.rs:694-731` generates castling based on rights, path emptiness, and attack tests without confirming the king and rook exist on their required orthodox squares. `board.rs:325-400` assumes those pieces exist when making special moves.

An external probe compiled against the extracted `main` code produced:

```text
FEN: 4k3/8/8/8/8/8/8/4K3 w K - 0 1
castle_legal=["e1g1"]
castle_hash_incremental=939b9ada67fd5b78
recomputed=b22245a5deffe394
 equal=false
```

A FEN with the white king on e2 and `K` rights caused `legal()` to attempt synthetic `e1g1` and panic at `board.rs:332:40` with `make: no piece on from-square`. The parser also accepted a five-rank FEN, a seven-square rank, and an empty rank. En-passant is similarly weak: `movegen.rs:627-635` can emit en-passant without confirming a capturable enemy pawn, while `board.rs:354-359` removes one unconditionally. A malformed en-passant position produced a legal-looking `e5d6` with `hash_equal_after_make=false`.

UCI reaches this parser through `uci.rs:831-890`, including the `position fen` path.

**Verdict:** **confirmed.** The behavior is externally reachable and can cause illegal moves, state/hash corruption, and a process panic from malformed-but-accepted FEN. This is an availability and correctness defect, not a demonstrated memory-safety vulnerability.

**Anomalies found:** Valid-position randomized copy/make/hash checks passed 1,000 iterations. A valid en-passant probe preserved the recomputed hash. Deep perft passed. Those positive tests do not cover malformed accepted state.

**What this does not establish:** It does not prove every malformed FEN causes failure or that valid FEN handling is broadly incorrect. It does establish that the parser accepts structurally invalid inputs and downstream code trusts them.

**Confidence and what would raise it:** High. The panic, illegal castling, and hash mismatch were reproduced. Confidence would increase with a complete FEN grammar/property-fuzz suite and differential legal-state generation.

### Claim: terminal draw and mate handling follows chess rules

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `search.rs:506-512` returns `draw(ply)` for `halfmove >= 100` before checkmate handling at later lines. In a known mate-in-one position with halfmove clock 99, `a1a8` produced a child with halfmove 100 and zero legal moves. Depth-one search reported:

```text
mate_at_100_move=a1a8 child_halfmove=100 child_legal_moves=0
mate_at_100_score=0 (mate_threshold=29488)
```

**Verdict:** **confirmed.** A checkmate reached at halfmove 100 is scored as a draw in the tested path. The implementation also conflates the internally used 100-halfmove threshold with a terminal policy; whether the root should claim or continue is a policy question, but mate precedence is not.

**Anomalies found:** The root and interior behavior is inconsistent. The root loop at `search.rs:1026-1060` can search and choose a resetting capture from a halfmove-100 root, while interior nodes return an immediate draw at the same threshold.

**What this does not establish:** It does not decide whether the engine intends the FIDE claimable 50-move rule, automatic 75-move rule, or a search approximation. It establishes that the current terminal ordering loses mate and that root/interior policy differs.

**Confidence and what would raise it:** High for mate precedence and root/interior inconsistency. A rule-policy specification plus a full terminal-state matrix would resolve the remaining policy question.

### Claim: repetition detection implements threefold repetition

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `search.rs:291-315` uses `.any(|&h| h == hash)` to identify an already occurring position, and `search.rs:506-512` treats one prior match as a draw. An external probe started from `4k1n1/8/8/8/8/8/8/Q3K3 w - - 0 1`, played `e1f1 g8f6 f1e1`, and examined `f6g8`. The child matched the initial position with one historical occurrence, meaning two total occurrences after the move. Output:

```text
twofold_child_equals_initial=true
child_halfmove=4
historical_occurrences=1
twofold_move=f6g8 score_with_one_prior_occurrence=0 score_without_history=-865
```

**Verdict:** **confirmed.** The tested path declares a draw on the second occurrence rather than the required third occurrence.

**Anomalies found:** The search comments describe an “already occurred” heuristic rather than clearly implementing a rules-level repetition count. The exact interaction with root history and irreversible moves was not exhaustively fuzzed.

**What this does not establish:** It does not establish how every GUI adjudicator or external game manager treats repetition. It establishes the engine's own score behavior in the reproduced twofold case.

**Confidence and what would raise it:** High. A reference repetition-key test over randomized legal games and explicit twofold/threefold/fivefold fixtures would raise coverage.

### Claim: Zobrist identity normalizes irrelevant en-passant state

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `board.rs:308-323` XORs an en-passant file key whenever `ep != NO_EP`; lines 334--338 and 374--377 maintain it unconditionally. In `4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1`, no black pawn can capture e3. Legal move lists were identical with and without the en-passant field, but hashes differed:

```text
irrelevant_ep_legal_equal=true
ep=81607e0d394685b4
no_ep=9e853282a70936ab
equal=false
```

**Verdict:** **confirmed.** The position hash distinguishes an en-passant field that confers no legal capture. This can fragment TT identity and miss repetitions if the hash is used as the repetition key.

**Anomalies found:** The exact treatment of a pinned pseudo-legal en-passant opportunity was not exhaustively tested. The correct key requires a legally available capture, not merely a syntactically plausible target.

**What this does not establish:** It does not establish that every TT collision or repetition decision is wrong. It establishes the specific normalization mismatch.

**Confidence and what would raise it:** High for the shown position. A reference repetition-key implementation and pinned-EP corpus would raise confidence.

### Claim: SEE accounts for promotion material

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `see.rs:140-154` initializes gain with the captured piece and handles the promoted piece only as later attacker material; the backward pass at 201--209 cannot add the pawn-to-piece material delta. Search uses SEE for ordering and pruning at `search.rs:342-375` and 429--436. A standalone probe using pawn 82, rook 477, and queen 1025 reported:

```text
see_quiet_promotion=0
see_capture_promotion=477
```

Expected material deltas for an undefended queen promotion are +943, and for an undefended rook capture followed by queen promotion +1420.

**Verdict:** **confirmed.** Promotion gain is omitted from SEE. The defect can misorder promotions and affect SEE-based pruning.

**Anomalies found:** Ordinary captures and the existing SEE tests pass. The missing term is specific to promotion material accounting.

**What this does not establish:** It does not prove a particular game outcome changes in the default configuration. It establishes incorrect SEE semantics for the reproduced moves.

**Confidence and what would raise it:** High. A promotion SEE matrix covering all promotion pieces, captures, recaptures, and defenders would raise confidence.

## Search and evaluation findings

### Claim: search limits are hard resource bounds

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `search.rs:271-288` checks node and time limits only when `self.nodes & 2047 == 0`; node increments occur in qsearch and negamax before those checks. An external harness requested one node:

```text
nodes_1: elapsed_ms=3173
result=[("d2d4", 14, 4, ...)]
```

The run completed four iterations and did not stop at one node. A separate UCI harness reported a final info line with 1,088 nodes for `go nodes 1`. The code therefore permits substantial overshoot for small limits.

`Limits::budget` at `search.rs:143-145` imposes a five-millisecond minimum for movetime. Root setup calls legal move generation before the deadline can be observed. Lazy magic-table construction occurs in `movegen.rs:334-339` and 735--743. Fresh-process `Limits::movetime(1)` probes returned 3,188, 3,208, and 3,279 ms, each only reaching depth 1.

**Verdict:** **confirmed.** Small node limits and very short time limits are not hard bounds. The large cold-start overrun is reproducible, while exact latency is host-dependent.

**Anomalies found:** A warmed `movetime_1` path in one harness returned in 8 ms, showing that initialization dominates some runs. This does not rescue the hard-limit contract. The five-millisecond floor also makes a one-millisecond request semantically different from its literal request.

**What this does not establish:** It does not establish typical overrun for ordinary long time controls or that all search configurations overrun by seconds. It establishes failure of the requested small-bound behavior.

**Confidence and what would raise it:** High for the reproduced cases. Countdown-based node checks, pre-initialization, and a cold/warm deadline matrix would establish the repaired boundary.

### Claim: quiescence handles stalemate as a draw

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `search.rs:395-404` initializes a non-check qnode's best score from static evaluation and generates captures only. Lines 416--480 return that score unless the side is in check with no legal capture. There is no non-check/no-legal-move stalemate return. A direct harness produced:

```text
q_stalemate: (-1245, -1245)
q_checkmate: (-1256, -30000)
```

The stalemate FEN was `7k/5Q2/6K1/8/8/8/8/8 b - - 0 1`; the control checkmate FEN was `7k/6Q1/6K1/8/8/8/8/8 b - - 0 1`.

**Verdict:** **confirmed.** A legal stalemate qnode is scored as static evaluation rather than the draw score.

**Anomalies found:** The checkmate control followed the expected terminal path, so this was not a generic terminal-test failure. A root-level exhaustive comparison was attempted to assess move-selection reachability; the direct qsearch defect remains confirmed even where a particular root fixture may not select a different move.

**What this does not establish:** It does not establish the frequency of the bug in games or that every stalemate changes a root decision.

**Confidence and what would raise it:** High for qsearch behavior. A curated legal stalemate suite integrated at root, qsearch, and transposition boundaries would raise end-to-end confidence.

### Claim: the TT, mate-score conversion, NNUE state, and SIMD paths are free of confirmed defects

**Evidence level achieved:** **Source inspection** plus **targeted automated tests**.

**Exact method:** The extracted core tests passed 118 tests. The focused test `nnue::tests::simd_kernels_match_scalar_reference` passed on a host advertising `avx2 fma`. `tt.rs:145-193` was inspected for score clamping, mate conversion, and XOR-key reads. No deterministic false TT hit, NNUE incremental-state mismatch, or AVX2/FMA parity failure was reproduced.

**Verdict:** **partially confirmed.** The exercised paths passed and no defect was independently reproduced. This is not confirmation of universal safety.

**Anomalies found:** Nonfinite NNUE payload handling remains a source-level suspicion: `nnue.rs:151-161` deserializes f32 values, while 410--430 validates size and dimensions but not finiteness. A full malicious correctly sized model was not executed.

**What this does not establish:** No TSAN, Miri, cross-platform, 32-bit, non-AVX2, adversarial model corpus, or long-running concurrent stress campaign was run.

**Confidence and what would raise it:** Medium for untested paths. Sanitizers, differential model-loading tests, and repeated concurrent stress would raise it.

## UCI and runtime findings

### Claim: the adapter obeys the requested movetime

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `uci.rs:1171-1267` performs pending-opponent analysis, including a depth-14 or 400,000-node search and potentially a depth-12 or 250,000-node search, before the actual timed search. The search clock begins later in `search.rs:915-919`.

Reproduction against the extracted release adapter, with `Threads=1`, `OwnBook=false`, and the shipped NNUE, used `position startpos moves a2a3` followed by `go movetime 5`. It returned `bestmove g8h6` after approximately 664.8 ms and logged opponent analysis. This is approximately 133 times the requested move time.

**Verdict:** **confirmed.** Adapter-side analysis is not charged to the requested move budget.

**Anomalies found:** The magnitude depends on the position, model, host, and options. This is separate from the generic search polling defect.

**What this does not establish:** It does not establish that every adapter move exceeds its budget or that non-adapter reviewer searches have the same pre-search work.

**Confidence and what would raise it:** High for the tested pending-opponent path. A full UCI timing matrix across options and clocks would quantify scope.

### Claim: multi-threaded UCI searches return by the deadline

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `uci.rs:1388-1431` spawns `Threads-1` helper searches in a scoped thread block. The scope must join before `bestmove` is printed at 1463. Each helper uses the same nominal limit, while search cancellation is coarse at 2,048-node polling.

With accepted `Threads=64` and `go movetime 5`, a real binary returned `bestmove e2e4` after 54.7 ms. Earlier fresh trials returned 37.4, 58.9, and 34.8 ms.

**Verdict:** **confirmed.** The permitted configuration materially exceeds the five-millisecond request.

**Anomalies found:** Absolute latency is host/load-dependent. The join requirement means even a main-search stop does not guarantee prompt UCI response if helpers remain active.

**What this does not establish:** It does not establish a failure at ordinary time controls or a data race. It establishes a deadline violation under the tested permitted configuration.

**Confidence and what would raise it:** High for deadline violation. A deterministic cancellation/join benchmark across thread counts and loads would raise coverage.

### Claim: `searchmoves` restricts the root move set

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** `uci.rs:892-920` recognizes depth, clocks, nodes, and infinite, but has no `searchmoves` branch. The real Reviewer binary was given:

```text
position startpos
go depth 1 searchmoves e2e4
```

It returned:

```text
bestmove d2d4
```

**Verdict:** **confirmed.** The command is silently ignored and the engine can return a move outside the requested root set.

**Anomalies found:** Invalid position move lists were rejected cleanly in separate smoke tests. The defect is specifically unsupported restricted-root semantics.

**What this does not establish:** It does not establish behavior for any future protocol implementation or external GUI workarounds.

**Confidence and what would raise it:** High. A protocol conformance corpus should include one-move, multiple-move, illegal-root, and empty-intersection cases.

### Claim: pondering obeys the UCI ponder lifecycle

**Evidence level achieved:** **Runtime execution** and **source inspection**.

**Exact method:** The same parser region has no `ponder` or `ponderhit` handling, and the command loop at `uci.rs:370-421` has no ponderhit dispatch. A real binary given `go ponder movetime 50` returned `bestmove b1c3` in 27.9 ms without `ponderhit` or `stop`.

**Verdict:** **confirmed.** Ponder is not implemented, and the engine emits a best move instead of waiting for the lifecycle transition.

**Anomalies found:** Earlier `go ponder depth 2` and `go ponder infinite` probes also behaved as ordinary searches. Unknown tokens are ignored rather than rejected.

**What this does not establish:** It does not establish whether a particular GUI requires pondering or whether ordinary analysis is otherwise stable.

**Confidence and what would raise it:** High. A GUI-like hit/miss/race harness would be required to verify a future implementation.

## Python data, security, and operational findings

### Claim: v5 train/validation splits are game-disjoint under duplicate input games

**Evidence level achieved:** **End-to-end runtime reproduction** and **source inspection**.

**Exact method:** `tools/pretrain_v5_data.py:342-343` derives `game_hash` from source basename plus Round/ordinal, and line 537 passes `Path(path).name`. Identical games copied into differently named PGNs receive different identities. Lines 598--610 then can assign them to opposite splits.

A hermetic reproducer created identical `source-a.pgn` and `source-b.pgn` and used `--val-games 1`. It returned success but showed:

```text
GAME_HASH_INTERSECTION=[]
ROWWISE_SEMANTIC_EQUAL=True
semantic digest: 6319c99cd345575d9b4a375d6d478875d0b57cae2acb23b8d4d2d652324deb85
```

**Verdict:** **confirmed.** The hash sets are disjoint while semantically identical games cross the train/validation boundary. This violates the intended guarantee whenever sources overlap or contain renamed duplicate games.

**Anomalies found:** The tool reports a successful build, making the leakage silent. The problem is identity construction, not a parser crash.

**What this does not establish:** It does not establish leakage for every unique, non-overlapping corpus. It establishes a reproducible leakage case under ordinary duplicate-input conditions.

**Confidence and what would raise it:** High. A corpus-level canonical PGN/game identity and duplicate-leakage regression suite would raise confidence.

### Claim: malformed rows fail without publishing corrupt datasets

**Evidence level achieved:** **End-to-end runtime reproduction** and **source inspection**.

**Exact method:** `tools/pretrain_move_dataset.py:186-198` preallocates NumPy arrays, lines 200--229 continue after invalid rows without retaining a valid-row index, and lines 301--306 remove a trailing prefix based on the number of bad rows rather than actual bad indices. Shards are written directly at lines 312--314 before manifest/histogram completion.

A five-row JSONL input with an invalid middle row produced:

```text
return code: 1
stderr: KeyError: 17
shard-train.npz GAME_IDS=[10, 32587, 12]
shard-train.npz ENGINE_IDS=[0, 17, 0]
shard-val.npz GAME_IDS=[13]
MANIFEST_EXISTS=False
```

**Verdict:** **confirmed.** An invalid middle row can remain as uninitialized/corrupt data, valid rows can be discarded, and partial shards can remain after failure without a manifest.

**Anomalies found:** The failure is not cleanly fail-closed. The corruption is visible in the generated artifact rather than only in logs.

**What this does not establish:** It does not establish corruption for every invalid-row position or for successful all-valid input.

**Confidence and what would raise it:** High. Tests should cover invalid first, middle, last, duplicate, missing, NaN, and out-of-range rows with atomic publication assertions.

### Claim: caller-selected PyTorch checkpoints are safely loaded

**Evidence level achieved:** **Source inspection**; exploit execution was intentionally not performed.

**Exact method:** The following `main` paths use `torch.load(..., weights_only=False)` or equivalent unsafe pickle-capable loading: `tools/export_unarchitectured_v1.py:37`, `tools/pretrain_v1_a100.py:482-483`, `tools/train_nnue_xt_a100.py:402`, `tools/train_unarchitectured_v1_a100.py:1015,1372,1661,1704`, and `tools/train_unarchitectured_v1_student_a100.py:726`. Caller-selected `--resume`, `--oracle`, `--student`, or checkpoint paths reach these loads. In several paths, format checks occur only after deserialization.

**Verdict:** **confirmed** conditional security defect. If an attacker can cause a user or CI job to load an untrusted checkpoint, pickle deserialization can execute code with that account's privileges.

**Anomalies found:** Torch was unavailable in the audit sandbox, so a malicious payload was not run. The risk is established by the API and input flow, not by a live exploit.

**What this does not establish:** It does not establish that the normal production workflow accepts attacker-controlled paths or that privilege escalation occurs. It establishes unsafe deserialization for untrusted input.

**Confidence and what would raise it:** High for API behavior; medium for deployment impact. A safe-loader test and an explicit trusted-artifact policy would resolve operational scope.

### Claim: cloud engine fetching cannot execute environment-derived shell text

**Evidence level achieved:** **End-to-end runtime reproduction** with a hermetic command path.

**Exact method:** `tools/maia3_cloud_selfplay/generate.py:1252-1255` interpolates `SF_ARCH` and virtual-environment hints into a command string. `_run` at 1196--1206 invokes strings with `shell=True`.

A temporary environment set `SF_ARCH` to `x86-64; touch /tmp/<temporary>/injected #`. The formatted command contained the injected command and execution created the marker file:

```text
FORMATTED_COMMAND=make -j1 build ARCH=x86-64; touch /tmp/.../injected # COMP=gcc COMPCXX=g++
MARKER_EXISTS=True
```

**Verdict:** **confirmed** conditional command-injection defect. It requires attacker control over the build-process environment or equivalent template input.

**Anomalies found:** The reproducer used no network access and no elevated privileges. The injected command ran with the account executing the tool.

**What this does not establish:** It does not establish root compromise or that an external service exposes these variables to untrusted users.

**Confidence and what would raise it:** High for shell execution behavior. A subprocess argument-vector implementation and hostile-environment regression test would resolve it.

### Claim: v5 shard publication is crash-safe and atomic

**Evidence level achieved:** **End-to-end runtime reproduction** and **source inspection**.

**Exact method:** `tools/pretrain_v5_data.py:488-491` opens the final shard pathname directly and writes a zero placeholder header. Lines 503--508 backpatch and fsync later; no temporary file plus `os.replace` is used.

A temporary writer reproducer observed the final name before close:

```text
FINAL_NAME_VISIBLE=shard-000-000.v5
SIZE_BEFORE_CLOSE=21824
HEADER_ALL_ZERO=True
RECORD_BYTES_AFTER_HEADER=21760
PARSE_HEADER=ValueError bad data magic b'\x00...' expected b'UNCHD5R0'
```

The same direct-write publication pattern was found in `tools/train_nnue.py:377-389` and `tools/nnue_relabel_existing.py:132-144`.

**Verdict:** **confirmed.** An interruption, kill, or disk-full event can leave a final-named corrupt artifact that looks published.

**Anomalies found:** This is a crash/integrity path; ordinary completion can still produce a valid file. The v5 header is intentionally backpatched, but the final pathname is visible too early.

**What this does not establish:** It does not establish corruption on normal successful completion or data loss under every filesystem.

**Confidence and what would raise it:** High. Kill/disk-full tests with temporary publication and post-crash directory scans would raise confidence.

### Claim: teacher manifests faithfully attest their input bytes

**Evidence level achieved:** **End-to-end runtime reproduction** and **source inspection**.

**Exact method:** `tools/unarchitectured_v1_uci_teacher_worker.py:291` accepts optional `--input-sha256`, but line 343 records that value verbatim instead of comparing it with `sha256_file(args.input)` at line 345's output-hash path.

A valid one-record shard plus a fake local UCI engine was run with a false input hash. It returned success and produced:

```text
MANIFEST_INPUT_SHA256=not-the-real-sha256
ACTUAL_INPUT_SHA256=f5c0ccc551bc723081ea9cda959c8a227be9e40d4aaa4312328d100861120ac7
MISMATCH_ACCEPTED=True
```

**Verdict:** **confirmed.** The manifest can contain a false input digest while the worker exits successfully.

**Anomalies found:** Output hashing is computed, so the defect is specifically the optional input assertion being treated as metadata rather than verified evidence.

**What this does not establish:** It does not establish that every downstream consumer trusts the false field or that output bytes are otherwise invalid.

**Confidence and what would raise it:** High. A test should pass an incorrect digest and require a nonzero exit with no published output.

### Claim: depth/time calibration enforces its timeout

**Evidence level achieved:** **End-to-end runtime reproduction** and **source inspection**.

**Exact method:** `tools/unarchitectured_v1_depth_time_calibration.py:77-81` and 94--108 call blocking `proc.stdout.readline()` inside deadline loops. A silent fake engine was run with `--timeout 0.1` under an outer three-second timeout:

```text
timeout 3s python3 tools/unarchitectured_v1_depth_time_calibration.py \
  --engine /tmp/audit-silent-engine.py \
  --model /tmp/nonexistent \
  --timeout 0.1
outer_timeout_rc=124
```

**Verdict:** **confirmed.** A silent or stalled engine can block beyond the advertised internal timeout.

**Anomalies found:** The fake engine was deliberately silent, which is the relevant boundary case. Normal responsive engines may not show the hang.

**What this does not establish:** It does not establish a hang in every calibration command or operating-system process configuration.

**Confidence and what would raise it:** High. Nonblocking I/O or a reader thread with a hard kill path plus silent-engine tests would raise confidence.

### Claim: NNUE relabeling is bounded in memory for large shards

**Evidence level achieved:** **Source inspection**. Production-scale OOM was not attempted.

**Exact method:** `tools/nnue_relabel_existing.py:66-73` reads the complete score sidecar into `raw` and a Python list. Lines 107--113 retain every 104-byte record in `blobs` plus all old scores. Output begins only at line 140 after all inputs are retained. There is no CLI maximum or streaming two-file implementation.

**Verdict:** **confirmed** resource-bound defect. Memory use is proportional to the entire shard plus Python object overhead and can exhaust memory before output begins.

**Anomalies found:** The audit did not stress a giant production file to avoid exhausting the VM. This is direct source behavior, not a measured production OOM.

**What this does not establish:** It does not give an exact maximum safe shard size or prove an OOM on every host.

**Confidence and what would raise it:** High for full-input retention; medium for operational severity. A bounded-memory stress test with representative shard sizes would quantify impact.

### Claim: v5 manifests bind outputs to immutable input provenance

**Evidence level achieved:** **Source inspection**.

**Exact method:** `tools/pretrain_v5_data.py:649-679` records source path strings at line 652 and generated-shard hashes at 669--677, but no source SHA-256 or source revision/content digest. By contrast, `tools/pretrain_move_dataset.py:284-295` records stronger input provenance.

**Verdict:** **confirmed.** A v5 manifest cannot independently prove which source bytes produced its output after a path is reused or changed.

**Anomalies found:** Generated outputs are hashed, so the gap is specifically input binding rather than output hashing.

**What this does not establish:** It does not establish that a particular v5 dataset was changed or forged.

**Confidence and what would raise it:** High. A rebuild from changed same-path sources with manifest comparison would demonstrate the practical reproducibility failure.

## Genuine improvements and enhancements supported by evidence

This section records improvements that are directly motivated by verified failures. It is not an implementation plan and does not claim any change has been made.

1. **Make FEN parsing strict and special moves defensive.** Validate exactly six fields, eight ranks, eight files per rank, legal side/castling syntax, coherent king/rook placement, en-passant rank and backing pawn, and reject impossible state before move generation. Keep defensive checks in castling and en-passant make paths so malformed external input cannot panic.
2. **Separate rules terminal detection from search heuristics.** Checkmate and stalemate must precede draw scoring. Implement explicit policy for claimable 50-move, automatic 75-move, repetition, and insufficient-material semantics. Require two prior occurrences for a threefold repetition claim.
3. **Normalize the repetition key.** Include en-passant only when a legal en-passant capture exists, or maintain a rules-specific normalized repetition key distinct from the raw FEN serialization.
4. **Correct SEE promotion gain.** Add promoted-piece value minus pawn value to initial gain and add promotion fixtures to the SEE suite.
5. **Make resource limits observable and hard.** Check node caps every node or with a countdown, check deadlines before and after root setup, account for adapter pre-search work, and design helper cancellation so UCI response does not await unbounded helper work.
6. **Implement or reject unsupported UCI commands explicitly.** `searchmoves` must constrain the root set; ponder must either implement the lifecycle or return a documented unsupported response rather than silently behaving as ordinary search.
7. **Use canonical game identity and atomic data publication.** Canonicalize duplicate games across filenames, write to temporary names, fsync as appropriate, and atomically rename only after complete validation and manifest creation.
8. **Fail closed on metadata and malformed rows.** Verify caller-supplied hashes against bytes, track valid-row indices explicitly, and remove partial artifacts on failure.
9. **Use safe checkpoint loading and argument-vector subprocesses.** Avoid unsafe pickle loading for untrusted paths and never interpolate environment-derived values into `shell=True` command strings.
10. **Enforce the gates in CI.** Add exact toolchain/dependency metadata, native-linker preflight, release tests, deep perft, formatting, lint, Python archive/Git behavior, and conditional hardware tests with explicit skip reasons.

None of these improvements was applied during this audit.

## Stability assessment

The current `main` branch has useful baseline stability on valid, ordinary paths: 118 core tests passed, deep perft passed when explicitly enabled, the compiled UCI smoke passed, SIMD scalar parity passed on the available AVX2/FMA host, and 1,000 randomized legal copy/make/hash checks passed. The source also contains structural and CRC checks for TensorPackage and malformed-length handling for Polyglot books.

That stability is bounded. The audit reproduced process panic from malformed-but-accepted FEN, terminal-score errors, exact-limit violations, protocol contract failures, data corruption after malformed rows, final-name publication of incomplete shards, a timeout hang, and conditional command execution. Therefore the appropriate characterization is:

> **Ordinary valid-input paths have meaningful regression coverage, but the branch is not robust at adversarial input, strict resource-bound, protocol-compliance, or crash-integrity boundaries.**

No playing-strength, Elo, or production-readiness conclusion follows from the passing tests or the bounded runtime probes.

## Not verified / out of scope

The following were not verified and must not be inferred from this audit:

- Exhaustive legal chess correctness across the game tree.
- Differential agreement with Stockfish, python-chess, Syzygy, or another independent legal-move oracle over a large randomized corpus.
- Memory safety under Miri, AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer, or Valgrind.
- Data races or deadlocks in transposition-table access and Lazy SMP under sanitizer or long-duration stress.
- 32-bit, Windows, macOS, ARM, non-AVX2, or non-FMA builds.
- GPU, Torch, ONNX, A100, cloud, or production-scale training behavior.
- Execution of a malicious full-sized PyTorch checkpoint payload; unsafe deserialization was confirmed by code inspection and input flow only.
- Production-scale OOM measurements for relabeling or data builders.
- Every malformed FEN, malformed NNUE file, malformed package, malformed book, and hostile UCI command sequence.
- A complete UCI conformance test against all GUI lifecycle commands.
- A real fixed-suite NPS regression with stable aggregate timing.
- Human-versus-engine games.
- A statistically powered new engine-versus-engine match or SPRT decision for `main`.
- Elo, playing strength, tactical strength, or default-change benefit.
- Any remediation. No engine or tooling source was edited as part of this audit.

To raise confidence, run the proposed checks in an isolated follow-up: strict FEN and protocol fuzzing, differential legal-state testing, terminal-rule reference fixtures, hard-limit timing/node tests, sanitizer and race-detector builds, crash-injection publication tests, safe-loader tests, canonical duplicate-corpus tests, and a provenance-complete paired-game campaign for any game-facing candidate.

**Final audit boundary:** This report describes independently verified behavior of `main` at commit `818ef9dd5bb7be64fd6085f7c1910b953390da6e`. It does not describe the correctness of any uncommitted or future repair on `manus/research-facilities`.
