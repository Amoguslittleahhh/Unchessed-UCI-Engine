# Policy and NNUE improvement round — 2026-09-05

This round implements two training-pipeline improvements chosen for compatibility with the existing Unchessed runtime and for measurable validation value.

## Policy network: local skill coherence

The dual-Elo oracle already replaces the inert single scalar rating path with separate self- and opponent-skill projections. This round adds an optional symmetric KL regularizer between predictions at nearby skill values. During training, the same position is evaluated at `rating - 200` and `rating + 200`; the regularizer discourages abrupt distribution changes while allowing the preferred move to change naturally. The canonical pretraining configuration enables the term at weight `0.01`.

This is motivated by Maia-2, which introduces skill-aware attention to model human play coherently across a wide skill range and evaluates coherence separately from accuracy [1]. The implementation is intentionally local and conservative: it does not force identical policies across ratings, and setting `skill_coherence_weight` to `0` reproduces the prior objective.

## NNUE: duplicate-safe validation

The NNUE trainer now assigns validation by a deterministic board-position hash rather than randomly splitting records. All records with the same 96-byte board representation stay in the same partition, preventing exact-position leakage from nearby plies, duplicate games, or relabelled shards. The exported `UNCHNNUE` format, HalfKAv2_hm feature scheme, factorized training table, piece-count output buckets, and Rust runtime are unchanged.

This addresses a generalization failure mode consistent with the NNUE dataset literature: noisy or redundant position construction can produce optimistic validation and poor deployment behavior [2]. It complements, rather than replaces, the existing quiet-position filtering and WDL-oriented loss.

## Toolchain

The branch was validated with Rust `1.98.1` and Cargo `1.98.1`. Official Cute Chess `1.5.1` was installed as `/home/ubuntu/.local/bin/cutechess-cli`; its AppImage required the Ubuntu `libopengl0` runtime library.

## Validation

The Rust workspace passed **129 tests**, with six existing ignored tests, and the release build completed successfully. Direct UCI validation returned `readyok` and a legal node-limited `bestmove`. Python sources passed bytecode compilation. The NNUE trainer integration tests are present but skipped on this host because PyTorch is not installed. The repository-wide `cargo fmt --check` reports pre-existing formatting differences in unrelated files; no formatter changes were applied to avoid touching the user’s baseline.

The short Cute Chess process-level match did not complete on this sandbox within the bounded timeout, although `cutechess-cli --version` launches successfully. A real Elo claim still requires a longer run on a host with a supplied NNUE and trained dual-Elo policy artifact.

## References

[1]: https://arxiv.org/html/2409.20553v1 "Maia-2: A Unified Model for Human-AI Alignment in Chess"
[2]: https://arxiv.org/html/2412.17948v1 "Study of the Proper NNUE Dataset"
[3]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html "Stockfish NNUE documentation"
[4]: https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/ "Technical Explanation of Leela Chess Zero"
