# Stockfish reference provenance

The user supplied an upstream Stockfish source archive for reference-only analysis. It was inspected outside the repository at `/tmp/stockfish-upstream-reference/stockfish-upstream-source`.

Archive SHA-256: `3db4129e7086bd149f7d9b7930219c32fe1f52182865547ea0d384dde6e3e286`

The archive provenance identifies the official Stockfish repository, master commit `edb0d9db6731067ec50ce619ff372b463bc4dd5d`, and GPLv3 licensing. It contains evaluation/UCI source and an NNUE binary. None of those files are copied into this repository, linked as build inputs, or used as runtime dependencies.

The supplied provenance revealed that the older files under `docs/source-evidence/` were exact or near-exact Stockfish source copies. Those files have been removed from the research branch. This correction is required before making any clean-room parity claim.
