# SFNNv16 Reference Audit

## Provenance

The supplied `stockfish-upstream-source.zip` identifies official Stockfish commit `edb0d9db6731067ec50ce619ff372b463bc4dd5d` and contains a reduced audit bundle plus the official network file. The bundle is not a complete build tree because it omits core position, move-generation, layer, and UCI translation units. The exact commit was therefore checked out separately outside this repository and built as a reference executable. Neither the source checkout nor the network file is a repository dependency.

## Architecture observations

The audited definitions use three input families: a king-and-piece perspective feature set, a full-threat feature set, and a three-wide pawn-pair feature set. The feature transformer combines the families into a 1024-dimensional transformed representation with two perspective halves. The transformer maintains accumulator state and supports sparse changed-feature updates.

The positional network has an initial sparse-input affine layer with 32 outputs. It forms paired activation channels using squared clipped and clipped rectified activations, applies a 64-to-32 affine layer, forms a second pair of 32-wide activation channels, and applies a final 128-to-1 affine output. Eight PSQT buckets are associated with the transformer and eight layer stacks are declared. The audited output applies a skip-style correction using two initial-layer channels before integer scaling.

The king-and-piece feature definition uses a perspective-normalized king bucket scheme and a mirrored board orientation. Its declared feature dimension is derived from 64 squares and 11 piece-square groups divided across the two perspectives. The source declares a maximum of 32 simultaneously active feature changes for incremental updates.

The audit also records the following independently useful design constraints: feature hashes are embedded in network metadata; quantized parameter serialization is distinct from runtime SIMD permutation; the first layer is optimized for sparse input; the network output is scaled after integer propagation; and the public evaluator applies additional material, optimism, and rule-50 scaling outside the raw network output.

## Clean-room implementation boundary

These observations are architectural facts used to guide an independently authored Unchessed design. No Stockfish source code, weight bytes, feature-index tables, hash constants, layer templates, or implementation fragments were copied into Unchessed. The Unchessed tactical module uses legal-state summary features and a bounded gated multi-head projection. It is comparable in design intent but is not numerically equivalent to the audited network.

## Reference build record

The exact upstream commit was compiled with the Stockfish `x86-64-modern` build target. The executable and network checksums are recorded in `unchessed-eval-bar/results/reference/environment.txt`. Black-box comparisons use only UCI commands and preserve the reference binary outside the repository.

## Parity conclusion

The audit removes uncertainty about the supplied architecture definition, but architecture knowledge alone cannot produce stable 100/100 numerical parity. That requires matching trained weights, training distribution, quantization procedure, and a predeclared untouched evaluation corpus. Current results must therefore report measured gate vectors rather than an unverified parity score.
