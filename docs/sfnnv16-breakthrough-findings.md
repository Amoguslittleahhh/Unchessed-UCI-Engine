# SFNNv16 breakthrough findings

## Primary sources

1. Stockfish 19 release announcement: https://stockfishchess.org/blog/2026/stockfish-19/
2. Official NNUE documentation: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html

## Evidence captured

The Stockfish 19 release announcement states that SFNNv16 removes redundant threat features, adds new pawn-pair features, retires the secondary neural network introduced in Stockfish 16.1, and uses Quantization-Aware Training (QAT) over hundreds of billions of training positions consistently rescored with a strong Leela net. It also says universal binaries select CPU features automatically and that strict position validation was added.

The official NNUE documentation describes sparse feature inputs, a feature transformer/accumulator, multiple perspectives, incremental accumulator updates, clipped-ReLU-style bounded activations, quantized integer inference, and the importance of minimizing the cost of the first layer. These are the lawful architectural principles to reproduce in Unchessed. Exact SFNNv16 weights and training data are not available in this repository, so the implementation must be an independent compatible prototype rather than a claim of exact network parity.

## Design target

The high-value reproducible target is a shared incremental feature state: bitboard changes update a small set of feature rows, the Elo detector consumes the same immutable snapshot metadata, and the eval bar reads a side-to-move-normalized score plus confidence without recomputing board scans. Performance work should prioritize cache locality, integer arithmetic, bounded feature updates, and no lock or allocation on the search hot path.
