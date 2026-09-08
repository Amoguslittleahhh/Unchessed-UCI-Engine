# Practical Training Procedure for an SFNNv16-Rivaling Original Evaluator

## Short real-world procedure

1. Provision an A100 80 GB or equivalent accelerator, plus the largest available host CPU and fast local NVMe storage. Use a PyTorch build matching the host CUDA or ROCm stack.
2. Generate or collect a legally usable, game-disjoint position corpus. Label every position through the separately built Stockfish reference using a fixed UCI protocol, fixed depth or node budget, and explicit terminal/mate encoding.
3. Split by game before shuffling into training, validation, and untouched test sets. Keep quiet, tactical, checking, capture, promotion, endgame, and terminal strata represented in every split.
4. Train with `scripts/train_nnue_portable.sh`, starting with a self-check and a short pilot. On an A100, begin with `BATCH_SIZE=131072`; on smaller devices reduce it until the run is stable.
5. Save content-addressed checkpoints, the environment manifest, and the exact reference binary checksum. Select only by untouched validation performance; never tune on the final test set.
6. Evaluate the best checkpoint on the untouched test set for score MAE, WDL, terminal classification, mate distance, sign accuracy, p95 latency, and search overhead.
7. Run a fixed-control match only after the accuracy and speed gates pass. Keep the new model opt-in until the match is statistically powered and reproducible.

## Efficient procedure

### 1. Freeze the target and reference

Build the exact Stockfish reference once and record the commit, executable SHA-256, network SHA-256, compiler, CPU flags, GPU driver, PyTorch version, and UCI command template. Use one target policy for ordinary positions and separate targets for terminal and mate-distance positions. Do not mix mate scores into ordinary regression without a declared encoding.

### 2. Construct the corpus without leakage

Replay legally sourced games with deterministic board validation. Partition by game identifier before position sampling. Keep a fixed untouched test set that is never used for architecture search, calibration, early stopping, or threshold selection. Sample strata deliberately: quiet middlegame, opening, endgame, checks, captures, promotions, king exposure, low material, checkmate, and stalemate. Store each shard in sequential binary form on local NVMe; avoid millions of tiny files.

For each position, store the canonical board representation, side to move, reference static score, reference WDL or calibrated win probability, terminal class, mate distance where proven, source game identifier, and label-manifest hash. Reject illegal, duplicate, malformed, out-of-range, or reference-timeout records rather than silently training on them.

### 3. Label efficiently

Use a producer/consumer pipeline. Run enough Stockfish workers to saturate the available CPU without causing context-switch collapse, and write large append-only shards. Keep the reference executable outside the Unchessed repository. Pin workers to physical cores where possible, reserve a few cores for storage and orchestration, and measure throughput rather than assuming that every logical thread helps. For an A100, pre-label on a large CPU host or in a separate phase so the GPU remains dedicated to training.

### 4. Train in stages

Run a pilot on 1--5 million positions to catch ABI, feature, target, and numerical errors. Then run the full model with mixed precision only where the backend is validated. Keep feature-transformer accumulation numerically stable; use gradient scaling and gradient clipping if needed. Use large batches to saturate the A100, but scale learning rate only after confirming optimizer stability. Maintain deterministic seeds for the validation run, and run a second seed for robustness.

The first credible higher-capacity architecture should remain original while borrowing only public principles: a sparse king-and-piece stream, threat and pawn-pair streams, an accumulator/transformer, bounded paired activations, and separate score/WDL/terminal/mate heads. Train the terminal heads with stratified weights rather than allowing the abundant quiet positions to dominate. Export a versioned model ABI and test the exported model against the training model position by position.

### 5. Use the hardware correctly

The portable launcher supports CUDA, ROCm's CUDA-compatible PyTorch interface, Apple MPS when explicitly enabled, and CPU fallback. For an A100 80 GB, use a large batch, pinned host memory if the data loader exposes it, persistent workers, local NVMe shards, and TF32 or AMP only after numerical checks. For a 180/360-core CPU, use all available threads for decoding and labeling, but benchmark physical-core versus SMT settings; maximum thread count is not always maximum throughput. For smaller GPUs, reduce batch size and use gradient accumulation only if it does not distort the optimizer schedule.

Maximize useful compute and bandwidth by keeping shards contiguous, prefetching the next batch, avoiding Python per-record loops, coalescing sparse feature indices, and measuring GPU utilization, host-to-device bandwidth, dataloader wait time, and cache misses. Do not overclock, disable thermal protections, or force unsafe power settings. Vendor power-limit and clock controls belong to the host scheduler and must stay within the machine's cooling, cloud-provider, and administrator policies.

### 6. Validate before promotion

Require all of the following on untouched data: overall score MAE at or below 100 cp; separate MAE for tactical and terminal strata; WDL calibration; terminal accuracy; mate-distance accuracy; sign accuracy; bounded worst-case error; and reproducible exported-model output. Measure p50/p95/p99 inference latency, nodes per second, cache behavior, and search overhead against the current evaluator. Reject promotion if overhead exceeds the declared 5% limit or if terminal errors regress.

Finally, run a fixed-control, balanced-color match with fixed openings, threads, hash, time control, seed, adjudication, and model files. Use a predeclared SPRT or confidence-interval protocol. Only after that result is complete should the model become the default search evaluator.

## Current best starting point

The supplied original `unchessed-nnue.bin` is a valid fallback and currently measures 148.683 cp MAE against the exact Stockfish reference on the 120-position diagnostic corpus. It is not Stockfish-trained and is not sufficient for the final 100 cp gate. The next high-value run is therefore a new Stockfish-UCI-labeled, leakage-safe corpus and a richer original multi-head model, not more post-hoc affine calibration.
