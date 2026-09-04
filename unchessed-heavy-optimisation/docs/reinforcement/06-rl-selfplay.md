# 06 — Reinforcement-learning self-play viability

**Investigation ID:** `01-rl-selfplay`  
**Repository / line:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/rustc-bootstrap-trial`  
**Scope:** One Tier 1 item only: whether a minimal AlphaZero-style self-play loop is viable for the approximately 5.7M-parameter `HalfKAv2_hm` NNUE, compared with the repository's Stockfish-style supervised NNUE route.  
**Disposition:** Design and feasibility review. No production code, model, training run, self-play match, cloud job, commit, or push was performed.

## Executive decision

**Defer the AlphaZero-style loop; do not drop the research question.** It is technically possible, but it is not a minimal change to this engine and it is not currently an economical path to a stronger NNUE. The existing network is a scalar evaluator designed to sit inside alpha-beta search. AlphaZero instead requires a policy/value predictor, legal-action encoding and masking, MCTS/PUCT, replay storage, batched inference, recurrent promotion/evaluation, and repeated candidate-versus-parent gates. The repository has no verified policy network, no MCTS implementation, no self-play/replay pipeline, and no measured NNUE batch-inference throughput.

The smallest responsible next step is an **offline diagnostic**, not a strength campaign: implement no default behavior, use a tiny new policy/value head or a scalar-only uniform-prior prototype, and demonstrate that one complete game can produce legal masked actions, visit-count targets, terminal outcomes, replay records, and a deterministic CPU update. Stop if the loop cannot reach a declared throughput target or if the learned policy does not beat its uniform-prior baseline on held-out positions. Only after that gate, and only with a measured fast batched evaluator, should a small real self-play run be considered.

## What the repository establishes

The six preceding reinforcement reviews are unusually explicit about evidence boundaries. `00-synthesis.md` requires provenance before scale, a real paired-game SPRT for game-reachable behavior, and rejects MAE, fixed-position probes, simulations, and interrupted matches as strength evidence. `01`/`02` document that the NNUE relabel/retrain path is blocked on complete source shards, full state provenance, labels, and a declared teacher. `03` keeps persona diagnostics offline and opt-in. `04` records that the shipped sidecar is tied to a small GAB and that quantization changes require retraining. `05` is blocked on a missing oracle checkpoint and explicitly says not to substitute the student or NNUE artifacts.

The implementation and historical results add the following concrete facts.

| Fact | Repository evidence and interpretation |
|---|---|
| Network size | `docs/full-scale-bug-audit-2026-08-21.md` records **5,767,937 parameters**. The checked-in `unchessed-nnue.bin` is **23,071,768 bytes**; this is a scalar NNUE evaluator, not a policy/value AlphaZero model. |
| Features | `docs/nnue-architecture-audit.md` verifies byte-identical `HalfKAv2_hm` king buckets against Stockfish. The inference path is deliberately f32, unlike Stockfish's int16/int8/int32 path. |
| Existing trainer | `tools/train_nnue.py` trains a supervised position score: its documented target mixes sigmoid(cp/400) and WDL, and its v4 head is eight piece-count output buckets. It exports the best validation-MAE checkpoint. This is not an outcome-trained policy/value loop. |
| Existing labels and scale | `docs/nnue-round-12-results.md` reports a committed corpus of about **95k games / ~1M candidate positions**, and says no net was retrained or SPRT'd in that round. `docs/nnue-v4-training-recipe.md` describes the production family as 108M original records, with 15-epoch cap, Adam at 1e-3, batch 65,536 on CPU or 131,072 on A100, and early stopping. |
| Existing search | `unchessed-datagen/src/main.rs` labels with a **5,000-node HCE search**. The engine exposes alpha-beta-oriented UCI controls (Hash, Threads, MultiPV, pruning margins, aspiration, ProbCut, etc.); there is no repository MCTS/PUCT implementation found by source search. |
| Existing neural policy | `docs/policy-prior-calibration.md` says a PUCT implementation would need a network evaluation at every expanded node and records a **9.72 ms** forward-pass measurement for the separate Unarchitectured model. It warns that a few hundred nodes can consume a move budget and that batching tree evaluations is not currently solved. This is a useful blocker signal, **not** an NNUE throughput measurement. |
| Available verification | On this host, `python3 tools/test_king_buckets.py` ran **6/6 tests successfully**. The existing release adapter answered UCI and advertised `PolicyFile`, but then reported “no policy net found — using heuristic move priors”; it also reported no NNUE file in that invocation's working directory. No inference benchmark was run. |

## What AlphaZero would actually require here

The AlphaZero paper describes a neural network that supplies both a move policy and a position value to MCTS. Its supplementary material states that illegal moves are masked and priors renormalized, each chess MCTS used **800 simulations**, and chess training used **44 million games**, **700k mini-batches**, and **9 hours** on the reported system; each MCTS ran on a machine with four TPUs. The paper reports roughly **80k positions/second** for AlphaZero chess versus **70,000k positions/second** for Stockfish in its evaluation-speed table. These numbers are historical reference points, not a promise that this repository can reproduce them.[1]

A `HalfKAv2_hm` scalar NNUE can provide a value-like score, but it cannot provide a calibrated prior over legal moves. A minimal design therefore has three choices:

1. **Uniform-prior scalar prototype.** Run PUCT with uniform priors, evaluate leaves with the NNUE, and train only the existing scalar/value pathway from terminal outcomes. This tests plumbing but is a weak AlphaZero analogue and conflates search with evaluation.
2. **Small policy head beside the NNUE.** Reuse the feature transformer/accumulator, add a policy head over a fixed move encoding, mask illegal moves, and train policy targets from visit counts plus a value head from game outcomes. This is the smallest credible AlphaZero-style design, but it changes the model ABI and requires a new trainer/exporter/runtime path. The current v4 file cannot simply be loaded as this model.
3. **Keep alpha-beta and do outcome fine-tuning.** Generate games with the existing engine, train the scalar net on outcomes, and omit MCTS/policy. This is self-play reinforcement learning in a broad sense, but not AlphaZero-style and is vulnerable to self-confirmation and search-label leakage. It should be compared as a baseline, not called AlphaZero.

The recommended diagnostic uses option 1 first, then option 2 only if the plumbing gate passes. It must retain the current NNUE as a frozen evaluator and write a new experimental format outside the shipped `unchessed-nnue.bin` path.

## Honest scale and compute estimate

The following are planning estimates, not measurements. They intentionally use ranges because this repository has no NNUE batch benchmark and no MCTS implementation.

Assume a short diagnostic game averages **80 plies**, **64 simulations per move**, and one fresh leaf evaluation per simulation. One thousand games then yield about **80,000 replay positions** and **5.12 million leaf evaluations**. At the separately measured 9.72 ms forward time, a naive serial implementation would take about **49,800 seconds (13.8 hours)** merely for those leaf evaluations. That timing belongs to Unarchitectured and must not be substituted for an NNUE number; it demonstrates why a local benchmark and batching decision are prerequisites. If a future NNUE benchmark measured 1,000, 10,000, or 100,000 evaluations/second, the same 5.12M evaluations would take about 85, 8.5, or 0.85 minutes of evaluator time, respectively, before chess move generation, tree management, synchronization, training, and I/O. Real throughput could be much worse at batch 1 because tree search supplies irregular batches.

For a more AlphaZero-like **800 simulations**, the same 1,000-game diagnostic would require **64 million leaf evaluations**. At 100,000 evaluations/second that is about 10.7 minutes of raw evaluator time; at 10,000/second it is about 1.8 hours; at 1,000/second it is about 17.8 hours. These are lower bounds and do not establish useful playing strength. A serious run of 100,000 games at 80 plies and 800 simulations would imply **6.4 billion leaf evaluations**, already a multi-day to multi-month workload depending on the unmeasured evaluator/search throughput.

Memory is not the primary blocker for 5.7M parameters. The existing 23 MB weights fit in ordinary RAM, and a replay buffer of 80,000 positions with compact board/state, policy target, and outcome metadata could plausibly fit in hundreds of MB to a few GB. The hard problems are search/evaluation throughput, adding policy capacity, data freshness, and proving improvement. Stockfish-style supervised training is materially cheaper in search: it labels positions once with a declared teacher search and trains large batches using the existing sparse-data machinery. The repository already has an approximately million-position diagnostic corpus and a documented 108M-record production recipe, whereas RL must repeatedly regenerate correlated games and search targets.

For comparison, Stockfish's own official description says NNUE is trained on evaluations of **millions of positions at moderate search depth**, exploits incrementally updated CPU evaluation, and is used as an evaluation inside alpha-beta search.[2] The official `nnue-pytorch` project documents a fast C++ data loader that produces whole sparse batches asynchronously because one-sample-at-a-time input is not viable, and its README points to a substantial Docker/toolchain setup (roughly 30–60 GB for the container). That is evidence for the supervised route's mature data path, not evidence that this repository can run Stockfish's trainer unchanged.[3]

## Smallest diagnostic experiment

This is the smallest experiment that answers viability without pretending to answer strength.

### Phase A — throughput and interface gate

Create an offline prototype (not a UCI default) with a deterministic fixed seed and these limits: one CPU thread, one position, one fixed legal-move encoder, **16 MCTS simulations**, and a maximum of **20 plies**. Benchmark the existing NNUE at batch sizes 1, 8, 32, and 128 on 1,000 fixed legal positions, reporting warm-up, median, p95, positions/second, and peak RSS. Do not use the 9.72 ms Unarchitectured timing as a substitute. Also measure move generation and one MCTS traversal with a stub value function, so evaluator and tree overhead are separated.

The gate passes only if: legal masks never select an illegal move; terminal and repetition/no-progress handling is deterministic; the same seed reproduces the same trajectory; a complete game writes replay records with board state, side to move, legal-action mask, visit target, and terminal result; and the measured evaluator is fast enough to run the bounded test in **under one hour on the declared host**. A failure is a defer/stop result, not a reason to buy cloud compute.

### Phase B — learning sanity gate

Run **100 games**, alternating colors, from a small fixed opening set, using 16 simulations and the frozen scalar NNUE with uniform priors. Verify replay schema, value-target sign conventions, and that a tiny value-only update lowers loss on a held-out slice without NaNs. Then, only if Phase A is clean, attach a randomly initialized policy head and run **1,000 games / 64 simulations** with a replay cap of 80,000 positions. Compare against (a) uniform-prior frozen NNUE and (b) the untrained policy-head checkpoint using the same openings, seed policy, simulation count, and time limit.

Report only plumbing and learning diagnostics: games completed, plies/game, decisive/draw rate, legal-mask violations, replay size, evaluator throughput, policy cross-entropy, value loss, visit entropy, and held-out policy/value metrics. Do not report Elo from this sample. A positive learning sanity result authorizes design work on batching and a real candidate gate; it does not authorize replacing the NNUE.

### Required artifacts and stop conditions

Freeze hashes for source revision, net, opening list, encoder, config, and replay shards. Stop if any illegal action is emitted, if replay cannot be deterministically regenerated, if throughput misses the one-hour bounded-test limit, or if the learned head fails to beat its untrained/uniform baseline after a predeclared number of updates. Do not install CUDA/PyTorch stacks or rent cloud resources for Phase A/B; the repository's prior reviews already identify missing assets and expensive training as blockers.

## Supervised NNUE versus self-play RL

| Dimension | Existing Stockfish-style supervised NNUE | AlphaZero-style self-play proposal |
|---|---|---|
| Target | Teacher/search evaluation plus WDL mixture; current trainer exports best validation-MAE checkpoint | Terminal outcome for value and MCTS visit counts for policy |
| Search cost | One declared label search per sampled position; existing datagen uses 5,000-node HCE labels | Search at every move, typically tens to hundreds of leaf evaluations per move; canonical AlphaZero used 800 simulations |
| Model | Existing `HalfKAv2_hm` scalar v4 ABI, 5,767,937 params | Requires policy/value outputs and a new experimental ABI; scalar net alone lacks priors |
| Data | Existing shard format, sparse whole-batch loader, approximately 1M diagnostic positions and documented 108M production corpus | Correlated, continually regenerated games; replay freshness and policy-target storage are new |
| Compute risk | High but bounded and already instrumented; prior docs still gate complete shards, teacher provenance, and SPRT | Dominated by repeated self-play search and irregular NN evaluation; no local throughput number exists |
| Strength evidence | Candidate net still needs paired SPRT; validation MAE is not strength | Same SPRT requirement, plus parent/candidate gating and protection against self-play collapse |
| Near-term decision | **Pursue when missing corpus/state/teacher assets are restored** | **Defer pending Phase A/B viability evidence** |

The supervised path is not automatically good: prior repository evidence records a historical 108M-label result that was **47.8 cp / −155.6 Elo** and rejects spending on a larger 178M run under the old recipe. That finding does not prove RL would win, but it does show why neither more data nor an attractive training curve is sufficient. Any new supervised or RL candidate needs its own real match gate.

## Verification ledger

### Ran in this investigation

| Check | Result |
|---|---|
| Branch/status | Confirmed branch `manus/rustc-bootstrap-trial`; worktree was clean before this report was created. |
| Lightweight feature test | `python3 tools/test_king_buckets.py`: **6/6 passed**. |
| UCI smoke | Existing `target/release/unchessed-adapter` answered `uci`; it advertised `PolicyFile`, then reported no policy net and heuristic priors when launched from the tested directory. |
| Asset inspection | `unchessed-nnue.bin` and `nnue-shards-safe/unchessed-nnue-v4-overtrained.bin` each measured **23,071,768 bytes**. |
| Source/document review | Read `docs/reinforcement/00-synthesis.md` through `05-oracle.md`, relevant NNUE, policy-prior, training-recipe, audit, datagen, and UCI sources. |
| Literature review | Read the Science AlphaZero paper/supplement and official Stockfish NNUE documentation/repository pages listed below. |

### Design-only or not run

No MCTS/PUCT code was written or executed. No NNUE inference throughput benchmark, policy head, self-play game, replay generation, gradient update, SPRT, full `cargo test`, CUDA installation, cloud run, or strength claim was made. The AlphaZero game/simulation numbers are from the paper, while all proposed diagnostic limits and compute projections above are explicitly assumptions or arithmetic estimates. The 9.72 ms figure is a repository measurement for a separate Unarchitectured model and is included only as a cautionary upper-bound-style scenario, never as an NNUE result.

## Recommendation

**DEFER.** Pursue the bounded Phase A interface/throughput diagnostic only when it can be done offline with existing dependencies and no production integration. If Phase A and B pass, pursue a separate policy/value prototype and compare it against a supervised/outcome baseline. Do not spend cloud money, build a 44-million-game ambition, or alter the shipped NNUE based on this review. If the measured NNUE cannot sustain the bounded diagnostic, or if a policy/value head cannot beat the uniform baseline under frozen reproducibility conditions, **drop AlphaZero-style self-play for this project at the current scale** and return to the better-supported supervised NNUE route. In all cases, retain the repository's provenance and real paired-SPRT gates.

## References

[1]: https://doi.org/10.1126/science.aar6404 "Silver et al., A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play, Science (2018); supplementary Table S3 reports 800 simulations, 44M chess games, 700k mini-batches, and 9h/4 TPU-machine training."

[2]: https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/ "Stockfish, Introducing NNUE Evaluation (2020); millions of moderate-depth evaluated positions, incremental CPU evaluation, and alpha-beta integration."

[3]: https://github.com/official-stockfish/nnue-pytorch "Official Stockfish NNUE PyTorch trainer repository; whole-batch sparse data loading, training workflow, and environment requirements."

[4]: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html "Official Stockfish NNUE documentation; model specification, sparse batch structure, data-loader bottlenecks, and training implementation details."

[5]: https://arxiv.org/abs/1712.01815 "Public AlphaZero paper record and supplementary-material access point."

## Final status

This report completes exactly the Tier 1 RL self-play viability item. It does not start Tier 2/3 work, alter engine behavior, or claim that either self-play or supervised retraining is strong. The evidence supports a cautious **defer with a tiny diagnostic**, not an AlphaZero-scale build.
