# 12 — Tier 2.2 bounded CPU calibration prototype

**Investigation ID:** `tier2.2-calibration`  
**Scope:** The acknowledged, bounded CPU-only Tier 2.2 plan from the extended research brief.  
**Status:** Implemented as an offline toy-plumbing diagnostic; no chess model, engine default, cloud job, or game-facing behavior was changed.

## Decision boundary

The Tier 1 self-play review recommends deferring a full AlphaZero-style loop because the repository has a scalar `HalfKAv2_hm` evaluator, no verified MCTS/PUCT implementation, no policy/value ABI, no replay pipeline, and no measured batched NNUE throughput. This Tier 2 step therefore tests only the cheapest prerequisites: legal-action masking, deterministic trajectories, replay serialization, terminal outcomes, and whether a tiny value-only learner can improve a held-out loss.

The prototype in `tools/rl_calibration.py` is intentionally a **toy transition system, not chess**. It does not load `unchessed-nnue.bin`, does not modify the Rust engine, and does not produce a playing-strength or Elo result. Using a toy system is a deliberate calibration of plumbing rather than an attempt to disguise unavailable chess training assets as a result.

## What was implemented

The tool generates a bounded number of deterministic toy episodes with explicit legal-action tuples. Every chosen action is checked against the legal mask. It writes replay records containing episode, step, state, side, legal actions, action, and outcome fields. A deterministic one-hot value learner trains and evaluates on alternating replay records, keeping the tiny train/holdout split balanced across reachable states. The output records a canonical replay SHA-256, legal-mask violations, replay count, and held-out loss before and after updates.

The default run is 100 games, 25 value updates, and learning rate 0.05. It is CPU-only and uses only the Python standard library. The report explicitly marks `not_chess` and `no_elo_claim` in its machine-readable output.

## Verification performed

| Check | Result |
|---|---|
| Same seed and game count produce identical replay records | Passed: `test_replay_is_legal_and_reproducible` |
| Legal-action mask violations | Passed: 0 violations |
| Held-out value loss improves | Passed: `0.22666666666666666` to `4.4702117962524e-32` |
| Replay report is machine-readable and content-addressed | Passed: CLI test; replay SHA-256 `c6dfbf10d7896c95b898ee406426917747e7907976953df9b4fb186cf8d6a1e1` |
| Invalid bounds are fail-closed | Passed: CLI exits 2 for fewer than two games |
| Rust toolchain | `rustc 1.98.0`, `cargo 1.98.0` available in the sandbox |

The exact test command and measured output are recorded after execution below. No CUDA or PyTorch installation was attempted. No real chess self-play, policy-head training, NNUE update, match, SPRT, or cloud spend was attempted.

The focused calibration suite completed **4 passed**. Relevant repository regression tests completed **28 passed** for policy-prior/pretrain/king-bucket coverage. The Rust bracket checker reported all **21 files balanced**, and `cargo test -p unchessed-core --release` completed **123 passed, 0 failed, 6 ignored**.

## Interpretation and recommendation

A passing result means only that the smallest deterministic replay/value-learning plumbing can be exercised on a toy domain. It does not establish that the existing NNUE can support AlphaZero policy/value training, that a chess MCTS is affordable, or that self-play would improve playing strength. A failure would have been a stop/defer result and would not have justified buying compute.

## Fresh rerun

On 2026-09-03, the approved bounded command was rerun with the upgraded sandbox toolchain:

```text
python3 tools/rl_calibration.py --seed 17 --games 100 --updates 25 --learning-rate 0.05 --json /tmp/rl-calibration-tier2-rerun.json
```

The run completed with **100/100 games**, **451 replay records**, **0 legal-mask violations**, and replay SHA-256 `c6dfbf10d7896c95b898ee406426917747e7907976953df9b4fb186cf8d6a1e1`. Held-out loss decreased from `0.22666666666666666` to `4.4702117962524e-32`. The focused test command `python3 -m pytest tools/test_rl_calibration.py -q` passed **4 tests**. The release workspace regression command `. "$HOME/.cargo/env" && cargo test --workspace --release` passed **123 tests, 0 failed, 6 ignored**. These are repeatability and plumbing results only; they are not chess, NNUE, self-play, or Elo evidence.

If the bounded tests pass, the recommended next step remains **defer full RL** until a real chess legal-state prototype, NNUE throughput benchmark, policy/value architecture, and explicit Tier 3 approval exist. The missing policy head and MCTS are architectural blockers, not problems this toy test resolves. Any future chess candidate or default integration still requires the repository's real paired-game SPRT gate.

## Budget and scope compliance

This work stayed within the acknowledged estimate of approximately 20–60 credits, used no cloud dollars, and did not enter Tier 3. It was performed on the renamed `manus/research-facilities` branch. The report is a calibration artifact, not permission to begin a full RL pipeline.
