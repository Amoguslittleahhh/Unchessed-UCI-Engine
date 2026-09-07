# Reproducing the resilient detection experiments

This document describes how to rebuild the released `AcceleratedDetection` implementation and reproduce the real asymmetric experiments. The protocol is intentionally explicit about engine versions, clock handling, detector options, telemetry, artifacts, and interpretation. It does not use simulated observations.

## 1. Source and version pinning

Use the `manus/research-facilities` branch at or after commit `99aad93`. The detector change is in `unchessed-core/src/adapt.rs`; the series-scoped driver is `unchessed-heavy-optimisation/tools/run_unarchitectured_metal_asymmetric_latency.py`.

Use the official Stockfish 19 Linux universal binary for all Stockfish controls. Do not substitute the distribution package, because package versions may lag. Record the SHA-256 digest of the downloaded binary in the experiment manifest. For a Maia-style opponent, use the official current Maia-3 package and its 5M UCI runtime where possible. The earlier matched 40-game replication used the `mcognetta/simple-maia3-inference` ONNX export with `onnxruntime`, temperature-1 policy sampling, and fixed random seeds; it is a separate runtime condition and must not be merged with official Maia-3 UCI results.

A minimal Linux environment is:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config python3 python3-pip python3-venv \
  pdflatex git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
sudo pip3 install python-chess onnxruntime
```

Clone the repository and select the editing branch without changing `main`:

```bash
git clone https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine.git
cd Unchessed-UCI-Engine
git checkout manus/research-facilities
git checkout 99aad93
```

Build the release adapter:

```bash
. "$HOME/.cargo/env"
cargo build -p unchessed-adapter --release
```

## 2. Real Stockfish 19 asymmetric latency test

The driver launches two real UCI subprocesses: the Unchessed adapter and Stockfish. The same six fixed opening prefixes are used in both arms. The standard arm sets `AcceleratedDetection=false`; the accelerated arm sets it to `true`. Both arms use one thread, 64 MiB hash, `Adaptive=true`, `OwnBook=false`, `AdapterTelemetry=true`, `EngineDetectV2=false`, and `UCI_Opponent=- - human UnknownOpponent`. The final setting prevents the known-engine table from deciding the result before move-quality evidence is observed.

The clock-controlled command is:

```bash
python3 unchessed-heavy-optimisation/tools/run_unarchitectured_metal_asymmetric_latency.py \
  --adapter target/release/unchessed-adapter \
  --stockfish /absolute/path/to/stockfish-19 \
  --metal-file unchessed-heavy-optimisation/artifacts/unarchitectured-metal-final.unmetal \
  --output unchessed-heavy-optimisation/results/stockfish19-reproduction \
  --games-per-arm 20 \
  --max-plies 48 \
  --clock-ms 60000
```

The harness refuses clocks below 10,000 ms because the adapter suppresses observations below that floor. This avoids incorrectly interpreting zero observation coverage as a detector failure. Each game sends real `wtime` and `btime` values to both engines, records the move history, stores complete subprocess logs, and counts `opponent_observation` and `observation_skipped` telemetry events.

Analyze the output using:

```bash
python3 unchessed-heavy-optimisation/tools/analyse_unarchitectured_metal_asymmetric_latency.py \
  --root unchessed-heavy-optimisation/results/stockfish19-reproduction \
  > unchessed-heavy-optimisation/results/stockfish19-reproduction/analysis.json
```

The primary latency metric is the first telemetry event with `mode_after=FULL`, measured in plies from the UCI game start. Report the mean, median, range, and per-game values separately for each arm. Also report the first `suspect_reason`, observation count, low-time skip count, and whether every game reached the same ply horizon.

## 3. Official Maia-3 UCI false-positive test

Install and run the official Maia-3 runtime according to its upstream repository instructions. Use the same `--clock-ms 60000`, 48-ply cap, fixed opening prefixes, one thread, and telemetry settings as the Stockfish experiment. Replace the `--stockfish` argument with the Maia-3 UCI launcher only after verifying that it responds to `uci`, `isready`, and `bestmove`.

Run at least 20 games per arm for a stable estimate. Record the Maia model name, model checksum, inference backend, CPU configuration, random seed policy, temperature, and whether opponent-response ranking is enabled. Do not combine these results with the simplified ONNX pipeline.

The safety metric is the accelerated-arm false-confirmation rate: the proportion of games in which `mode_after=FULL` occurs against Maia. The semantic metric is reason purity: every accelerated confirmation must have `suspect_reason=legacy_accelerated_resilient`. A clock or stable-fusion reason in accelerated mode is a release-blocking regression.

## 4. Matched simplified-ONNX replication

The earlier 40-game apples-to-apples comparison used `tools/maia3_false_positive_test.py` from the main-branch research record. It drove `mcognetta/simple-maia3-inference` directly with `onnxruntime`, temperature-1 policy sampling, two bands (`elo_self=elo_oppo=1500` and `2000`), 10 games per arm per band, alternating colors, fixed openings, 60,000 ms clocks, zero increment, and a 48-ply cap. The adapter used `Adaptive=true`, `OwnBook=false`, NNUE evaluation, and `UCI_Opponent=- - human UnknownOpponent`.

Reproduce this condition only with the exact same model export, seeds, opening list, and harness. The ONNX export does not include the official Maia-3 one-ply opponent-response ranking, so it is not interchangeable with official Maia-3 UCI. Store the raw result JSON, the harness revision, model checksum, and all runtime parameters together.

The matched 40-game results were:

| Arm | Games | Full confirmations | Mean confirmation ply | Reason purity |
|---|---:|---:|---:|---|
| Standard legacy | 20 | 20/20 | 10.0 | Legacy clock, expected |
| Prior clock-corroboration accelerated | 20 | 13/20 | 24.6 | Mixed clock/fusion paths |
| Strict resilient-only accelerated | 20 | 14/20 | 25.7 | 14/14 resilient-only |

The strict semantic fix therefore succeeded—no accelerated confirmation bypassed the resilient channel—but this matched ONNX sample did not lower the false-confirmation rate. The correct release conclusion is that reason purity is fixed, while model-pipeline-specific false-positive calibration remains unresolved.

## 5. Observation-probe saturation throttle
The long-game efficiency fix is intentionally part of the same `unarchitectured-metal` series. The opponent-quality probe performs a real search before the engine's own move search, so its budget is adaptive. While the detector is still forming a verdict it uses depth 14 and 400,000 nodes. Once `engine_suspect()` has remained true for ten completed observations, `OpponentModel::observation_saturated()` switches the next probe to the calibrated pre-bump depth 9 and 60,000-node budget, and suppresses the depth-12/250,000-node fallback when the played move is absent from the shallow PV. The saturation check occurs before the current observation is applied, so a fresh clock tell cannot retroactively make the sample that created it cheap. A subsequent non-verdict resets the streak and restores the high-fidelity budget.

This is a homemade evidence-economics breakthrough: precision is spent while it can still change `Mode::Full`, then the system degrades gracefully after the verdict has held, preserving a live Elo update without repeatedly searching and discarding the same information. Telemetry records `suspect_streak`, `observation_saturated`, and source profiles `probe_high_fidelity`/`probe_saturated`, making the throttle directly auditable. In the final telemetry-enabled four-game Stockfish 19 probe on fixed openings and 60-second clocks, every game produced 23 observations and zero low-time skips; standard first-Full mean was 30 plies, accelerated/fusion mean was 25 plies, and both accelerated confirmations used `legacy_accelerated_resilient`. One accelerated game recorded three saturated observations, demonstrating engagement without a false or low-time skip. This is an efficiency/safety smoke test rather than a strength claim. Larger matched runs should additionally capture engine-search depth or node totals.

## 6. Unit, parser, and release checks

Run the complete Rust library tests and telemetry parser tests:

```bash
. "$HOME/.cargo/env"
cargo test -p unchessed-core --lib
python3 -m unittest tools.test_analyse_adapter_telemetry -q
cargo build -p unchessed-adapter --release
```

The expected released baseline is at least 135 passing workspace tests, including the saturation regressions, and 6 passing telemetry-parser tests. The smoke test should confirm that the adapter responds with `uciok`, `readyok`, and exposes `AcceleratedDetection` defaulting to false.

## 6. Reproducibility and interpretation safeguards

Never report a latency result without observation coverage. A game with zero observations is a protocol failure, not evidence for or against the detector. Never infer false-positive safety from final game score; the detector’s target metric is first Full-mode confirmation. Use paired opening prefixes, identical engine settings, and alternating colors. Keep raw logs outside the commit if they contain protocol noise, but commit JSON summaries, exact commands, model checksums, environment versions, and the paper source.

The calibrated release candidate adds one more promotion guard on top of strict resilient-only semantics: at least two consecutive cp-loss-at-most-40 moves and volatility at most 460 Elo-scale points. This prevents the resilient accumulator from promoting solely on broad noisy evidence while retaining the accumulator for diagnostics. The observation-probe throttle is independent of promotion semantics: it changes only the cost of later opponent-quality measurements after a held verdict, never the definition of `engine_suspect()` or the first confirmation. A fresh official Maia-3 probe at three games per arm per Elo band produced 1/6 accelerated confirmations, 4/4 reason events using `legacy_accelerated_resilient`, and zero low-time skips; this is encouraging but smaller than the earlier 20-game sample. The strict policy is therefore ready to ship as a semantic and targeted safety improvement, not as a universal Maia false-positive guarantee. Future promotion-grade work should repeat the calibrated gate on the official Maia-3 UCI runtime at least 20 games per arm and retain the same telemetry and coverage criteria.
