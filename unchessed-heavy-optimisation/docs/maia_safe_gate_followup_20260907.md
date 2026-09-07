# Maia-safe gate follow-up

The official Maia-3 evidence established that strict resilient-only reason purity was fixed but false-confirmation safety was not: the matched 40-game official-UCI evaluation reported 12/20 accelerated confirmations (60%), while the simplified ONNX replication reported 14/20 (70%). The remaining problem was that the resilient accumulator itself was broad enough to promote on human-like noisy traces.

## Change tested

The release candidate retains the strict semantic rule that only `legacy_accelerated_resilient` can promote Full when `AcceleratedDetection=true`. It adds an independent engine-like consistency fingerprint to `accelerated_resilient()`:

| Guard | Requirement |
|---|---:|
| Resilient observations | At least two qualifying observations |
| Good-quality mass | At least 3.0 |
| Catastrophic-error mass | At most the existing bounded maximum |
| Clean streak | At least two consecutive moves with cp-loss at most 40 |
| Volatility | At most 460 Elo-scale points |

The purpose is not to reject all noisy strength. It prevents a broad accumulated score from being sufficient by itself: promotion needs a short, independent near-perfect run and a bounded volatility trace. The accumulator and telemetry remain available for diagnosis.

## Real validation

A fresh official Maia-3 5M UCI evaluation used the same engine, 60-second starting clock, zero increment, `SelfElo=OppoElo` bands of 1500 and 2000, fixed openings, real UCI subprocesses, `Adaptive=true`, `OwnBook=false`, `UCI_Opponent=- - human UnknownOpponent`, and a 32-ply cap. There were three games per arm per Elo band, for six games per arm. The run used the real `wtime`/`btime` clock protocol and produced zero low-time skips.

| Arm | Games | Full confirmations | Mean first-Full ply | Range | Reason purity |
|---|---:|---:|---:|---:|---|
| Standard legacy | 6 | 5/6 | 8.6 | 6--11 | Legacy clock |
| Calibrated accelerated | 6 | 1/6 | 22 | 22--22 | 4/4 reason events resilient |

The calibrated probe therefore reduced the accelerated confirmation rate to 1/6 in this small official-Maia sample, but it must not be substituted for the earlier 20-game estimate. As a sensitivity control, three fresh Stockfish 19 games per arm under the same 60-second/32-ply protocol produced 1/3 standard confirmations at ply 30 and 2/3 calibrated accelerated confirmations at plies 24 and 28; all 15 observations per game were available and there were zero low-time skips. The appropriate conclusion is that the new gate is a promising safety improvement supported by targeted real runs, while retaining some strong-engine sensitivity; it is not a universal false-positive guarantee. The 20-game official-Maia result remains the larger baseline, and a future release decision should repeat the calibrated gate at the same 20-games-per-arm scale.

## Reproduction command

```bash
. "$HOME/.cargo/env"
cargo build -p unchessed-adapter --release
python3 unchessed-heavy-optimisation/tools/maia3_uci_false_positive_test.py \
  target/release/unchessed-adapter \
  --maia-cmd '/home/ubuntu/engines/maia3-5m/maia3-5m-uci.sh' \
  --games-per-arm-per-band 3 \
  --elo-bands 1500,2000 \
  --clock-ms 60000 \
  --inc-ms 0 \
  --max-plies 32 \
  --out unchessed-heavy-optimisation/results/unarchitectured-metal-maia3-calibrated-v2/results.json
python3 unchessed-heavy-optimisation/tools/analyse_maia3_calibrated_results.py \
  unchessed-heavy-optimisation/results/unarchitectured-metal-maia3-calibrated-v2/results.json
```

The result JSON, analysis text, and harness are retained with the Unarchitectured Metal series. The release build and detector tests must pass before interpreting the run.
