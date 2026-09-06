# Independent replication and resilient-channel correction

## Evidence that motivated the fix

An independent replication used Stockfish 19 rather than Stockfish 16, with eight real games per arm, six fixed opening prefixes, a 60-ply cap, one thread, 64 MiB hash, adaptive mode, no own book, telemetry enabled, and a 1500 ms move budget. The reported means were 28.0 plies for standard detection and 27.75 plies for AcceleratedDetection, with medians of 28 and 28 and ranges of 26--31 and 26--29. This is consistent with the earlier conclusion: the stable fusion lane can be operationally safe and occasionally earlier, but there is no established uniform latency improvement on clean engine play.

The replication also identified a methodological trap. A first 100 ms attempt produced zero Full confirmations because every opponent observation was skipped by the adapter's low-time gate. The adapter intentionally stops expensive opponent probing when the side-to-move clock is below 10 seconds. A latency experiment therefore needs a real starting clock above that floor, must count observation coverage, and must fail fast when a requested clock is unsafe.

## Resilient-channel correction

The original fusion lane used volatility as a veto. That is appropriate for clean engine-like play but mis-targets the motivating case: a strong opponent whose move-quality trace is noisy or intermittently inconsistent. The improved implementation adds a second, default-off resilient channel. It retains rating evidence and confidence, but replaces the volatility veto with leaky good-quality mass, a catastrophe guard, and a two-observation streak. Moderate mistakes contribute noise rather than automatic rejection; repeated severe mistakes accumulate bad mass and block confirmation.

The resilient lane requires at least 10 observations, estimated mean at least 2450 Elo, score at least 0.55, evidence at least 0.48, two consecutive qualifying observations, good-quality mass at least 3.0, and bad mass no greater than 1.80. It emits `legacy_accelerated_resilient` and exposes score, evidence, and streak fields in telemetry. The legacy ceiling and stable fusion lane remain intact, and the option remains default-off.

## Validation

The new deterministic opponent-model regressions cover both a strong-but-noisy trace and a catastrophic erratic trace. The opponent-model subset passes 25 tests. The telemetry parser passes six tests, including resilient fields and backward compatibility for older schema-v1 captures.

A real clock-controlled run against the installed distinct Stockfish opponent used a 60-second starting clock, two standard games, and two fusion games. All four games produced 15 live observations, zero low-time skips, and Full confirmation. Standard first-Full plies were 30 and 30; resilient fusion first-Full plies were 26 and 24. The observed resilient reason triggered in both fusion games. This is a small validation run, not a universal claim.

The experiment driver now accepts `--clock-ms`, rejects values below 10,000 ms, records observation and low-time-skip counts, and supports live `wtime`/`btime` UCI games. A deliberate `--clock-ms 1000` invocation fails before starting a game, confirming the protocol guard.

## Remaining limitation

The resilient channel is a targeted correction, not proof of a solved detector. The next decisive test remains a human-like or Maia-style opponent with strong-but-inconsistent play, plus a weaker erratic control and real human games. Those datasets should report false positives, missed detections, observation coverage, and latency confidence intervals separately.
