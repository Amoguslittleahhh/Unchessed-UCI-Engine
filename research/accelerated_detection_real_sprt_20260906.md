# AcceleratedDetection: real paired-game SPRT follow-up

## Scope

This note records the follow-up to the opponent-detection-latency paper in `ieee-paper/opponent_detection_latency.tex`. The feature was implemented as a default-off UCI option named `AcceleratedDetection` in commit [`b5e5b32`](https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/commit/b5e5b32). The real paired-game experiment was committed in [`dfc6f86`](https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine/commit/dfc6f86).

## Implementation

The option adds an independent confirmation path while leaving the existing legacy weight/mean ceiling unchanged. It requires at least eight observations, estimated mean at least 2550 Elo, confidence width at most 220, volatility at most 300, and a non-negative single-step trend. It confirms after two consecutive qualifying observations or one qualifying observation combined with a live clock tell. A single qualifying move is insufficient. The detector reports `SuspectReason::LegacyAcceleratedCeiling`.

An isolation check using a 0.4 difficulty weight per observation reached accelerated confirmation at observation 21 versus observation 30 for the legacy ceiling. That check is synthetic and is included only as an implementation sanity check; it is not treated as the real-game result.

## Real SPRT

The real match used cutechess-cli with the same main binary on both sides, the same explicit NNUE, `Adaptive=true`, `tc=5+0.05`, the same opening source, and 1,000 games. The baseline condition set `AcceleratedDetection=false`; the treatment set it to `true`. The SPRT used `elo0=0`, `elo1=5`, `alpha=beta=0.05`, giving continuation bounds of approximately `[-2.94, 2.94]`.

The match reached the 1,000-game cap without crossing an SPRT boundary. The final aggregate was 296 baseline wins, 307 accelerated wins, and 397 draws. The reported Elo difference was `-3.8 +/- 16.7`, LOS was `32.7%`, draw ratio was `39.7%`, and the final LLR was `-0.434`. The run reported zero illegal moves and zero crashes for both conditions.

## Interpretation

This mirror match establishes operational safety and provides no evidence of a material strength cost in this configuration. It does not directly measure the feature's target metric, moves until Full-mode confirmation, because both sides are the same strong engine and can use ordinary engine-detection evidence to identify one another. The experiment therefore cannot confirm or reject a latency benefit.

## Required next experiment

Run an asymmetric test against a distinct strong opponent. Capture raw telemetry for each condition, including move index, `mode_before`, `mode_after`, first qualifying observation, detector reason, estimated Elo, confidence, volatility, suspicion, and clock tell. Compare paired moves-to-Full distributions with the option off and on. Report false Full entries and no-confirmation rates against strong human-like and Maia-like opponents, along with clock-tell contribution and search-overhead changes. The primary endpoint is detection latency, not final game score.

## Source artifacts

The committed experiment script is `scripts/research/wsl_sprt_accelerated_detection.sh` on the main-branch result commit. The full log and PGN are under `wsl-workspace/results/adapter/sprt_gates/` in commit `dfc6f86`.
