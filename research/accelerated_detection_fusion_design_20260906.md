# AcceleratedDetection robust-fusion design

## Motivation

The original default-off `AcceleratedDetection` path used a hard conjunction of estimated Elo, confidence width, volatility, and one-step trend, followed by a short streak rule. That design improved the reported deterministic stress case but still treated all evidence windows similarly and exposed little information for calibration.

The revised path adds a second, bounded evidence layer while preserving the existing legacy ceiling and the original accelerated streak path. It is inspired by sequential change detection methods such as one-sided CUSUM and Bayesian online change-point reasoning, but it is implemented directly in Rust with no runtime dependency. The relevant background methods are described by Adams and MacKay’s Bayesian Online Changepoint Detection paper and standard CUSUM/SPRT literature: https://arxiv.org/abs/0710.3742 and https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/SPRT-and-CUSUM-in-HIDDEN-MARKOV-MODELS/10.1214/aos/1056562468.pdf.

## Homemade mechanism: concordant sequential evidence fusion

For every post-opening observation, four bounded signals are computed: rating strength, confidence tightness, low volatility, and non-negative trend. Their combination uses a harmonic mean rather than an arithmetic mean. This is the central safeguard: one weak or unstable channel materially suppresses the combined evidence instead of being hidden by three strong channels. A small clock-support term is blended into the result when instant strong replies are independently observed.

The combined evidence is accumulated by a leaky, signed CUSUM-style score. Sustained agreement raises the score, stale evidence decays, and clearly weak or highly volatile observations subtract evidence. The score is clamped to a finite range, so it cannot accumulate without bound over a long game. Confirmation requires the post-opening sample minimum, a minimum score, and a minimum current agreement value. The existing legacy ceiling remains unchanged and is still evaluated independently.

This design is intended to improve uniformity across engines that differ in move-time behavior. A fast engine can contribute clock support, while a slow engine can still confirm through sustained quality, confidence, stability, and trend agreement. Conversely, a human or sandbagger with isolated strong moves can be held back by the harmonic-mean veto and volatility penalty.

## Diagnostics

Observation telemetry now exposes `accelerated_score_milli`, `accelerated_evidence_milli`, and `accelerated_streak`. These fields are optional to the parser so older schema-v1 captures remain valid and default the new diagnostics to zero. The new suspect reason is `legacy_accelerated_fusion`.

## Validation

The focused opponent-model suite passes 23 tests, including deterministic confirmation at observation 29 in the existing 0.4 difficulty-weight stress case, rejection of a mixed-quality erratic sequence, preservation of default-off behavior, and preservation of the original accelerated path. The complete `unchessed-core` library suite passes 128 tests with six ignored tests. The telemetry parser suite passes five tests, including backward compatibility for legacy records. The Rust repository-wide formatter still reports pre-existing formatting drift in unrelated files; no repository-wide formatting rewrite was applied.

These are implementation and regression results, not a claim of real-game detection accuracy. The next measurement should use the asymmetric real-game telemetry protocol from `research/accelerated_detection_real_sprt_20260906.md`, comparing moves-to-Full and false-positive rates against distinct strong, Maia-like, and human-like opponents.
