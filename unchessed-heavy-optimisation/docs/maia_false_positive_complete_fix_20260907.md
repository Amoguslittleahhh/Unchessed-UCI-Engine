# Complete Maia false-positive fix

## Problem reproduced

An independent real 40-game evaluation using the `mcognetta/simple-maia3-inference` ONNX export found that the earlier clock-corroboration fix reduced, but did not eliminate, false confirmations. The legacy arm confirmed Full in 20/20 games. AcceleratedDetection confirmed in 13/20 games, with a mean confirmation ply of 24.6 versus 10.0 for legacy. The accelerated confirmations came from `legacy_clock` in 11/20 games and `legacy_accelerated_fusion` in 5/20 games; the dedicated resilient reason fired in 0/40 games. This established that correlated clock and stable-fusion evidence could still promote Full without a genuine resilient-channel confirmation.

## Complete policy change

When `AcceleratedDetection` is enabled, promotion is now strict and resilient-channel-only. The detector first preserves known-computer handling. For unknown opponents, clock evidence, the legacy ceiling, and stable-fusion scores remain telemetry diagnostics but cannot independently promote Full. Only `accelerated_resilient()` can return `legacy_accelerated_resilient`, and it must satisfy the existing sample minimum, 2450+ estimated rating, resilient score and evidence floors, two-observation streak, good-quality mass, and catastrophic-error guard. When the option is disabled, the legacy path is unchanged.

This is a semantic fix rather than another threshold patch: the confirmation reason must correspond to a complete evidence channel, not merely correlated intermediate fields.

## Real validation

The official current Maia-3 5M UCI runtime was run on CPU for 10 games, split evenly between standard and strict accelerated arms, with a 60-second clock, zero increment, fixed openings, live telemetry, and a 48-ply horizon. The standard arm confirmed in 5/5 games at a mean ply of 14. The strict accelerated arm confirmed in 1/5 games at ply 36 and did not confirm in the other four games. It produced 23 observations per game and zero low-time skips. The one confirmation used `legacy_accelerated_resilient`; no clock or stable-fusion promotion remained.

The official Stockfish 19 binary was run for 6 games under the same safe clock protocol. Standard confirmed in 3/3 games at plies 32, 32, and 30. Strict accelerated confirmed in 3/3 games at plies 24, 30, and 28, all through `legacy_accelerated_resilient`. Each game produced 23 observations and zero low-time skips. This confirms that strict policy removes the Maia false-positive promotion paths while preserving and slightly improving genuine strong-engine detection in this small control sample.

## Validation status

The focused opponent-model suite passes 26 tests. The complete core-library suite and telemetry parser suite must be rerun before merge. The strict policy intentionally sacrifices early confirmations from clock and stable-fusion evidence in exchange for an interpretable promotion guarantee. Larger samples are still required before claiming a calibrated false-positive rate or universal latency improvement.

The independent ONNX result and the official Maia-3 UCI result are not identical model/runtime conditions. The former has no one-ply opponent-response ranking, while the latter uses the official current UCI inference package. Reporting both is necessary to separate model-pipeline artifacts from detector policy behavior.
