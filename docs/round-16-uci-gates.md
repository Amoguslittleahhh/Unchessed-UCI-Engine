# Round 16: UCI gates for round-15 adapter experiments

**This sandbox has no `rustc`/`cargo`.** `cargo test --workspace --release`
was **not** run here. Do not treat this note as a compile certificate.
A machine with the toolchain must compile before merge.

## Is Unarchitectured v1 “done” (fast enough to actually use)?

**No.** `UnarchitecturedHint` stays **default-off**. Real SPRTs (round 7
and later) never trended positive. Speed work (AVX2, int8 weights/int16
acts, exits) made the forward cheaper; it did not make the hint a
default-on search term. That is the round-0 rule, unchanged.

## Round 15 rejection — what this branch does now (`502eb26`+)

Both live-game behavior changes are **opt-in**, old path is default:

| UCI option | Default | When true |
|---|---|---|
| `PersonaSmooth` | **false** | `PersonaState::update` EMA+dwell |
| `EngineDetectV2` | **false** | Maia≠FULL, opening clock mute, ceiling streak |

With both false, `PersonaState::update` calls `decide_mode` on the raw
eval (legacy hysteresis). `engine_suspect` is the old
`is_computer \|\| suspicion≥3 \|\| (weight≥10 && mean≥2450)` rule.

A real cutechess SPRT (`Adaptive=true` both sides, same shape as
`scripts/sprt-history/sprt_punish_latch.sh`) is still required **before
either default flips**. Simulation is not that gate.

NNUE 108M / cloud 178M: not this round. Reviewer 108M result stands
(−155.6 Elo, 178M still NO-GO).
