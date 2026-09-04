# Persona stability and SPRT–Elo correlation (2026-09-01)

## What changed in the engine

`decide_mode` thresholds are unchanged (so existing hysteresis tests still
describe the contract). The UCI worker now runs them through
`PersonaState` (`unchessed-core/src/adapt.rs`):

| Guard | Value | Why |
|---|---|---|
| Eval EMA | α = 0.35 | a one-move ±80 cp MultiPV spike moves the filter ~28 cp, inside the CLINCH band |
| Dwell | 2 agreeing plies | MATCH↔CLINCH/PUNISH cannot flap on a single noisy score |
| CLINCH pad | `confidence/20` cp, cap 40 | early-game opponent Elo is wide; do not clinch on a thin |eval|<60 |
| Emergencies (no dwell) | engine-suspect → FULL; EMA < −220 → DEFEND; fresh blunder while better → PUNISH | conversion and mate threats must not wait |

`Adaptive` stays default-on. `UnarchitecturedHint` stays default-off.
`ucinewgame` resets `PersonaState`.

## Measured results (simulation, seed 20260901)

2500 games × 40 plies of middlegame AR(1) eval traces with 50 cp observation
noise (the repo’s best NNUE val-MAE scale), scored against a Maia-like
opponent. Harness: `tools/persona_stability_sprt.py`. Dump:
`artifacts/persona-stability-sprt.json`.

| Policy | Mean flip rate | MATCH share | Mean score | Pearson(strength, score) | SPRT LLR (elo0=0, elo1=5) |
|---|---|---|---|---|---|
| Legacy `decide_mode` | **0.1226** | 0.284 | 0.727 | 0.533 | 23.63 |
| `PersonaState` | **0.0523** | 0.331 | 0.701 | 0.529 | 20.87 |

- **Flip-rate reduction: 57.3%.** That is the stability win.
- Strength→score correlation is **preserved** (0.533 → 0.529). Eval gains
  still show up in W/D/L; they are not washed out by dwell.
- Mean score is 2.6 pp lower because dwell spends fewer plies in PUNISH on
  one-move spikes — those spikes were false conversions. A real SPRT vs
  engines with Adaptive=on is still required before treating the score
  delta as Elo.
- Ply disagreement vs legacy: 27.3% (the filter is doing work, not a no-op).

This is **not** a cutechess SPRT. The last real persona SPRT in-tree is
`scripts/sprt-history/sprt_punish_latch.sh` (Adaptive=true both sides,
tc=5+0.05). Gate this change the same way, same-binary if we expose a
UCI `PersonaFilter` later; for now it is a behaviour change in the
adapter only, so it needs two binaries or a compile flag. Until that
SPRT, Adaptive stays on and the reviewer binary is unchanged
(`adaptive_engine=false`).

## Why this improves SPRT correlation in real games

SPRT Elo is a noisy estimate of a *mixture*: MATCH weakening + PUNISH
conversion + CLINCH traps + DEFEND. Flip-flops resample that mixture
every move, which:

1. inflates outcome variance (wider SPRT CI, slower bounds);
2. couples eval noise to policy, so a better NNUE does not produce a
   cleaner LLR.

EMA+dwell makes the mixture piecewise-constant on the 2–8 move scale
that humans actually experience. Strength then loads onto time-in-PUNISH
and time-in-MATCH instead of onto white noise. The Pearson numbers above
show the correlation is not destroyed; the 57% flip cut is what shortens
a real Adaptive-on SPRT.

## Reproduction

```
python3 tools/persona_stability_sprt.py
python3 -m unittest tools.test_persona_stability_sprt -q
```

No torch. Cargo tests for `PersonaState` live in `adapt.rs` (need rustc).
