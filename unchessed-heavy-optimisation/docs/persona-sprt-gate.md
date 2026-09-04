# PersonaSmooth / EngineDetectV2 cutechess SPRT gate

This commit does **not** flip either UCI default. Both stay `false`.

The gate is `scripts/sprt-history/sprt_persona_smooth_detect.sh`:

- Adaptive=true both sides (otherwise `decide_mode` never reaches the
  new paths)
- same binary, options only: trial has `PersonaSmooth=true` and
  `EngineDetectV2=true`; baseline has both `false`
- `tc=5+0.05`, SPRT elo0=0 elo1=5, alpha=beta=0.05

Smoke: `scripts/sprt-history/smoke_persona_smooth_detect.sh` (2 games).

This sandbox has no rustc and no cutechess. The SPRT was **not** run
here. Do not treat this script as a substitute for a real result.
