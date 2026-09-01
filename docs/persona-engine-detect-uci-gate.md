# PersonaSmooth / EngineDetectV2 (default-off)

Round 15 put EMA/dwell persona and a retuned `engine_suspect` on the live
`run_go` path with no UCI opt-out and no cutechess SPRT. That was rejected.

This round:

- Fixes `run_go(..., persona: Arc<Mutex<PersonaState>>)` so it matches the
  worker spawn (the previous signature still said `Arc<Mutex<Mode>>`).
- Live default is the **old** adapter: `PersonaState::update` calls
  `decide_mode` on the raw eval; `engine_suspect` is the pre-retune rule
  (labelled computer, `suspicion >= 3`, or `weight >= 10 && mean >= 2450`).
- Opt-in UCI (Adapter only, both default **false**):
  - `PersonaSmooth` — EMA + 2-ply dwell
  - `EngineDetectV2` — opening-discounted observe, middlegame clock tell,
    labelled-computer + streak ceiling

Do not flip either default without a real SPRT (`scripts/sprt-history/`
style). Python simulations are not that gate.

This sandbox has no `rustc`; do not treat this note as a compile proof.
