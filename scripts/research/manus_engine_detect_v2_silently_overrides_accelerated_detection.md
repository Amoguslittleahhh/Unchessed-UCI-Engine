# EngineDetectV2 silently disables AcceleratedDetection (found via real gameplay, not testing)

Found this from a user playing real games in En Croissant, not from any
of this session's automated batches -- worth noting because it's
exactly the kind of gap a synthetic test harness wouldn't surface (it
only shows up when someone turns on multiple detection-related UCI
options at once, which none of our test scripts do).

## The bug

`OpponentModel::suspect_reason()` (`unchessed-core/src/adapt.rs`):

```rust
pub fn suspect_reason(&self) -> SuspectReason {
    if self.experimental_detect {
        return self.suspect_reason_v2();
    }
    if self.is_computer {
        return SuspectReason::LegacyComputer;
    }
    // Strict opt-in policy: AcceleratedDetection may promote Full only
    // after the resilient channel itself has satisfied every guard...
    if self.accelerated_detect {
        if self.accelerated_resilient() {
            return SuspectReason::LegacyAcceleratedResilient;
        }
        return SuspectReason::None;
    }
    ...
}
```

`experimental_detect` is set by `EngineDetectV2`. When it's true, the
function returns immediately -- the `accelerated_detect` branch below
it (set by `AcceleratedDetection`) is never reached. Turning both UCI
options on at once, which is a completely natural thing for a user to
do ("more detection settings = better detection," reasonably), doesn't
combine their effects. `EngineDetectV2` silently wins and
`AcceleratedDetection` -- all of this session's clock-corroboration,
resilient-only, and fingerprint-gate work -- becomes dead code for the
rest of the game. Nothing warns the user; the UCI options list gives
no indication these are mutually exclusive rather than additive.

## Real-game consequence

A real 60+1 game vs actual Stockfish 19 with both options on
(`AcceleratedDetection=true`, `EngineDetectV2=true`,
`PersonaSmooth=true`, `AdapterTelemetry=true`) never once reached
`Mode::Full` in 47 moves of a game that was already +9 for White by
move 8. `persona MATCH -> DEFEND` fires once (move 8, eval collapsed);
there is no `-> FULL` transition anywhere in the telemetry.

Root cause: with `EngineDetectV2` active, the only path for an
undeclared opponent is `V2AnonymousCeiling`:

```rust
if self.weight >= 11.0
    && self.samples >= 16
    && self.mean >= 2500.0
    && self.low_loss_streak >= 12
{
    SuspectReason::V2AnonymousCeiling
}
```

`low_loss_streak >= 12` requires twelve *consecutive* observations
with cp-loss <= 40, no interruption. Against a real Stockfish that is
already crushingly ahead, its moves don't need to be "the probe's idea
of objectively best" every single time -- once winning, "good enough"
moves are common, and each one that reads as imprecise to the shallow
probe resets the streak to 0. Twelve-in-a-row never happened once in
this real 47-move game despite Stockfish's estimate repeatedly sitting
well above 2500 (peaked ~2754). Two real move-quality blunders
(+0.75 -> +2.54, then +2.30 -> +7.80, both flagged `?`/`??` by
Stockfish's own analysis) were both played in `Mode::Match` -- the
persona was still deliberately choosing human-plausible-quality moves
against an opponent it never once concluded was suspicious, and those
lower-quality moves happened to be the real losing mistakes. Once the
position collapsed, the game spent the rest of its length in
`Mode::Defend` at "maximum resistance" without ever unlocking
full-strength search, because the V2 ceiling requirement never came
close.

## Confirming AcceleratedDetection alone still works

Same user, same real 60+1 vs Stockfish 19, `EngineDetectV2` turned
back off, `AcceleratedDetection=true` left on alone: `Mode::Full`
locked in at move 18 (`persona MATCH -> FULL`, `suspect_reason=legacy_accelerated_resilient`,
`opponent~2718`) -- consistent with this session's own latency
numbers for the resilient channel against real Stockfish. One brief
single-move flap back to `Match` around move 25 self-corrected within
two plies (the resilient streak dipping below threshold for exactly
one observation, then reconfirming) -- not a bug, just normal
oscillation right at a threshold boundary. The three real inaccuracies
in that second game all happened *after* Full mode was already active,
so that loss is a genuine strength gap against real Stockfish 19, not
a repeat of the detection failure above.

## Suggested fix

Two independent things worth considering, not mutually exclusive:

1. **Don't let `EngineDetectV2` silently shadow `AcceleratedDetection`.**
   Either refuse/warn on the combination via UCI (`setoption` handler
   could print an `info string` explaining the two are exclusive when
   both are set true), or make the two genuinely composable instead of
   one hard-overriding the other, if that's ever a real intended use
   case. Silent override with zero indication is the actual problem,
   more than which one wins.
2. **`low_loss_streak >= 12` for `V2AnonymousCeiling` looks
   miscalibrated against real strong-opponent conversion play**,
   independent of the override bug -- it's real, in-production code
   (`experimental_detect`/`EngineDetectV2` is a shipped UCI option, not
   dead code) that apparently never fires against a genuinely dominant
   opponent in a real game. Worth the same kind of real-game
   verification this session gave the `AcceleratedDetection` path, if
   `EngineDetectV2` is meant to stay a real, usable option rather than
   be deprecated in favor of `AcceleratedDetection`.

Source: real games played by the user in En Croissant
(`2026.09.07_Unchessed Game Adapter - Stockfish.pgn` /
`logs (unchessed vs stockfish v2).csv` for the failure,
`2026.09.07_Stockfish - Unchessed Game Adapter v3.pgn` /
`logs (unchessed vs stockfish v3).csv` for the confirming re-run),
`unchessed-core/src/adapt.rs` at current `main` (`158f79e`).
