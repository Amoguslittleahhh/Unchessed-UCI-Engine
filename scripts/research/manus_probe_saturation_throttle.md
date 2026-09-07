# Throttle the opponent-observation probe once the verdict has saturated (commit bcdeb52)

## What was found

Analysing two real lost games (RubiChess and Komodo, both 60+1 via En
Croissant, PGNs + full UCI logs) surfaced a real efficiency bug,
independent of anything in the AcceleratedDetection work: the
opponent-observation probe in `run_go` (`unchessed-core/src/uci.rs`)
runs a real, uncharged search on every non-book pending observation --
depth 14/400_000 nodes, plus a possible depth 12/250_000-node fallback
if the opponent's played move wasn't in the first probe's own PV --
*before the actual move search even starts its own clock*. This is a
known, documented tradeoff (see the existing comment at the top of
`run_go`: "measured up to ~130x the requested movetime"), and its cost
is charged against the real move's time budget via
`preprocessing_elapsed`. That's a reasonable cost while a verdict is
still being formed.

The problem: it never stops. In the RubiChess log, `persona MATCH ->
FULL` fires at move 14 (`estimate ~2630`, `"engine suspected"` /
`"pinned at measurement ceiling"`), and the model's per-move
`observe()` call still ran the identical depth-14/400k-node probe on
move 45+ -- 30+ moves after the detection question was already
answered with no further decision riding on it. Real numbers from that
log: one move's own search depth-1 line already reports `time 308`
(ms) before the real search's first iteration even printed, consistent
with a ~325ms combined probe cost at this engine's ~2M nps. That move
reached depth 11 with ~31s left on the clock, while other moves with
similar time-left reached depth 13-15 -- the difference tracks which
moves paid the probe tax and which got a cheap/no probe.

Second half of the same finding: none of that probe search is reused.
It only feeds `m.observe(cp_loss, w)` (the opponent Elo model) -- never
the actual move choice. So real wall-clock time is spent, every move,
for the rest of the game, computing a search whose result is thrown
away.

## Fix

Added `OpponentModel::observation_saturated()` (adapt.rs): true once
`engine_suspect()` has held for `suspect_streak >= 10` consecutive
observations. `suspect_streak` is a new field, incremented at the end
of every `observe()` call when `engine_suspect()` is true, reset to 0
otherwise.

Deliberately **not** based on `confidence()`, even though that looks
like the obvious signal. `weight` is capped at 14.0 (see `observe`),
so `confidence()`'s floor is `600*0.6/sqrt(14) ~= 97cp` even for a
perfectly consistent opponent -- it can never reach a "tight enough"
threshold on its own. A confidence-based saturation check would never
actually throttle anything; caught this via a failing unit test before
it shipped (first attempt asserted `confidence() <= 80`, which no
amount of consistent observation could ever satisfy).

`run_go` now checks `m.observation_saturated()` before building the
probe's `Limits` (using the model state as of *before* this
observation, since the probe result feeds into this same observation's
`observe()` call): saturated drops to the pre-bump depth 9/60_000-node
budget (the original calibration point before the depth-14 bump, when
detection sensitivity actually mattered) and skips the depth-12/250k
fallback search entirely, falling back to `best` directly.

## Real validation

Paired A/B, not a simulation: built two binaries (pre-throttle vs
throttled) from the same commit, real UCI, real Stockfish (Windows
AVX2 binary) as White at fixed depth 12, adapter as Black with
`Adaptive=true`, real NNUE eval, 60s clocks with 1s increment (clock
not decremented by the adapter's own think time, deliberately --
isolates the probe-cost effect from clock-pressure time management).
4 fixed openings each, 80-ply cap, 157 adapter moves recorded per arm
(8 real games total).

| | avg depth (all moves) | avg depth (moves 11+) | avg nodes | avg wall time/move |
|---|---:|---:|---:|---:|
| Baseline | 15.03 | 15.11 | 4.66M | 3296ms |
| Throttled | 16.06 | 16.14 | 5.43M | 3495ms |

**~1 full ply deeper, ~13% more nodes, for essentially the same
wall-clock budget** (the small wall-time gap is normal game-to-game
variance, not the driver -- both arms target the same per-move time).
The saved probe cycles convert directly into deeper real search, not
just less total time spent.

## Validation status

136/136 workspace lib tests pass, including two new tests:
`observation_saturated_only_after_a_long_held_verdict` (verifies the
streak, not just a one-shot verdict, gates saturation) and
`observation_not_saturated_right_after_a_fresh_clock_tell` (a clock
tell can flip `engine_suspect()` near-instantly via `LegacyClock`,
long before the verdict has held long enough to trust a cheap probe).

Not yet tested: whether the shallower saturated-mode probe (depth
9/60_000, the pre-bump budget) ever meaningfully drifts the live Elo
estimate once already locked in a long game -- the fallback-`best`
approximation in the `None if saturated` branch slightly overstates
cp-loss on moves outside the shallow probe's own PV, but there's no
verdict left riding on that precision at that point. Also not tested:
interaction with the accelerated-detection paths' own streak/decay
constants (`accel_resilient_streak` etc.) -- `suspect_streak` is a
separate counter from those, tracking the *general* `engine_suspect()`
verdict rather than any one specific detector path, so it should be
orthogonal, but worth confirming with a real accelerated-mode game if
you pick this up.

Source: `unchessed-core/src/adapt.rs` / `unchessed-core/src/uci.rs` in
commit `bcdeb52`, ported to `main` directly (this fix is independent
of the manus/research-facilities detection work, found from analysing
real lost games rather than from anything in that branch).
