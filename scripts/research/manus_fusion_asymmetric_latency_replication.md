# Independent replication of the asymmetric detection-latency test (commit 5d43f6a)

Ported the evidence-fusion path (commit 8c460d0, main-branch port at
6db3650) and ran an independent asymmetric latency test against a
different real distinct opponent -- Stockfish 19, not 16 -- to see if
your result held up on different hardware/software.

## Setup

Real UCI games, not simulated: `unchessed-adapter` (main HEAD, this
port) vs the official Stockfish 19 Windows universal binary, one
thread, 64 MiB hash, `Adaptive=true`, `OwnBook=false`,
`AdapterTelemetry=true`, `UCI_Opponent=- - human UnknownOpponent`
(same anonymization approach as your test, to keep the known-engine
table out of it). 8 games per arm, 6 fixed opening prefixes,
`movetime=1500ms`, 60-ply cap. Telemetry parsed directly for the first
`persona_decision` record where `mode_after=FULL`.

One real methodological trap worth flagging: my first attempt used
`movetime=100ms` and got **zero** Full confirmations in all 16 games.
Turned out every single opponent observation was being skipped with
`reason=low_time` -- the engine's own low-time gate
(`hard_ms < 1000ms`) correctly suppresses opponent-observation probing
whenever the move budget itself is under a second, which a 100ms
movetime always triggers. Worth keeping in mind for any future
telemetry-based latency test: `movetime` (or any clock setting) has to
clear that 1-second floor or the detector never gets fed any evidence
at all, regardless of which detection path is under test.

## Result

| Arm | Games | Mean first-Full ply | Median | Range |
|---|---:|---:|---:|---:|
| Standard | 8 | 28.0 | 28 | 26-31 |
| Accelerated (fusion) | 8 | 27.75 | 28 | 26-29 |

Per-game plies -- standard: 26, 31, 28, 27, 26, 29, 28, 29. Accelerated:
26, 27, 28, 27, 28, 29, 28, 29.

**~0.25 ply difference on average -- no meaningful latency benefit**,
independently confirming your own asymmetric result against Stockfish
16 (mixed: earlier in some paired openings, later or equal in others,
no established uniform improvement).

## A candidate explanation for why both arms converge

Both arms land in a tight 26-31 ply band regardless of which detector
is active. That's consistent with the legacy ceiling rule already
being fast enough against a genuinely strong, *consistently* strong
opponent -- there's no "looks erratic but is actually strong" period
for the fusion path's extra robustness to shorten, because Stockfish
never looks erratic in the first place. That gap is specifically what
happened in the real game that motivated this whole feature (Dragon by
Komodo, 17 moves of "erratic (sandbagging?)" before confirming) --
which suggests the fusion path's real target isn't "any strong
opponent," it's "a strong opponent whose move-quality signal looks
noisy or inconsistent for a while." Stockfish's play is about as
un-noisy as it gets, so this test may be structurally unable to show
the benefit even if it's real.

**Suggested next test:** an opponent that's strong but plays with more
apparent inconsistency -- a Maia model (human-like, intentionally
imperfect but not weak) would stress exactly the erratic-but-actually-
strong distinction this feature is meant to catch, unlike a clean
engine-vs-engine match.
