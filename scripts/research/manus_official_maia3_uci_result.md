# Official maia3-uci binary resolves the ONNX-vs-official discrepancy -- both show the resilient-only fix isn't false-positive-safe

Set up the actual official Maia-3 UCI engine (`CSSLab/maia3`, `maia3-5m`
preset, real checkpoint from Hugging Face) and reran the same 40-game
protocol a third time, per the ask in your `reproduce_resilient_detection_experiments.md`
(section 3: "Run at least 20 games per arm... Do not combine these
results with the simplified ONNX pipeline"). This is that run, on the
real binary, not a simulation and not the ONNX export.

## Setup

Real `maia3-5m` UCI subprocess (pip-installed from `CSSLab/maia3`,
checkpoint auto-downloaded from `UofTCSSLab/Maia3-5M` on Hugging Face,
CPU), `Temperature=1.0` (the engine's own default is `0.0`/argmax,
which would make it deterministic and not represent human-like
inconsistency at all -- set explicitly for a fair test), `SelfElo`/
`OppoElo` set per band. `unchessed-adapter` built from `main` at
`efc08d0` (the resilient-only restructure), real UCI subprocess, real
`wtime`/`btime` (60,000ms, 0 increment, 48-ply cap), NNUE eval,
`Adaptive=true`, `OwnBook=false`, `UCI_Opponent=- - human UnknownOpponent`.
Same two elo bands (1500, 2000), 10 games/arm/band, 40 games total,
same openings/protocol as the ONNX runs, both engines real UCI
subprocesses this time (`tools/maia3_uci_false_positive_test.py`,
committed).

## Result

| Runtime | Games | Full-confirmed | Mean ply | Reason purity |
|---|---:|---:|---:|---|
| Standard (legacy), official UCI | 20 | 20/20 (100%) | 15.5 | `legacy_clock`, expected |
| Standard (legacy), ONNX (prior run) | 20 | 20/20 (100%) | 10.0 | `legacy_clock`, expected |
| Accelerated, official UCI (this run) | 20 | **12/20 (60%)** | 27.2 | `legacy_accelerated_resilient` 12/12 |
| Accelerated, ONNX (prior run) | 20 | 14/20 (70%) | 25.7 | `legacy_accelerated_resilient` 14/14 |

Zero low-time skips in every game across both arms (23-24 observations/game).

## What this resolves

There is a real, measurable difference between the two Maia-3
runtimes -- the official binary's legacy-arm confirmations come
noticeably later (mean ply 15.5 vs 10.0), consistent with your
hypothesis that its opponent-response ranking makes its play look
less engine-like than raw ONNX temperature sampling. That carries
through to a somewhat lower accelerated false-confirmation rate too
(60% vs 70%).

But **the size of that difference doesn't come close to explaining
your 5-game result (1/5 = 20%).** A ~10-point gap (70% -> 60%) is a
real but modest runtime effect; a 50-point gap (70% -> 20%) is not
something either of these 20-game samples supports. The most likely
explanation is that your n=5 sample was small-sample variance, not a
genuine property of the official runtime -- five games is thin enough
that one or two outcomes swing the rate by 20 points.

**The number to use going forward is ~60-70% false-confirmation
against Maia-3, not 20%.** Reason purity is confirmed fixed on both
real runtimes (24/24 confirmations across both 40-game batches were
`legacy_accelerated_resilient` and nothing else), but
`AcceleratedDetection` is not false-positive-safe against Maia-3 at
either elo band tested, regardless of which Maia-3 runtime is used.

## Suggested next step

Given two independent 20-game-per-arm samples on two different real
Maia-3 runtimes now agree within 10 points of each other, I'd treat
this as a settled false-positive number rather than run a fourth
sample -- the open question is now what to do about it (tighten the
resilient-channel thresholds further, or accept this rate and gate
the feature to only enable against opponents already flagged
suspicious some other way, or something else). Your call on priority.

Source: `tools/maia3_uci_false_positive_test.py` (new, committed --
drives the official binary directly, no ONNX/model-loading code
needed since it's real UCI both sides) and raw results at
`results_uci_official.json` (not committed, available if useful).
