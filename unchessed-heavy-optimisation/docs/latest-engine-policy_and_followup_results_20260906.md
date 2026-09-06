# Latest-engine policy and resilient-channel follow-up

## Engine selection

The official Stockfish download page identified Stockfish 19 as the current release, so the follow-up control used the official Linux x86-64 universal Stockfish 19 binary. For the human-like opponent, the current official CSSLab Maia-3 inference repository was used with its `maia3-5m` model on CPU. The exact runtimes were recorded rather than substituting the older apt Stockfish 16 binary or a simulated opponent.

## Stockfish 19 asymmetric control

The corrected clock protocol used a 60-second starting clock, zero increment for the latency probe, one thread, 64 MiB hash, 32-ply cap, fixed openings, live UCI telemetry, and an unknown-human opponent declaration. With two games per arm, standard first-Full confirmation was observed in one game at ply 30 and not observed in the other within the 32-ply horizon. The resilient arm confirmed at plies 26 and 24, both with `legacy_accelerated_resilient`. All games produced 15 observations and zero low-time skips.

This four-game result is directionally positive for the resilient channel but is too small for a stable estimate. It does, however, replace the earlier Stockfish-16 control in the follow-up evidence with Stockfish 19.

## Maia-3 false-positive evaluation

The latest Maia-3 5M CPU UCI model was run as a real opponent under the same 60-second clock protocol. In the first two-game-per-arm run, both arms produced early `legacy_clock` confirmations around plies 12--14. This exposed a safety problem in the legacy clock path rather than proving a resilient-channel false positive.

The detector was then changed so that when `AcceleratedDetection` is enabled, clock suspicion requires at least 10 samples, mean at least 2450 Elo, resilient score and evidence floors, and two consecutive resilient observations. After the fix, the Maia-3 rerun produced no Full confirmation in one resilient game and a stable-fusion confirmation at ply 26 in the other. The resilient arm therefore did not promote the low-strength Maia-3 trace through the resilient clock path in this small run. The standard arm continued to show the pre-existing legacy-clock behavior, which remains a separate compatibility concern.

Every Maia-3 game produced 15 observations and zero low-time skips. The result is a safety signal, not a calibrated false-positive rate: the sample is four games and the standard legacy path still needs a separate policy decision.

## Strength SPRT probe

A real paired standard-versus-resilient harness was added with alternating colors, identical Unarchitectured Metal settings, 10-second starting clocks, 100 ms increments, and a 40-ply cap. The bounded four-game probe reached the cap in all games, producing 0--0--4 from the resilient perspective and an LLR of 0.000 within the continuation bounds [-2.944, 2.944]. Because the games were unfinished at the cap, they are treated as inconclusive draws; this is not a promotion-grade strength SPRT. An earlier 60-second/80-ply attempt exceeded the practical runtime budget and was stopped without using partial games as evidence.

## Conclusion

The follow-up establishes the requested latest-engine policy and provides concrete real-engine testing. It supports the resilient channel against Stockfish 19, identifies and partially fixes a Maia-3-adjacent clock false-positive path, and shows no evidence of a strength loss in the small bounded probe. It does not yet establish a final false-positive rate or promotion-grade strength neutrality. The next valid campaign requires completed paired games, a larger Maia-3 sample across at least two model sizes or Elo settings, and explicit reporting of legacy-clock versus resilient-channel reasons.

Sources: https://stockfishchess.org/download/ ; https://github.com/CSSLab/maia3 ; https://github.com/CSSLab/maia-chess
