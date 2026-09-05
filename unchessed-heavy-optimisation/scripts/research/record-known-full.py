from pathlib import Path
root = Path('/home/ubuntu/unchessed-research/unchessed-heavy-optimisation')
audit = root/'docs/regression-audit-main-vs-heavy-2026-09-05.md'
s = audit.read_text()
old = "The candidate build and smoke gates have passed; the Cute Chess paired match is the remaining causal-strength test."
new = "The candidate build and smoke gates have passed. A first 20-game paired start-position Cute Chess run with the canonical NNUE, one thread, Hash 64 MiB, Adaptive=true, OwnBook=false, PersonaSmooth=false, EngineDetectV2=false, and 0.2+0.05 seconds scored `main 4 - 12 - 4 known_full`, corresponding to **+147.2 +/- 160.4 Elo for known_full-only relative to main**, LOS 2.3%. All games completed legally; the wide interval means this is directional evidence, not a conclusive causal estimate. The raw PGN and log are `benchmarks/known-full/known-full-vs-main-20-startpos.pgn` and `benchmarks/known-full/known-full-vs-main-20-startpos.log`."
s = s.replace(old, new, 1)
s += "\nThe first attempted run used `matetrack.epd`, which contains comment lines and fixed mate positions; Cute Chess warned about invalid comment FENs and the resulting 20-game score was heavily color-biased. It is retained only as a harness diagnostic, not as strength evidence. The clean start-position run above is the valid isolation result.\n"
audit.write_text(s)

paper = root/'ieee-paper/main.typ'
s = paper.read_text()
old = "A dedicated paired-game SPRT is the remaining causal-strength test; the official Cute Chess v1.5.1 source is now available in the sandbox for that run.\n"
new = "A first 20-game paired start-position Cute Chess run has now been completed with the canonical NNUE, one thread, Hash 64 MiB, Adaptive=true, OwnBook=false, PersonaSmooth=false, EngineDetectV2=false, and 0.2+0.05 seconds. It scored `main 4 - 12 - 4 known_full`, or **+147.2 +/- 160.4 Elo for known_full-only relative to main**, with LOS 2.3%. All games completed legally. The direction supports the MultiPV-narrowing hypothesis, but the interval is too wide to establish that it explains the full approximately 240-Elo branch result; a larger colour-balanced SPRT remains required. The raw PGN and log are archived in `benchmarks/known-full/known-full-vs-main-20-startpos.pgn` and `.log`.\n"
s = s.replace(old, new, 1)
paper.write_text(s)
print('recorded known_full isolation result')
