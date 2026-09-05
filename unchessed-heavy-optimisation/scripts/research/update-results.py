from pathlib import Path

root = Path('/home/ubuntu/unchessed-research/unchessed-heavy-optimisation')
audit = root / 'docs/regression-audit-main-vs-heavy-2026-09-05.md'
s = audit.read_text()
s = s.replace(
"The reported 223-game and 200-game results are real reports, but they were not independently rerun in this sandbox because `cutechess-cli` is unavailable and the attached reports did not include complete machine-readable game packages. The corrected direction is based on the raw score and termination summaries supplied in the reports. A portable reproduction package should contain both binaries, compiler/toolchain information, NNUE hash, UCI transcript, opening suite, exact cutechess command, per-game PGNs, and SPRT logs.",
"The reported 223-game and 200-game results are real reports. The official Cute Chess v1.5.1 source is now installed under `/home/ubuntu/tools/cutechess`; a paired-game reproduction is being prepared with the explicit NNUE asset. The corrected direction is based on the raw score and termination summaries supplied in the reports. A portable reproduction package should contain both binaries, compiler/toolchain information, NNUE hash, UCI transcript, opening suite, exact cutechess command, per-game PGNs, and SPRT logs.")
s = s.replace(
"The isolated candidate must pass the usual UCI, legality, crash, and SPRT gates before any default or main-branch port is considered.",
"The isolated candidate must pass the usual UCI, legality, crash, and SPRT gates before any default or main-branch port is considered. The candidate build and smoke gates have passed; the Cute Chess paired match is the remaining causal-strength test.")
s += "\n## Explicit-NNUE ISA benchmark\n\nThe portable-versus-`x86-64-v3` harness was rerun with the canonical NNUE file passed explicitly. The file SHA-256 was `38845a16d73a6fe0bd4ac95c86c017c65c97bc82c7ce2f6dce2f1b3fbe8577b5`. Across 20 observations per build and TT sizes from 4 to 64 MiB, portable averaged 275,144 NPS and `x86-64-v3` averaged 281,851 NPS, a measured **2.44% v3 NPS advantage** in this VM. Mean resident memory was 18,592 versus 18,572 KiB; the build did not show a meaningful RSS difference. This result supersedes the unverified external 26% figure for this controlled run: the 26% value is retained only as an external report requiring hardware-specific reproduction. The raw TSV is `benchmarks/results/portable-v3-20260905-034215.tsv`.\n"
audit.write_text(s)

paper = root / 'ieee-paper/main.typ'
s = paper.read_text()
s = s.replace(
"The reported 223-game SPRT was not independently rerun in this sandbox because `cutechess-cli` is unavailable and the attached report did not include a complete machine-readable game package or exact launcher. The approximately +240 Elo advantage for heavy optimization is therefore retained as a supplied real result.",
"The reported 223-game SPRT was not independently rerun before the present tool installation because the attached report did not include a complete machine-readable game package or exact launcher. The official Cute Chess v1.5.1 source is now installed under `/home/ubuntu/tools/cutechess`, and the approximately +240 Elo advantage for heavy optimization is retained as a supplied real result pending the isolated paired reproduction.")
s = s.replace(
"A dedicated paired-game SPRT remains outstanding because `cutechess-cli` is unavailable in this sandbox.",
"A dedicated paired-game SPRT is the remaining causal-strength test; the official Cute Chess v1.5.1 source is now available in the sandbox for that run.")
needle = "A dedicated paired-game SPRT is the remaining causal-strength test; the official Cute Chess v1.5.1 source is now available in the sandbox for that run.\n"
insert = needle + "The controlled explicit-NNUE ISA benchmark used the canonical asset (SHA-256 `38845a16d73a6fe0bd4ac95c86c017c65c97bc82c7ce2f6dce2f1b3fbe8577b5`) at TT sizes 4--64 MiB. Portable averaged 275,144 NPS and `x86-64-v3` averaged 281,851 NPS over 20 observations per build, a 2.44% v3 advantage in this VM. This is lower than an external 26% report; the latter is documented as an unverified hardware-specific observation rather than silently presented as reproduced.\n"
s = s.replace(needle, insert, 1)
paper.write_text(s)
print('updated audit and paper')
