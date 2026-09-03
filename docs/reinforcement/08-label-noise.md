# 08 — NNUE label-noise evidence update

**Investigation ID:** `08-label-noise`
**Tier:** 1 (exactly one item)
**Repository:** `/home/ubuntu/Unchessed-UCI-Engine`
**Branch:** `manus/research-facilities`
**Decision:** **Reopened.** A real paired live-generation measurement contradicts the earlier working assumption that the 5,000-node labels have an approximately 50–56 cp noise floor. This reopens label-quality analysis, but does **not** authorise retraining, a default change, or a strength claim.

## Executive summary

The original report correctly stopped at the missing-input blocker: the archived 104-byte NNUE records do not retain castling rights, en-passant, or a complete replayable position, so they cannot safely be re-searched at higher depth. A reviewer-run diagnostic avoided that trap rather than fabricating state. It generated fresh labels during ordinary PGN replay, when the full legal `Position` was still in memory, and compared the existing 5,000-node HCE score with a second search on the **same accepted position**.

Across four PGN sources and two depth multipliers, the paired differences were **17.41–21.99 cp MAE** and **0.929–0.983 Pearson correlation**. The 10× cases (50,000 nodes) were 20.96 cp / 0.983 and 17.41 cp / 0.957; the 20× cases (100,000 nodes) were 21.20 cp / 0.929 and 21.99 cp / 0.940. This is materially below the prior modelled 50–56 cp floor and therefore **contradicts that numerical assumption at the tested scale**.

The result is a bounded search-depth agreement measurement, not proof that the deeper score is correct. It does not measure within-depth nondeterminism, the unavailable 108M corpus, or the Elo effect of retraining. The practical consequence is precise: stop treating shallow-search label noise as the primary established explanation for the NNUE plateau; investigate effective reachable-position coverage, architecture/representation, and other label-quality axes before proposing a large relabel/retrain. Any candidate still requires the normal reject-only screen and real paired-game SPRT.

## Question and decision boundary

The question is whether changing the label search budget from the repository’s 5,000-node HCE search to a substantially deeper search moves scores by an amount compatible with the former assumed floor. For paired scores `old_i` and `new_i`, the diagnostic reports:

```text
MAE = mean(abs(new_i - old_i))
Pearson = corr(new_i, old_i)
```

The comparison is meaningful only when both scores belong to the same ordered positions. Running the generator twice with different node budgets is **not** sufficient: the M2 quiet-position filter itself uses search, so changing the budget changes which positions are accepted. The measurement caught this methodology trap: an initial unpaired attempt changed 288 of 300 output boards and produced the misleading MAE 159 cp / Pearson 0.008. The accepted-position side-channel was added specifically to prevent that selection confound.

## Measurement method and provenance

The reviewer-run measurement used two environment-gated datagen hooks. `UNCHESSED_NNUE_LABEL_NODES` overrides the primary label search, while `UNCHESSED_NNUE_LABEL_NODES_COMPARE` runs a second search immediately after the primary search has accepted the position. The second score is printed as `LABEL_COMPARE old=<primary> new=<compare>` to stderr and does not modify the NNUE output file. When unset, the hooks are no-ops and normal behavior remains unchanged; the associated report records the full release workspace test count increasing from 118/118 to 123/123 without regressions.

| Source | Comparison nodes | Multiplier vs 5,000 | Paired records | MAE | RMS | Sign flips | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| `data/training-elo/elo-1700.pgn` | 50,000 | 10× | 300 | 20.96 cp | 31.94 cp | 11 (3.7%) | 0.983 |
| `data/training/leagues/bl0607.pgn` | 50,000 | 10× | 1,000 | 17.41 cp | 26.90 cp | 66 (6.6%) | 0.957 |
| `data/training-elo/elo-2500.pgn` | 100,000 | 20× | 800 | 21.20 cp | 34.20 cp | 70 (8.8%) | 0.929 |
| `data/training/players/Carlsen.pgn` | 100,000 | 20× | 800 | 21.99 cp | 36.05 cp | 63 (7.9%) | 0.940 |

The four samples total **2,900 paired records**. Moving from 10× to 20× increased MAE only modestly (about 1–4 cp), while Pearson declined from 0.957–0.983 to 0.929–0.940 and sign flips increased from 3.7–6.6% to 7.9–8.8%. That pattern is consistent with real but bounded search disagreement in this experiment; it is not evidence of a runaway error at the 5,000-versus-50,000/100,000 boundary.

## What is established and what is not

### Established by the real measurement

The specific round-14 assumption of approximately 50–56 cp error, derived from an assumed teacher standard deviation of about 70 cp and a Gaussian MAE calculation, is **not supported** by this direct 10×/20× comparison. The observed 17–22 cp MAE is roughly one-third of that assumed floor and is below the reported 47.8–57.4 cp validation-MAE range from the earlier scaling analysis. The label-noise explanation for the strength plateau must therefore be revisited rather than treated as settled.

### Explicit scope limits

This measurement does not establish that the 50,000- or 100,000-node score is a better ground truth; both are engine search point estimates. It does not estimate noise within repeated 5,000-node searches, test nondeterminism, compare against tablebases or a stronger independent engine, or evaluate tactical positions excluded by the quiet filters. It does not retroactively label the missing 108M records, because the old records lack the state required for legal replay. It is also a small sample (300, 1,000, 800, and 800 records), from one engine, four PGN sources, and only two multipliers; it cannot rule out a qualitatively different relationship at much deeper search or on a different position distribution.

The sign-flip rates matter. Even with high correlation, 3.7–8.8% of pairs crossed zero in these samples. Correlation measures association, not calibration or correctness, and MAE measures disagreement with the chosen comparator, not causal training benefit. A score-wise result cannot substitute for held-out evaluation or a playing-strength test.

## Consequences for the NNUE research plan

1. **Retire the 50–56 cp floor as a measured-looking premise.** It was a simulation/model assumption, not an observed property of this engine’s labels. Do not use it to argue that validation MAE is already at an irreducible label ceiling.
2. **Reopen, but do not escalate, label research.** The low-cost evidence supports further diagnosis if a concrete question justifies it. A much larger multiplier or larger sample is optional, not an automatic campaign; any extension should be preregistered and should preserve same-position pairing and provenance.
3. **Shift the leading hypotheses.** The plateau is now more plausibly related to effective data volume/support at reachable positions, architecture/representation, train–deployment distribution mismatch, or a different teacher bias. Raw record count is not effective coverage, and the current result does not distinguish these alternatives.
4. **Keep the state/provenance blocker for existing-corpus relabeling.** The result does not make it legal to search the old flattened records. Future corpora need a canonical full-position/state companion or an auditable deterministic regeneration path.
5. **No retraining or default movement follows.** A lower label MAE, higher Pearson, validation improvement, or smoke-test pass is not an Elo result. Any trained candidate must preserve the current default until it passes the project’s real paired-game SPRT with pinned binary, model, options, book, hardware, and time-control provenance.

## Reproduction command

Build the datagen binary once, then replay a PGN with the compare hook. The exact command form recorded by the measurement is:

```bash
cargo build --release -p unchessed-datagen
UNCHESSED_NNUE_MIN_BASE_SECS=0 UNCHESSED_NNUE_LABEL_NODES_COMPARE=50000 \
  ./target/release/unchessed-datagen nnue out.bin 0 1 <n> <pgn-file>
```

Use `UNCHESSED_NNUE_LABEL_NODES_COMPARE=100000` for the 20× setting. Replace `<n>` and `<pgn-file>` with the intended sample limit and one of the source files above. Parse the `LABEL_COMPARE` stderr records in generation order and compute MAE, RMS, sign flips, and Pearson over the paired rows. The `out.bin` output is incidental to this diagnostic; the authoritative comparison is the side-channel stream. **Do not** use `UNCHESSED_NNUE_LABEL_NODES` alone as the comparison method: it changes the primary label search and can change M2 acceptance, creating a different position set.

This command is a reproduction recipe, not a claim that it was rerun in this task. No expensive work was rerun here, and no labels were synthesized.

## Verification ledger for this update

| Item | Status |
|---|---|
| Master brief and reinforcement reports 00–12 read | Verified in this update |
| Historical real measurement and its later two-sample extension read from repository history | Verified; source commits `6c8513c` and `6667f6f` |
| Four-source paired result transferred without changing reported values | Verified |
| Existing-corpus high-node relabel rerun | **Not run**; still blocked by missing full state/shards |
| New label generation or retraining | **Not run** |
| Cutechess/SPRT | **Not run** |
| Engine/default change | **Not made** |
| New literature search/read | Official Stockfish NNUE documentation and the *Study of the Proper NNUE Dataset* were read for conservative context |

## Literature context

Official Stockfish documentation describes converting centipawn evaluations into WDL space and optionally blending evaluation targets with game results; it also notes that the scaling and loss choice depend on the engine and data. This supports keeping score units, POV, target transformation, and teacher configuration explicit; it does not imply that lower MAE predicts Elo. The NNUE dataset study describes filtering tactically unstable positions and emphasizes diversity across material, phase, and other position types. That is consistent with treating the present quiet-filtered PGN sample as a bounded distributional measurement rather than a universal noise estimate. Neither source validates the deeper search as ground truth for Unchessed.

## Recommendation

**Reopen the investigation and pursue only a bounded, provenance-complete follow-up if it answers a specific unresolved question.** The real 17–22 cp / 0.93–0.98 result is sufficient to reject the old 50–56 cp working assumption at the tested scale. It is not sufficient to launch a relabel/retrain, spend compute, change a model default, or claim strength. Prioritize the effective-coverage/capacity diagnostic already described in `13-nnue-ceiling.md`, while retaining the requirement for full state provenance before any existing-corpus search relabeling.

## References

[1]: `docs/nnue-stronger-labels-existing-corpus.md` (prior stronger-label rationale and relabeling interface)
[2]: `docs/reinforcement/00-synthesis.md` and `docs/reinforcement/02-nnue.md` (state/provenance and promotion gates)
[3]: `docs/nnue-v4-108m-recipe-result.md` and `docs/nnue-v4-retrain-data-scaling-finding.md` (prior validation and strength results)
[4]: `docs/nnue-label-noise-real-measurement.md` in historical measurement commits `6c8513c` and `6667f6f` (reviewer-run paired results and method)
[5]: [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
[6]: [Tan and Watkinson Medina, *Study of the Proper NNUE Dataset*](https://arxiv.org/abs/2412.17948)
[7]: [scikit-learn, `mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/08-label-noise.md`
