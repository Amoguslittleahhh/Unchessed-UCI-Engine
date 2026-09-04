# Unarchitectured v1 provenance-disjoint calibration

This is round 6 step 1: calibration of the exported Unarchitectured v1 student
against a real independent teacher, at a scale that can actually inform a
threshold decision.

## Why this was the blocking step

The only calibration numbers that existed before this round came from eight
hand-picked positions labelled by this engine's own HCE search at depth 4
(top-1 0.50–0.625). Those numbers could not inform anything:

- eight positions cannot separate signal from noise;
- the "teacher" was the same alpha-beta search the model would be hinting, so
  agreement measured self-similarity rather than correctness; and
- no random baseline was reported, so a top-1 number had no reference point.

## What was run

| Component | Value |
|---|---|
| Model | `artifacts/unarchitectured-v1-final.unarchv1` (real exported checkpoint) |
| Forward pass | `tools/reference_forward_unarchitectured_v1.py` (the repo's validated PyTorch reference, cross-checked against the Rust runtime at 5e-3) |
| Teacher | **Stockfish 17.1** (built from source in-sandbox), 400,000 nodes/position, `MultiPV` over *every* legal root move |
| Primary corpus | 600 positions, 301 distinct over-the-board games |
| Replication corpus | 300 positions, disjoint games, disjoint FENs, different seed |

Both corpora are committed (`artifacts/unarchitectured-v1-calibration-corpus.jsonl`,
`...-replication.jsonl`) along with the full reports
(`artifacts/unarchitectured-v1-calibration-report.json`, `...-replication.json`).

### Provenance claim, stated precisely

The student was trained on Lichess online play. This corpus is sampled from
**over-the-board tournament archives** (TWIC), which is a different population
of games entirely — so the two do not share games.

This repository contains no training-membership manifest, so **record-level
disjointness cannot be proven here and is not claimed**. What is claimed is
source-population disjointness. That distinction is recorded in the corpus
manifest itself, not just in this document.

Positions are additionally deduplicated by FEN, capped at two per source game,
stratified across opening/middlegame/endgame, and filtered to games where both
players are 2300+. All eight in-repo `TRIAL_FENS` smoke positions and both
frozen Python parity fixtures are explicitly excluded, so this corpus cannot
score the model on its own fixtures.

## Results — primary corpus (600 positions)

Mean legal moves per position: 30.00.

| Ordering | top-1 | top-3 | mean cp loss |
|---|---:|---:|---:|
| Random legal move | 0.0497 | 0.1457 | 256.7 |
| Static heuristic (MVV-LVA + promotion + check) | 0.1567 | 0.2700 | 260.2 |

| Exit | top-1 | top-1 95% CI | top-3 | mean rank | MRR | cp loss (mean) | cp loss (p90) | regret MAE | WDL Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2/128 | 0.1850 | [0.156, 0.218] | 0.3567 | 10.443 | 0.323 | 156.6 | 490 | 0.5208 | 0.8774 |
| 4/192 | 0.1950 | [0.165, 0.229] | 0.3767 | 10.007 | 0.335 | 144.7 | 459 | 0.5312 | 0.8991 |
| 8/256 | **0.2550** | [0.222, 0.291] | 0.4567 | 6.933 | 0.407 | **118.7** | 422 | 0.7260 | 0.9018 |

## Results — replication corpus (300 disjoint positions)

| Ordering | top-1 | top-3 | mean cp loss |
|---|---:|---:|---:|
| Random legal move | 0.0477 | 0.1382 | 251.7 |
| Static heuristic | 0.1733 | 0.2933 | 245.3 |

| Exit | top-1 | top-1 95% CI | top-3 | mean rank | MRR | cp loss (mean) | cp loss (p90) | regret MAE | WDL Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2/128 | 0.2033 | [0.162, 0.252] | 0.3867 | 9.797 | 0.348 | 141.2 | 473 | 0.5134 | 0.8880 |
| 4/192 | 0.2300 | [0.186, 0.281] | 0.4067 | 9.257 | 0.372 | 137.1 | 467 | 0.5208 | 0.9161 |
| 8/256 | 0.2433 | [0.198, 0.295] | 0.5133 | 6.113 | 0.424 | 117.2 | 412 | 0.7120 | 0.9195 |

The two corpora agree closely at the full exit (top-1 0.255 vs 0.243, cp loss
118.7 vs 117.2), so these are stable estimates, not one lucky sample.

## What the numbers mean

**The policy carries real signal.** Full-exit top-1 of 0.255 against a random
baseline of 0.050 is a ~5x lift, and the 95% confidence interval [0.222, 0.291]
is nowhere near the random floor. Mean teacher-best rank of 6.9 out of 30 legal
moves, and MRR 0.407, say the same thing.

**It also beats the free alternative, which is the question that actually
matters.** A root-ordering hint is only worth an ~11ms forward pass if it beats
what an engine can compute for nothing. Against a static MVV-LVA + promotion +
check ordering:

- top-1: 0.255 vs 0.157;
- mean centipawn loss of the top choice: **118.7 vs 260.2**;
- paired McNemar on the same 600 positions: model-only-correct 79,
  heuristic-only-correct 20, **p = 1.8e-9**. Replication: 27 vs 6, p = 3.2e-4.

So the answer to "is this informative enough to be worth SPRT-testing at all?"
is **yes** — that question is now settled with evidence rather than assumed.

**But the model is nowhere near strong enough to trust as a move chooser.** It
picks the teacher's best move roughly a quarter of the time, and when it is
wrong it is often badly wrong: p90 centipawn loss is 422 at the full exit. This
is a *move-ordering prior*, not an evaluator, and nothing here supports using it
for anything else.

**The elastic exits are not interchangeable.** The shallow 2/128 exit — the one
the integration trial actually used, because it is the cheapest — is
meaningfully worse than the full exit (top-1 0.185 vs 0.255, cp loss 156.6 vs
118.7). Any future SPRT must state which exit it tested; a good full-exit result
would not transfer to the shallow exit that the current trial harness uses.

**The auxiliary heads are weak.** WDL Brier of 0.87–0.92 against a three-class
target is poor (uniform guessing scores 0.667), so the evidential value head is
worse than useless as calibrated here and should not gate anything. Regret MAE
is *worse* at the full exit (0.726) than at the shallow one (0.521), which is
the opposite of the expected ordering and suggests the regret head's calibration
does not transfer to this distribution. Neither head is ready for use.

## What this does not establish

- **No Elo claim.** Ordering quality is not strength. Round 0 failed at 0-20-0
  with a hint that also looked reasonable in isolation.
- **No latency claim on deployment hardware.** Unchanged from prior rounds; the
  target CPU is still not known or reachable from this environment.
- **Not a training-membership proof.** Source-population disjointness only, as
  stated above.
- **Teacher depth is finite.** 400k nodes is a strong reference but not ground
  truth; a handful of positions have genuinely close alternatives where "the
  best move" is somewhat arbitrary. This affects top-1 far more than centipawn
  loss, which is why both are reported.

## Reproduction

```bash
# 1. Build the corpus from over-the-board PGN archives.
python3 tools/build_unarchitectured_v1_calibration_corpus.py \
    --pgn twic900.pgn --pgn twic901.pgn \
    --output artifacts/unarchitectured-v1-calibration-corpus.jsonl \
    --positions 600

# 2. Label with a real teacher and score every exit.
python3 tools/calibrate_unarchitectured_v1_policy.py \
    --corpus artifacts/unarchitectured-v1-calibration-corpus.jsonl \
    --engine /path/to/stockfish --nodes 400000 \
    --labels artifacts/unarchitectured-v1-calibration-labels.json \
    --report artifacts/unarchitectured-v1-calibration-report.json
```

Teacher labels are cached, so re-scoring the model after a runtime change is
fast and does not require re-running Stockfish (pass `--labels` without
`--engine`).

Requires NumPy, PyTorch, `python-chess`, and a UCI teacher binary. The
sandbox used Stockfish 17.1 built from source; `--help` works on all three
scripts without those dependencies installed.

## Next blockers

Step 1 is now answered. The remaining gates are unchanged, and the ordering
below reflects that calibration came back positive:

1. **Broad integrated depth/NPS** (round 6 step 4) using the default-off UCI
   candidate over a corpus this size, not the 8-position trial — this is the
   step that would have caught round 0 in isolation. The corpora committed here
   are directly reusable for it.
2. **Deployment-CPU latency** (step 2), still owner-dependent.
3. **Wider mate/only-move safety coverage** (step 3), which can now draw on real
   positions where the hint and the teacher's best move actively disagree —
   those disagreements are recorded in the committed reports.
4. **Paired-game SPRT** (step 5) via the existing
   `scripts/sprt-history/sprt_unarchitectured_v1_hint.sh`, testing the **full
   8/256 exit**, since the shallow exit calibrates materially worse.

The candidate stays default-off until those pass.
