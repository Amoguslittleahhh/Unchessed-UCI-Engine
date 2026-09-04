# Research note — "Checkmating One, by Using Many: Combining Mixture of Experts with MCTS to Improve in Chess" (arXiv:2401.16852)

Assessment of Helfenstein, Blüml, Czech, Kersting (TU Darmstadt), v2
2024-02-10 (v3 2025-06-17), for relevance to this engine.

## What the paper is

Not a new architecture: three standard AlphaZero-style CNN experts
(RISEv3.3 / CrazyAra, value + policy heads, 52-plane input including the
last 8 moves), one per game phase, selected by a **hand-crafted Lichess
phase definition** (no learned gate), integrated into MCTS (PUCT). The
contribution is the combination, plus a controlled comparison of how to
*train* the experts (separated / staged / weighted loss regimes).

Key facts (all from the paper):

- **Data:** Kingbase Lite 2019, 1,112,647 games (players ≥2200 Elo),
  91.4M positions; split by phase: opening 26.2M / middlegame 36.3M /
  endgame 28.9M positions (Table 3).
- **Training:** 8× V100, batch 2048, 7 epochs, NAG, lr 0.14→1e-5; TensorRT
  8.4.1 inference; CrazyAra 1.0.4.
- **Matches:** 1000 games per configuration, 500 random ianfab `chess.epd`
  openings each played from both sides, cutechess, 55 search configs per
  method (batch 1/8/16/32/64 × nodes 100–3200 or movetime 100–1600 ms),
  Elo with 95% CI following the cutechess method.

## Results to remember

| Training regime (vs. the authors' own one-for-all baseline) | Elo gain, average over 55 configs |
|---|---|
| **Separated learning** | **+122.20** (106.89 @ batch 1 → 129.49 @ batch 64) |
| Staged learning | +121.11 |
| Weighted learning, a=10 | +55.84 |
| Weighted learning, a=4 | +23.18 |

- The **middlegame and endgame experts carry the gain comparably; the
  opening expert is neutral to slightly harmful** (training data lacks
  diversity; test openings were computer-randomized while training
  openings were human — §3.2).
- **The phase definition matters**: Lichess's mixedness/backrank rules
  beat move-counter splits (2–5 partitions) substantially, and the paper
  concludes "simple features, like the move counter in chess, alone are not
  a sufficient game phase definition" (§3.3).
- Majority-vote batch routing is robust up to batch 64; behavior beyond
  512 unknown. All experts must reside in VRAM (immaterial at 3 experts).
- Self-declared limitations: overfitting susceptibility on smaller
  datasets; hand-crafted gate (a learnable gate is future work); trained on
  human GM games, not self-play.

## The Lichess phase definition (the part we can actually use)

Ported verbatim into `tools/measure_game_phase_definitions.py` from
lichess-org/scalachess `core/src/main/scala/Divider.scala` (accessed
2026-08-26; the paper's v2 link predates the repo reorganization into the
`core/` subdirectory, which v3 corrected):

- **endgame** iff majors-and-minors (queens, rooks, bishops, knights —
  kings and pawns do not count, either side) ≤ 6;
- **middlegame** iff not endgame and (majors-and-minors ≤ 10, or fewer
  than 4 of your own pieces on your own back rank — king included —, or
  mixedness > 150; mixedness sums a position-dependent score over the 36
  overlapping 2×2 blocks of the board);
- **opening** otherwise.

The gate is **stateless per position** (a game can return to the opening):
that is how the paper uses it in search, and what a training-data split
needs. Note this is the Lichess *analysis* definition; Lichess itself also
forbids backwards transitions when labeling a finished game — the paper
deliberately does not.

## Verdict

**Retrain-gated; no code change from this paper. It adds independent
evidence for the existing phase-specialization retrain item, and fixes the
gate that a future run should use.**

1. **The gain is not collectible here.** +122 Elo is relative to the
   authors' own one-for-all CNN baseline, on Kingbase GM data, with MCTS
   and TensorRT inference, on 8× V100. This engine is classical alpha-beta
   + NNUE eval (single 512-wide output head, v3 format — the bucketed-head
   variant was deliberately dropped from v2) + an optional policy prior
   (Unarchitectured v1). There is no MCTS to gate and no CNN to
   specialize; nothing here changes the current checkpoint, the
   `UnarchitecturedHint` default (still `false`), or `runtime_safety_suite`
   (still `false`).
2. **It supports a retrain backlog item, with a free concrete win
   attached.** `docs/research-notes-moe-2507.11181.md` already argues for
   phase-bucketed output heads on the next NNUE run. This paper is
   independent evidence that phase specialization is worth real Elo *when
   the gate is good*, and its §3.3 is actionable immediately: the gate
   should be the Lichess definition, not material or move count.
3. **Our corpus gate is exactly the class the paper demotes.**
   `tools/build_unarchitectured_v1_calibration_corpus.py::classify_phase`
   buckets by material + ply (`total pieces ≤ 12 or non-pawns ≤ 4` →
   endgame; `fullmove ≤ 12` → opening) — the "simplified framework,
   employing material-based criteria" that the paper compares against (and
   the dkappe/Scorpio line of work it criticizes in §5). The HCE's mg/eg
   interpolation (`unchessed-core/src/eval.rs:626-627`, 0–24 material
   phase) is in the same family.
4. **Measured on our data** (`tools/measure_game_phase_definitions.py`,
   artifact `benchmarks/unarchitectured-v1/game-phase-definitions.json`,
   2026-08-26; 600 corpus + 7 matetrack positions):

   | definition | opening | middlegame | endgame |
   |---|---|---|---|
   | Lichess (full) | 183 (30.15%) | 171 (28.17%) | 253 (41.68%) |
   | Lichess, material-only (no backrank/mixedness) | 259 (42.67%) | 95 (15.65%) | 253 (41.68%) |
   | corpus builder (stored tags) | 200 (32.95%) | 200 (32.95%) | 207 (34.10%) |

   - **76 of 607 (12.5%) positions change phase when the backrank /
     mixedness terms are dropped** — those terms are not decorative on
     real tournament play.
   - The stored corpus tag disagrees with the Lichess definition on **73 of
     600 (12.2%)** positions.
   - All 7 matetrack positions are endgame under all three definitions.
   - Data-side consequence for a phase-specialized retrain: our positions
     are 41.7% Lichess-endgame, and ~12% of current phase labels move
     buckets under the definition the paper validates. The paper's ~91M-
     position corpus is five orders of magnitude larger than ours; its
     overfitting caveat (separated learning on ~30M positions per phase) is
     the honest warning for a much smaller corpus.
5. **If phase specialization ever happens to Unarchitectured v1, the root
   is where it costs nothing.** The hint net runs once per move, so a
   per-position hand-crafted gate is free there, and the paper's design
   shows hand-crafted routing avoids the failure modes (routing overhead,
   expert collapse) that the 2507.11181 note argues against for per-node
   learned gates. But the policy prior has never trended positive in SPRT
   and is default-off, so there is no runtime work to do from this paper.

## Not done

No retraining, no model change, no Rust change (no toolchain in the
sandbox; none required by this note). The paper's Elo numbers are cited,
not reproduced. The Lichess definition was ported from the source the
paper cites; it was not cross-checked against Lichess's live analysis API
(unavailable from the sandbox).

## Files

- `tools/measure_game_phase_definitions.py` — the three phase definitions,
  verbatim Lichess port, CLI, JSON artifact writer. Dependency beyond stdlib:
  `chess` (already in `tools/requirements-dev.txt`); verified to run from a
  fresh clone (2026-08-26).
- `tools/test_game_phase_definitions.py` — 42 tests: hand-computed score-
  table values, threshold boundaries, two real corpus positions in which
  exactly one Lichess term flips the phase (pinned), mirror agreement with
  all 600 stored corpus tags, matetrack uniform endgame, committed-artifact
  consistency, `--help` standalone, dependency scan.
- `benchmarks/unarchitectured-v1/game-phase-definitions.json` — committed
  artifact of the run above.
