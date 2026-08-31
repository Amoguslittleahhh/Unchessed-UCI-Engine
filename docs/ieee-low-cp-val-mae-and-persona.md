# Centipawn Validation Loss Floors for NNUE Training under Live Persona Adaptation in a From-Scratch UCI Engine

**Unchessed AI technical report TR-2026-08-31**  
IEEE-style manuscript (not a journal submission). All Elo numbers are SPRT measurements already committed in this repository unless labelled as simulation.

---

**Abstract**—We analyse the Unchessed UCI engine end-to-end and ask whether NNUE validation mean absolute error (val-MAE) can be driven below 10–20 centipawns (cp) *without* disabling the adapter persona (MATCH / PUNISH / CLINCH / DEFEND). Using three already-published SPRT diagnostics (best val-MAE 51.1 cp at 27 M unique positions, −307 Elo vs the shipped net), a 100-epoch HalfKA rerun (best 53.6 cp), and 80 000-sample Monte Carlo experiments of the Bayes error under the trainer’s actual WDL^{2.5} objective, we show: (i) the current 5 000-node HCE labelling process has a Gaussian Bayes floor ≈ 56 cp, so sub-20 cp val-MAE is *structurally unreachable* on those labels; (ii) log-linear extrapolation of measured best-MAE vs unique count forecasts 49.6 cp at 108 M and 48.7 cp at 178 M, and would require ~10^{15} unique positions to hit 20 cp — data scale is the wrong lever; (iii) the only recipes that cross 20 cp (and 10 cp) in simulation reduce *label noise* (stronger teacher, quiet filters, optional clipping), not epoch count; (iv) persona mode selection is a search-side finite-state machine on `eval_cp` and is independent of the training loss. Lower eval MAE *stabilises* MATCH (mode-flip rate 30.5 % at 80 cp MAE → 4.5 % at 10 cp MAE; MATCH retention 62.8 % → 94.5 %). Adaptive stays on. `UnarchitecturedHint` stays default-off.

**Index Terms**—NNUE, centipawn MAE, Bayes error, quiet-position filtering, human-like chess engines, SPRT, persona adaptation.

---

## I. Introduction

Unchessed is a from-scratch Rust UCI engine with two binaries sharing `unchessed-core`: the Game Adapter (live opponent-Elo model + personas) and the Game Reviewer (raw strength). Evaluation is an NNUE (`unchessed-nnue.bin`, format UNCHNNUE v4) trained by `tools/train_nnue.py`. The adapter’s personas are implemented in `unchessed-core/src/adapt.rs` and are **selection policies over MultiPV lines**, not a second evaluator.

The practical question is not “can we print a small number.” It is:

> Can we train an evaluator whose *validation* error versus search labels is 10–20 cp, while the shipped Adaptive persona remains on and the engine remains a usable UCI opponent?

This report answers that from the committed codebase, committed SPRTs, the literature already triaged in `docs/research-survey-arxiv-2026-08-24.md`, and new stdlib Monte Carlo experiments (`tools/experiment_low_cp_val_mae.py`, artifacts in `artifacts/low-cp-val-mae-persona-experiments.json`). This sandbox has no PyTorch and no GPU; we therefore do **not** claim a newly trained 10 cp net. We claim a measured *floor*, a measured *persona coupling*, and a go/no-go recipe.

### A. Contributions

1. Isolation of val-MAE as a *label-noise* problem, not a capacity or epoch-count problem, using the project’s own 0.96 M / 9 M / 27 M SPRT ladder [1].
2. Demonstration that the trainer’s WDL^{2.5} loss is not the reported val-MAE (objective mismatch).
3. Simulation of seven labelling recipes; only σ ≲ 20 cp teachers hit sub-20 cp MAE.
4. Persona sensitivity: lower eval MAE reduces accidental mode flips; Adaptive is complementary, not competitive, with a stronger net.
5. Explicit NO-GO for “more epochs / more of the same HCE labels” as a path to 10–20 cp.

---

## II. Related Work

### A. NNUE training and datasets

Nasu’s NNUE and Stockfish’s HalfKAv2_hm remain the production reference [2]. Tan & Watkinson Medina, *Study of the Proper NNUE Dataset* (arXiv:2412.17948) [3], is the Grade-A paper in this repo’s survey: an NNUE is a *static* function, so labels from tactical positions are noise the net cannot represent. Their M1 (`|static − qsearch|`) and M2 (`|static − search|`) filters are already implemented in `unchessed-datagen` with this engine’s measured 60/70 cp margins [4]. Klein’s *Neural Networks for Chess* (arXiv:2209.01506) [5] documents WDL blending and the p=2.5 power loss that `wdl_loss()` copies.

ChessBench (Ruoss et al., arXiv:2402.04494) [6] shows even a 270 M transformer cannot perfectly distill Stockfish 16 search; “perfect distillation is still beyond reach.” That is the same ceiling Unarchitectured v1 hit (top-1 0.255).

### B. Human-aligned play

Maia [7] and Allie (arXiv:2410.03893) [8] condition policy on rating. Otter (arXiv:2608.05206) [9] adds clocks and history. Unchessed’s adapter is *not* a Maia clone: search always runs full strength; persona only *selects* among MultiPV (and, below 2200, a probed legal pool). That architectural split is why eval training and persona can be improved independently.

### C. Project-internal measurements (real SPRTs)

| Unique positions | Best val-MAE | Final val-MAE | SPRT vs shipped net |
|---|---|---|---|
| 959 102 | 57.4 cp | 83.6 cp | **−796.5 Elo** |
| 9 000 000 | 55.3 cp | 59.3 cp | **−383.5 Elo** |
| 27 000 000 | 51.1 cp | 54.3 cp | **−307.1 Elo** |

Shipped NNUE vs HCE: **+107.1 ± 27.0 Elo** (532 games, LOS 100 %). Incremental accumulators: **+68.6 ± 21.0 Elo** (657 games). HalfKA v3 vs v1: **−70.3 ± 22.1 Elo** (H0, 756 games); 100-epoch rerun best val-MAE **53.6 cp** then overfitting [10].

Round 13 fixed last-checkpoint export (every diagnostic net had been *worse* than its own best epoch) [1], [11]. That fix cannot create a 20 cp net from 50 cp labels.

---

## III. System Under Study

### A. Evaluator (`tools/train_nnue.py`)

- Features: HalfKAv2_hm, 32 king buckets, horizontal mirroring, own-king active, factorized virtual table at train time, coalesced `[22528 × 256]` export.
- Head: SCReLU, concat STM/NSTM (512), 8 piece-count output buckets.
- Loss: `|σ(raw) − t|^{2.5}` with `t = 0.7 σ(cp/400) + 0.3 (wdl/2)`.
- Reported val-MAE: `|400·raw − score_cp|` — **a different functional**.
- Optimiser: Adam 1e-3, ×0.3 at 60 % and 80 % of the *epoch cap*; early-stop patience 3, min-delta 0.1 cp; export **best** checkpoint.

### B. Persona (`adapt.rs`)

`decide_mode` is a hysteresis machine on `our_eval_cp` (search, not trainer val-MAE):

- DEFEND: enter < −180 cp, hold < −80.
- PUNISH: blunder while better, or huge skill gap with lead; hold > +200.
- CLINCH: fullmove > 28 and |eval| < 60, hold |eval| < 100, contempt > 0.
- else MATCH. Engine-suspect → FULL. `Adaptive=false` → FULL (or MATCH if `UCI_LimitStrength`).

MATCH samples among MultiPV (and a probed legal pool below target 2200) with temperature `max_loss/2` and an explicit blunder mixture. None of this is trained by `train_nnue.py`. Default UCI: `Adaptive=true`, `UnarchitecturedHint=false`.

### C. Label generation

Datagen: 5 000-node HCE search, quiet filters M1/M2, ply gaps, mate/`|score|>2000` reject. The committed git corpus used for the 959 k diagnostic is 0.54 % of the 108 M self-play shards that trained the shipped net; those shards are **not in git**.

---

## IV. Method

### A. Bayes floor

If labels are `y = v + ε`, `ε ~ N(0, σ²)`, the minimum achievable MAE versus `y` for a net that recovers `v` is

```
E[|ε|] = σ √(2/π).
```

A net cannot beat its teacher’s noise. Sub-20 cp val-MAE therefore requires `σ ≲ 25 cp` (and sub-10 cp requires `σ ≲ 12.5 cp`) *on the validation labels*.

### B. Monte Carlo recipes (seed 20260831, N = 80 000)

True values are a two-component Gaussian mixture (quiet vs material). Labels add Gaussian teacher noise. We measure:

- Gaussian Bayes MAE,
- oracle MAE of `v` vs `y`,
- MAE of an affine readout of `v` (capacity is not the bottleneck in 1-D),
- MAE of a *perfect WDL fit* (`raw = logit(t)`) versus the cp label — the trainer’s actual objective.

Seven recipes: HCE-like σ=70; quiet-filter σ=35; deeper teacher σ=20; Stockfish-like σ=12; clip±600 with σ=20; clip±400 quiet σ=12; pathological σ=3 clip±200.

### C. Persona coupling

20 000 synthetic roots. Gold mode from noiseless `eval_cp`; noisy eval with MAE ∈ {5,10,20,50,80,150} cp (Gaussian scaled so E[|ε|]=MAE). Count mode-flip rate and MATCH retention. Adaptive remains true; engine-suspect rate 4 %.

### D. Scaling fit

Ordinary least squares: `best_mae = a + b ln(unique)` on the three SPRT points.

---

## V. Results

### A. Label recipes (simulation)

| Recipe | σ (cp) | Bayes MAE | Affine MAE vs labels | Perfect-WDL MAE vs cp | <20? | <10? |
|---|---|---|---|---|---|---|
| HCE-like (current) | 70 | 55.85 | **55.91** | 21.97 | no | no |
| Quiet filter | 35 | 27.93 | 27.85 | 15.49 | no | no |
| Deeper teacher | 20 | 15.96 | **15.90** | 13.43 | **yes** | no |
| SF-like teacher | 12 | 9.57 | **9.61** | 11.70 | yes | **yes** |
| Clip ±600, σ=20 | 20 | 15.96 | 15.62 | 13.33 | yes | no |
| Clip ±400 quiet σ=12 | 12 | 9.57 | 9.53 | 9.92 | yes | yes |
| Pathological σ=3 | 3 | 2.39 | 2.40 | 7.19 | yes | yes |

**Finding 1.** The HCE-like recipe’s floor (55.9 cp) matches the *best* numbers this project has ever measured (51.1 / 53.6 / 55.3 cp). That is not a coincidence: the nets are already near the teacher’s Bayes floor. More epochs after that *raise* val-MAE (overfit), which is exactly the 100-epoch v3 curve and the three diagnostic last-vs-best gaps.

**Finding 2.** Perfect WDL fit can report *lower* cp MAE than the Bayes floor of the cp labels (21.97 vs 55.91 at σ=70) because `t` is a compressed, WDL-mixed target. Conversely, at very low σ, WDL fit is *worse* in cp (7.19 vs 2.40). **You cannot steer toward 10–20 cp val-MAE with the current loss and the current metric at the same time.** Direct L1/Huber on cp is the objective that actually minimises the number on the log line.

**Finding 3.** Clipping shrinks MAE by deleting the tails the persona *uses* (PUNISH at +250, DEFEND at −180). Clip±200 as a cheat to 2.4 cp MAE would blind those thresholds. Clip±400 is the largest clip that still leaves DEFEND/PUNISH on-scale.

### B. Data scaling cannot buy 20 cp

Fit: `best_mae = 81.924 − 1.748 ln(unique)`.

| Corpus | Forecast best val-MAE |
|---|---|
| 108 M (local shards) | 49.59 cp |
| 178 M (cloud) | 48.72 cp |
| 500 M | 46.91 cp |
| 2×10⁹ (nnue-pytorch superbatch scale) | 44.49 cp |
| Unique count for 20 cp | **2.43×10¹⁵** |
| Unique count for 10 cp | **7.43×10¹⁷** |

The 9 M → 27 M step recovered only 4.2 cp of *best* MAE and 76 Elo of a 383 Elo gap. Cloud 178 M with the same HCE labels remains **NO-GO** for a 10–20 cp target (and remains NO-GO for beating the shipped net until a 108 M best-checkpoint SPRT exists [11]).

### C. Persona stays on — and gets *more* reliable as MAE falls

| Eval MAE (cp) | Mode-flip rate | MATCH retention |
|---|---|---|
| 5 | 0.0238 | 0.9716 |
| **10** | **0.0454** | **0.9452** |
| **20** | **0.0916** | **0.8883** |
| 50 (≈ current best nets) | 0.2158 | 0.7393 |
| 80 (last-ckpt 959 k net) | 0.3051 | 0.6277 |
| 150 | 0.4424 | 0.4730 |

**Finding 4.** There is no trade-off that requires turning Adaptive off to train a low-MAE net. The persona FSM does not appear in the graph, the loss, or the export format. A 10 cp evaluator would *reduce* accidental DEFEND/PUNISH/CLINCH transitions by ~6–7× versus an 80 cp evaluator, which is the opposite of “persona dies when the net gets accurate.” MATCH still *intentionally* plays suboptimal moves via temperature and the blunder mixture; that behaviour is policy, not eval error.

Engine-suspect and `UCI_LimitStrength` paths are unchanged. `UnarchitecturedHint` remains default-off: rating input still inert (0/200), GAB undersized, hint p90 cp-loss 422 [12]. A stronger NNUE does not fix the policy prior.

### D. What “proven real-world” means here

Proven in this repo, on real games, not simulation:

- NNUE v1 **+107.1 Elo** vs HCE (SPRT).
- Incremental NNUE **+68.6 Elo** (SPRT).
- HalfKA v3 **−70.3 Elo** vs v1 (SPRT); overfitting after epoch ~13.
- Data-scale ladder **−796 / −383 / −307 Elo** vs shipped net (SPRT, tc=10+0.1).
- Quiet-filter rejection rates on a real PGN (21.5 % M1, 16.5 % M2) [4].
- Persona hysteresis and MATCH-never-into-mate are unit-tested in `adapt.rs`.

Proven in this session (simulation, seed 20260831): Bayes floors, WDL/cp mismatch, scaling extrapolation, persona flip rates. Not proven: a trained UNCHNNUE file with val-MAE < 20 cp — **no such file exists in the tree**, and this host cannot train one.

---

## VI. Recommended Recipe (if and only if a strong teacher is available)

Keep Adaptive on. Do not touch `UnarchitecturedHint`.

1. **Relabel**, do not re-epoch. Teacher must have label noise σ ≲ 20 cp on quiet positions (Stockfish / self-distill of the shipped net at high node count — not 5 000-node HCE).
2. **Keep M1=60 / M2=70** quiet filters [3], [4]. Optional clip at ±400 cp, not tighter.
3. **Train L1 or Huber on cp** *in addition to* (or instead of, under SPRT) WDL^{2.5} if the KPI is val-MAE. If the KPI is Elo, keep WDL^{2.5} and stop pretending val-MAE is the training objective.
4. **Early-stop on the KPI you care about**, export best (already implemented).
5. **SPRT vs `unchessed-nnue.bin`** at tc=10+0.1, elo0=0, elo1=10, Adaptive on for a *persona-on* gate and Adaptive off for an *eval-only* gate. Two gates; one change.
6. **Do not** spend cloud 178 M on the current labels to chase 10–20 cp.

A 10–20 cp val-MAE net that was bought by clipping or by training on a tiny homogeneous set will fail the eval-only SPRT and will starve DEFEND/PUNISH of the eval range they need. That is the failure mode to refuse.

---

## VII. Threats to Validity

- Teacher σ in Section V-A is a modelling assumption, not a measurement of 5 000-node HCE vs Stockfish on this engine. The *match* between σ=70 Bayes (55.9) and measured best MAE (51–54) is the calibration; a different teacher would need its own residual study.
- The log-linear fit uses three points. Diminishing returns make the 10^{15} unique-count for 20 cp a *lower* bound on pessimism: the real curve is flatter.
- Persona flip rates use Laplace/Gaussian eval noise, not the actual NNUE residual field (which is correlated across similar positions). Correlated errors could flip modes in streaks rather than i.i.d.
- No new SPRT was run this session (no `cargo`, no cutechess in the sandbox).

---

## VIII. Conclusion

Extremely low NNUE val-MAE (10–20 cp) is not a hyperparameter hunt on the current corpus. It is a **teacher-noise** requirement. The shipped training stack is already sitting on a ~50–56 cp floor. Persona adaptation is orthogonal and **benefits** from a more accurate eval; Adaptive should stay on. The Unarchitectured v1 hint is a different, still-blocked subsystem and must stay off. The honest production next step remains the round-13 108 M best-checkpoint SPRT for Elo, and a *new labelling campaign* if and only if someone needs the 10–20 cp number for its own sake.

---

## References

[1] Unchessed AI, “NNUE v4 retrain: data-scaling root cause,” `docs/nnue-v4-retrain-data-scaling-finding.md`, 2026-08-30.

[2] Y. Nasu, “Efficiently updatable neural-network-based evaluation functions for computer shogi,” 2018; Stockfish `half_ka_v2_hm`.

[3] D. Tan and A. Watkinson Medina, “Study of the Proper NNUE Dataset,” arXiv:2412.17948, 2024.

[4] Unchessed AI, “NNUE dataset quiet-position filters,” `docs/nnue-dataset-quiet-filters.md`.

[5] D. Klein, “Neural Networks for Chess,” arXiv:2209.01506, 2022.

[6] A. Ruoss et al., “Grandmaster-level chess without search,” arXiv:2402.04494, 2024.

[7] R. McIlroy-Young et al., “Aligning Superhuman AI with Human Behavior (Maia),” KDD, 2020.

[8] “Human-aligned Chess with a Bit of Search (Allie),” arXiv:2410.03893, 2024.

[9] “Otter: Time-Aware, History-Conditioned Human Chess AI,” arXiv:2608.05206, 2026.

[10] Unchessed AI, “NNUE v3 (HalfKA) Regression — Research Brief,” `nnue-shards-safe/v3_research_brief.md`.

[11] Unchessed AI, “NNUE v4 full-scale training recipe,” `docs/nnue-v4-training-recipe.md`, 2026-08-30.

[12] Unchessed AI, “Why the Unarchitectured v1 hint costs Elo,” `docs/unarchitectured-v1-why-the-hint-costs-elo.md`.

[13] Unchessed AI, “arXiv survey — chess AI and LLM research,” `docs/research-survey-arxiv-2026-08-24.md`, 2026-08-24.

[14] Experiment dump: `artifacts/low-cp-val-mae-persona-experiments.json`; harness: `tools/experiment_low_cp_val_mae.py`.

---

## Appendix A — Reproducing the simulation

```
python3 tools/experiment_low_cp_val_mae.py
```

Stdlib only. Seed 20260831. Writes the JSON cited above.

## Appendix B — Code map

| Concern | Location |
|---|---|
| WDL^{2.5} loss, val-MAE, HalfKAv2_hm | `tools/train_nnue.py` |
| Early-stop / LR cap | `tools/nnue_train_control.py` |
| Personas, MATCH sampling | `unchessed-core/src/adapt.rs` |
| NNUE inference | `unchessed-core/src/nnue.rs` |
| Quiet filters | `unchessed-datagen`, `docs/nnue-dataset-quiet-filters.md` |
| Shipped eval | `unchessed-nnue.bin` |
