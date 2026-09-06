# 05 — Oracle-side Rating-Conditioning Experiment

**Investigation status: BLOCKED on the missing oracle checkpoint; implementation path READY.** This review was performed on branch `manus/rustc-bootstrap-trial` at commit `818ef9dd5bb7be64fd6085f7c1910b953390da6e`. No tracked repository file was changed, no model training or SPRT was run, and this report does not claim an oracle result.

## Required structured fields

| Field | Value |
|---|---|
| **ID** | `05-oracle-rating-conditioning` |
| **Topic** | Oracle-side rating-conditioning experiment |
| **Summary** | The repository contains a reproducible 200-position, seven-rating student diagnostic and the Python definition/checkpoint contract of the original 58.4M offline oracle. It does **not** contain an oracle checkpoint or an oracle-compatible sweep executable. The current student-only analyser cannot be pointed at an oracle `.pt` file. A small, offline analysis tool can be added without touching engine behavior; it should fail with exit code 2 and an explicit missing-checkpoint message before importing Torch when the required file is absent. |
| **Implemented-ready work** | Implement the isolated oracle analyser specified below, reusing the frozen corpus, shared FEN encoder, and metrics schema. It requires only a trusted legacy oracle checkpoint plus a Python environment with `torch`; it performs inference only. |
| **Blocked work** | No `UNARCHV1_ORACLE_TRAINING_V1_DDP` checkpoint is present in the repository. A scan found only the exported 4,277,712-byte runtime student (`artifacts/unarchitectured-metal-final.unmetal`) and an unrelated NNUE binary; no oracle-named checkpoint was found under `/home/ubuntu` outside caches. This sandbox also lacks the optional `torch` package. Therefore the proposed sweep cannot be executed here and no oracle conclusion is available. |
| **Requires real SPRT** | The diagnostic itself does **not** require an SPRT: it is an offline causal-location measurement. Any retrain that follows either branch, and especially any integration or default change reachable in games, still requires the project’s real cutechess/pentanomial SPRT gate before a strength claim or behavior promotion. |

## Evidence ledger

### What is measured already — the exported student only

The committed student result is a genuine negative result, not an oracle experiment. The stored 200-position sweep measures the exported `UNARCHV1` runtime package at ratings 600, 1000, 1400, 1800, 2200, 2600, and 3200. It records **zero top-1 changes at every rating**, an endpoint maximum legal-logit shift of `0.00411534309387207`, and identical `0.265` external-label agreement. The separate fixed-1500 `POLICY_HUMAN`/`POLICY_GUIDE` comparison changes top-1 on 4 of 200 positions. The committed tests check the breadth of the sweep, the zero-change claim, non-zero plumbing deltas, monotonicity, and the documentation distinction between neural policy conditioning and the still-functional adaptive `UCI_Elo` path.[1][2]

The result is therefore strong evidence that the **student’s shipped policy hint** is effectively rating-invariant. It is not evidence that the teacher/oracle is invariant. The existing finding explicitly identifies the oracle sweep as the deciding experiment and says the checkpoint is absent.[1]

One reporting detail should be preserved accurately in an oracle implementation. The current analyser appends the **maximum** absolute legal-logit difference for each position, then reports its arithmetic mean as `mean |dlogit|`; it does not average every individual legal-logit difference.[2] The new output should retain this historical-comparability metric under the unambiguous name `mean_position_max_abs_delta`, and may additionally report `mean_abs_delta_all_legal` if desired.

### Why the current analyser cannot run the oracle

`tools/analyse_rating_conditioning.py` is specifically a runtime-student package tool. It calls `read_package()` from `reference_forward_unarchitectured_metal.py`, whose binary reader requires magic `UNARCHV1`, then runs the 8-layer/256-wide exported student reference forward pass.[2][3] It accepts only `package` and `corpus` positionals, does not load a PyTorch checkpoint, and its FEN batch has no move-history tensors because that exported reference implements the student layout.

The original offline oracle is materially different. `UnarchitecturedV1Oracle` is a 16-layer, 512-wide board trunk with a four-layer legal-action decoder; it takes `history`, `history_len`, `rating`, `time_class`, `policy_kind`, padded `safe_actions`, and `legal_mask`, and returns policy logits over legal-action slots.[4] Its trainer saves a PyTorch mapping with format `UNARCHV1_ORACLE_TRAINING_V1_DDP`, `config`, `model`, `optimizer`, metrics, and shard manifests.[4] The saved checkpoint is not a `UNARCHV1` export and cannot safely be fed to the student package reader.

A later pretraining path is also not interchangeable with this original oracle. It emits `UNARCHV1_PRETRAIN_DUAL_ELO_V1` and uses `UnarchitecturedV1OracleDualElo`, whose input contract adds `elo_oppo` and whose state dictionary replaces the scalar rating vectors with dual-Elo parameters.[5] The initial oracle-side experiment should either support only the legacy format explicitly or reject the dual-Elo format explicitly; silently instantiating the wrong model would invalidate the measurement.

### Inputs available and missing

The calibration corpus is present and is appropriate to reuse unchanged: `artifacts/unarchitectured-metal-calibration-corpus.jsonl` contains a manifest plus 600 deduplicated FENs, stratified overall as 200 opening, 200 middlegame, and 200 endgame positions. It was sampled from over-the-board tournament PGN, source-disjoint—but not record-proven disjoint—from the Lichess training population. The matching labels JSON contains 600 FEN-keyed score maps. The existing analyser already skips the manifest row and selects the first 200 FEN records at `--limit 200`.[2][6]

The rating-variation hypothesis has been narrowed independently. The pretrain data bridge stores both `elo_self` and `elo_oppo` per move, taking them from self-play labels or WhiteElo/BlackElo headers; thus “the student never saw rating variation” is not sufficient to explain the student result.[7] This does **not** establish that the oracle’s policy targets varied with rating, which is exactly why an oracle inference sweep is still needed.

The original training configuration specifies the expected 58,412,431-parameter oracle dimensions (`d_model=512`, 16 board layers, four decoder layers, `history_width=64`) and a scalar rating path through `rating_weight`/`rating_bias` in the history context.[4][8] The checkpoint itself is absent. Repository inventory found no `.pt`, `.pth`, `.ckpt`, `.safetensors`, or oracle-named artifact; the existing package and NNUE files are incompatible substitutes and must not be used. The local Python environment has `chess` and `numpy` sufficient for existing non-Torch tests, but `import torch` fails with `ModuleNotFoundError`.

## Implementation-ready, offline experiment path

### Scope and command contract

Add a **new analysis-only** tool named `tools/analyse_oracle_rating_conditioning.py`; do not overload or weaken the student package analyser. It should require `--oracle` and `--corpus`, accept optional `--labels`, and default to the exact canonical settings below.

```bash
python3 tools/analyse_oracle_rating_conditioning.py \
  --oracle /secure/path/unchessed-v1-oracle.pt \
  --corpus artifacts/unarchitectured-metal-calibration-corpus.jsonl \
  --labels artifacts/unarchitectured-metal-calibration-labels.json \
  --limit 200 \
  --ratings 600 1000 1400 1800 2200 2600 3200 \
  --policy-kind 1 \
  --time-class 2 \
  --json /secure/output/oracle-rating-conditioning.json
```

Before importing `torch` or opening model data, the CLI should check the two required paths. For a missing checkpoint it must emit exactly or equivalently:

```text
missing oracle checkpoint: /secure/path/unchessed-v1-oracle.pt
```

and return **2**, with no JSON result written. A missing corpus must be handled equivalently. If Torch is unavailable after paths validate, it must return 2 with an actionable dependency message, for example `missing dependency: torch; install it on the trusted analysis host`. This is stricter than the current student tool’s generic `missing: <path>` failure, which was confirmed to return 2 for an absent package; its `--help` path succeeds without model dependencies.

### Exact legacy-checkpoint loading contract

The first version should accept only a trusted, original-oracle checkpoint:

1. Load it on CPU using `torch.load(path, map_location="cpu", weights_only=False)` only after path validation. This matches the repository’s own trainer/evaluator contract.[4] Because such loading uses Python serialization, execute only a checkpoint received from a trusted project-controlled source; do not treat an unknown external pickle as data.
2. Require a mapping with `format == "UNARCHV1_ORACLE_TRAINING_V1_DDP"`, a mapping-valued `config`, and a mapping-valued `model`. Reject missing keys or another format with a clear `unsupported oracle checkpoint format` error and exit 2.
3. Instantiate `UnarchitecturedV1Oracle(checkpoint["config"])`, load `checkpoint["model"]` with `strict=True`, move to CPU, call `eval()`, and run under `torch.inference_mode()`. Wrap load/state-dict failures as a clear incompatibility error that identifies the checkpoint path; never fall back to randomly initialized weights.
4. Explicitly reject `UNARCHV1_PRETRAIN_DUAL_ELO_V1` pending a separate, tested two-input experiment design. Its `elo_oppo` requirement changes the experimental question and it must not be silently mapped onto the scalar-rating sweep.[5]

### Per-position batch construction and measurements

For each of the first 200 corpus FENs, call the shared `encode_position()` helper. It is already the repository’s exact mover-relative encoder for pieces, castling, en-passant, halfmove bucket, legal actions, and legal-move mapping.[9] Set a zero move history of shape `[7, config["history_plies"]]` and `history_len=0`, repeat the encoded board and padded legal action array for the seven ratings, set `time_class=2` and `policy_kind=1`, and call the oracle once for the seven paired rows. Zero history is necessary because the frozen FEN corpus has no preceding move sequence; holding it constant makes it a valid test of whether the **rating input alone** changes the oracle’s policy.

For each rating, compare legal-slot logits with rating 600 on the same FEN. Report: top-1 changes; mean of each position’s maximum absolute legal-logit delta; maximum absolute legal-logit delta; and, when `--labels` is supplied, agreement with its fixed best UCI move. Preserve the separate policy-kind comparison at rating 1500 (`0` versus `1`) to remain measurement-compatible with the established student report. The label agreement is auxiliary: it uses the fixed calibration score labels, not an oracle-specific truth set, and is not needed to establish a causal rating response.

The JSON should include a schema/version, `status: "completed"`, SHA-256 and format of the checkpoint, SHA-256 of the corpus, `positions_requested`, `positions_used`, ratings, fixed inputs, and all raw metrics. It must not write a positive/negative conclusion based on an unstated flip threshold. A zero/non-zero outcome can then be interpreted using the repository’s already committed decision procedure rather than a threshold invented after observing the result.[1]

### Decision use after a completed run

| Observed oracle result | Permitted conclusion | Next work | What it does **not** prove |
|---|---|---|---|
| Material rating response in the raw top-1/logit metrics | The teacher contains rating-varying information that the student path/translation did not preserve. | Prioritize the planned student-side conditioning redesign and distillation path; the documented dual-Elo design is a candidate. | No Elo gain and no authorization to wire it into search. |
| No top-1 response plus only negligible logit movement over the span | The evidence points upstream: the original oracle policy is itself rating-invariant on this probe. | Establish conditioning in oracle architecture/training data before spending on a student-only retrain. | Universal invariance across all positions or time/history contexts. |
| Checkpoint/model incompatibility or corrupted data | No model conclusion. | Repair provenance/format or add an explicitly tested loader. | Either scientific branch. |

These branches restate the repository’s documented decision procedure; they are not results from this review.[1] The report should preserve raw values and provenance so the next investigator can judge effect size, rather than merely emitting a narrative verdict.

## Verification performed and acceptance checks

### Performed in this investigation

The existing `python3 -m unittest tools/test_rating_conditioning.py -v` completed **15/15** tests successfully. `python3 tools/test_build_level_conditioned_moves.py` completed **6/6** named checks successfully, including its real-block smoke run. The existing analyser’s `--help` exited 0 without Torch, and invoking it with an absent package returned 2 and `missing: /tmp/absent-oracle.pt`. These validate the current student diagnostic and the availability/format of the supporting rating-labelled tooling; they do not execute an oracle.

The repository worktree remained clean. No Torch model forward, training, match, or SPRT was attempted. The absence of an oracle checkpoint and Torch prevents even an inference-only oracle smoke test in this sandbox.

### Required tests when the proposed tool is implemented

| Gate | Expected outcome |
|---|---|
| `--help` on a bare environment | Exit 0 without importing Torch or loading a checkpoint. |
| Absent `--oracle` | Exit 2; explicit `missing oracle checkpoint: ...`; no report file. |
| Absent `--corpus` | Exit 2; explicit missing-corpus message; no report file. |
| Unsupported `UNARCHV1` runtime package or dual-Elo checkpoint | Exit 2; identifies unsupported format rather than interpreting it as legacy oracle weights. |
| Tiny trusted legacy-oracle fixture | Strict load succeeds; one legal FEN produces seven legal-logit vectors and a provenance-complete JSON report. |
| Sensitivity fixture | A controlled non-zero scalar rating path changes at least a reported logit delta, proving the tool supplies the swept tensor rather than merely changing a printed label. |
| Corpus/encoder regression | Assert the manifest is skipped, exactly 200 FENs are used under defaults, each has 2–218 legal moves, and legal-slot top-1 maps back to an actual legal UCI move. |
| Reproducibility | Two CPU runs over the same trusted checkpoint/corpus produce identical discrete metrics and stable float output within a declared tolerance. |

A completed sweep is a **diagnostic gate only**. It can choose between oracle-first and student-first retraining work without an SPRT because it neither changes production code nor claims playing strength. The repository’s standing policy still requires a fresh real cutechess SPRT for any behavior reachable from games before a default flip, and the rating finding states that every retrained net needs its own SPRT.[1][10]

## Boundaries and caveats

The proposed FEN probe deliberately holds history, time class, and policy kind fixed, so it isolates the scalar rating input but does not characterize interactions with game history. Its 200 positions are a subset of the 600-position corpus, and source-disjointness is a population-level claim because no full training-membership manifest exists.[6] These are appropriate limits for locating the current defect, but they should be recorded in the eventual JSON and result document.

No production behavior changes under this plan. The offline oracle is documented as never loaded by the chess engine, and the existing runtime hint is default-off; a new tool and its tests remain in `tools/` only.[4][1] Do not substitute the exported student package, the NNUE checkpoint, a synthetic self-check model, or a dual-Elo pretrain artifact for the missing legacy oracle. Each would answer a different question.

## References

[1]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/rating-conditioning-finding.md "Student finding and committed oracle-side decision procedure"
[2]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/analyse_rating_conditioning.py "Current student-only sweep implementation"
[3]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/reference_forward_unarchitectured_metal.py "UNARCHV1 reader and student reference forward pass"
[4]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/train_unarchitectured_metal_a100.py "Original offline oracle interface and checkpoint serialization"
[5]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/pretrain_v1_a100.py "Dual-Elo oracle lineage and checkpoint contract"
[6]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/build_unarchitectured_metal_calibration_corpus.py "Calibration corpus provenance and construction"
[7]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/pretrain_move_dataset.py "Dual-Elo data bridge"
[8]: file:///home/ubuntu/Unchessed-UCI-Engine/config/unarchitectured_metal_training.json "Original oracle configuration"
[9]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/unarchitectured_metal_position_encoding.py "Shared FEN-to-model-input encoding"
[10]: file:///home/ubuntu/Unchessed-UCI-Engine/scripts/research/arena_agent_unarchitectured_metal_runtime_speed_prompt.md "Standing real-SPRT rule for game-reachable behavior"
