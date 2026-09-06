# 04 — Unarchitectured Metal GAB and quantization

**Investigation ID:** 04-unarch  
**Repository / branch:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/rustc-bootstrap-trial` (`818ef9dd5bb7be64fd6085f7c1910b953390da6e`)  
**Scope:** Unarchitectured Metal runtime, `UNARCHV1` package format, student/oracle trainers, exporter, calibration and analysis tools, test suites, and the supplied runtime-speed backlog.
**Disposition:** No tracked repository file was changed. No training, self-play, model forward pass, benchmark, or SPRT was run.

## Required structured fields

| Field | Value |
|---|---|
| `id` | `04-unarch` |
| `topic` | `Unarchitectured Metal GAB and quantization` |
| `summary` | `The shipped calibrated compact student has a real, load-bearing but under-sized GAB and a working int8-weight/int16-activation runtime. Widening GAB or moving activations to int8 cannot be safely obtained by package conversion or kernels: both require a new trained checkpoint, full numerical revalidation, and a real paired-game SPRT before any game-facing/default conclusion. The existing package, calibration corpus, labels, analyses, and default-off safety plumbing make offline candidate evaluation ready once a candidate checkpoint exists; this checkout lacks every Unarchitectured training/oracle/student checkpoint and training shard, and the local Python lacks Torch.` |
| `evidence` | `Inspected the Rust package loader/runtime, PyTorch student and A100 trainers, exporter/reference forward, checked-in GAB and int8 studies, UCI and SPRT runner, package integrity metadata, local asset inventory, and focused Python tests.` |
| `recommended_changes` | `Do not add an int8-activation kernel or widen/convert the shipped package. Correct the stale package-loader comment now if desired; when real retraining assets exist, run a factorial GAB/QAT experiment, constrain weights during training using target-scheme-derived bounds, export each candidate, retain all three named parity gates at 5e-3, and require an explicit full-exit real SPRT while UnarchitecturedHint remains false by default.` |
| `verification` | `UNARCHV1 inspection passed for the shipped 4,277,712-byte package (SHA-256 5fd9…b16d); 28 focused GAB/int8 Python tests passed with 8 Torch-dependent tests skipped; tool help checks passed. The three Rust parity tests were located and preserved but could not execute locally because installed Cargo 1.75 cannot parse the v4 lockfile required by the stable toolchain. No real SPRT/training was attempted.` |
| `report_file` | `/home/ubuntu/reinforcement_reports/04-unarch.md` |

## Executive decision

> **Implemented-ready now:** preserve the current runtime and its int8-weight/int16-activation arithmetic; use the existing package/corpus/labels to reproduce the GAB and quantization diagnostics when a Python environment with Torch is available; and, if making a documentation-only change, correct the stale `unarchitectured_metal.rs` module comment that says runtime readiness is still false even though the complete forward lives in `unarchitectured_metal_runtime.rs`. None of these changes enables the hint or claims Elo.
>
> **Retrain-gated:** widening GAB from the shipped student’s `d1=8, d2=32, d3=32` to the paper-comparable `32,64,64`; quantization-aware weight constraints; and any attempt to run activations at int8. Tensor shapes and learned weights must change together, so neither can be achieved safely by padding, reshaping, transcoding, or a new SIMD kernel on the frozen package.
>
> **Real-SPRT-gated:** all strength conclusions, use of a new checkpoint as a root hint, and especially any default change. `UnarchitecturedHint` remains **default `false`**. The existing history has four non-positive real batches, and the supplied backlog expressly directs that it remain default-off.

## Evidence ledger

### 1. The shipped package is healthy, calibrated, and fixed to the small GAB

A non-mutating `tools/inspect_unarchitectured_metal.py` inspection passed the actual package at `artifacts/unarchitectured-metal-final.unmetal`. It is 4,277,712 bytes, SHA-256 `5fd9fc3fbf47bd2620c2e832e24c98525b59feeea791abf1c7ae32b9d311b16d`, and contains 121 tensors: 75 per-tensor symmetric-int8 sections and 46 f32 sections. Its metadata identifies a calibrated `UNARCHV1_STUDENT_CALIBRATED_V1` checkpoint, records calibration on 344,600 labelled actions at target coverage 0.995, and embeds the actual student configuration. The output package exists; its source `.pt` checkpoint does not.

The package metadata and tensor shapes agree on the deployed student configuration. This is important because the repository’s oracle configuration contains larger numbers that do **not** describe the shipped model.

| GAB dimension | Shipped package / student metadata | `runtime_student` config | Offline oracle config | Paper 5M comparator |
|---|---:|---:|---:|---:|
| Token projection (`d1`) | 8 | 8 | 16 | 32 |
| Compressed context (`d2`) | 32 | 32 | 64 | 64 |
| Templates (`d3`) | 32 | 32 | 64 | 64 |

Direct package evidence is `gab.token_projection: [8,256]`, `gab.compress.weight: [32,512]`, `gab.templates: [32,64,64]`, and eight coefficient matrices `[256,32]`. The checked-in capacity test reads these shapes from the binary header rather than trusting prose, and deliberately fails if a new package widens GAB without the evidence being revisited.[1][2]

The generic `UNARCHV1` container can encode tensor shape, f32, i8, i16, i32, scale, zero point, alignment, per-section CRC, and table/payload CRC.[3][4] That is **not** runtime shape flexibility. `ChessformerWeights::validate_schema` in the Rust runtime requires the exact lengths implied by compile-time `GAB_TOKEN_PROJECTION=8`, `GAB_HIDDEN=32`, and `GAB_TEMPLATES=32`; the forward path also uses fixed-size coefficient/context arrays and loops bounded by those constants.[5] A `32/64/64` export may parse as a generic package but will be rejected by the deployed runtime schema until the runtime and its numerical reference are intentionally updated.

### 2. GAB is demonstrably load-bearing, but “larger is better” remains an untested retraining hypothesis

The committed GAB ablation has unusually good provenance: it uses the real exported package, all 600 positions in the Stockfish-labelled calibration corpus, and counts tied teacher-best moves correctly by zero regret rather than a single arbitrary UCI string. The baseline reconciles with the independent ordering-risk artifact; the focused tests lock that reconciliation down.[2][6]

| Frozen-weight variant | Top-1 | Mean first-move regret | Delta top-1 from baseline |
|---|---:|---:|---:|
| Baseline | 0.2683 | 146.3 cp | — |
| GAB templates zeroed | 0.2100 | 178.5 cp | -0.0583 |
| GAB templates shuffled | 0.2100 | 183.0 cp | -0.0583 |

Removing GAB loses 21.7% of its top-1 accuracy and costs approximately 32 cp of mean first-move regret; preserving the tensor but permuting its learned templates is comparably damaging. This establishes that the model uses the **learned GAB content**, rather than merely benefiting from an arbitrary bias-shaped tensor. It does **not** measure the effect of expanding `d1/d2/d3`: a wider GAB has new parameters with no valid values in the existing package, and training could redistribute capacity elsewhere. The capacity result therefore motivates a retraining ablation, not a runtime-only feature claim.[1]

The existing `tools/analyse_gab_contribution.py` is the appropriate post-export diagnostic for a candidate: it accepts the package, the committed 600-FEN corpus, and the committed full per-move labels, reports shapes read from the candidate itself, and emits JSON. Those three inputs are locally present. It needs Torch, which is not installed in this environment, so it was not rerun here. A future candidate should be compared with the same `rating=2700`, `time_class=2`, and guide-policy setting, while retaining the current artefact as baseline rather than overwriting it.[6]

### 3. The current quantized runtime is already a mixed int8-weight/int16-activation design

The runtime has a working, deliberately conservative quantization boundary. Matrix-heavy package sections retain symmetric `i8` weights; each activation row is dynamically quantized to signed `i16`; products accumulate in `i32`; and activation and weight scales are applied after accumulation. AVX2 kernels are selected where available, with scalar fallbacks. The package loader rejects a quantized section unless it is symmetric int8 with zero point zero, while non-quantized runtime tensors must be f32.[5]

This is not a speculative path. The exporter emits i8 for tensors with at least two dimensions and at least 256 values, using per-tensor symmetric scale `max(abs(x))/127`; smaller tensors remain f32.[7] The current documented complete-output difference between retained-int8 matrices and dequantized matrices is at most `1.21e-4`, with a separate enforced integer-path drift gate of `5e-4`; that is comfortably below the Python-reference parity tolerance.[8]

The `±127/64` comparison in the Stockfish notes should not be misread as a current `UNARCHV1` package-format overflow: the exporter’s per-tensor scale accommodates magnitudes above 1.984. It is evidence that the frozen model was not trained under the much narrower fixed-range constraints needed for a particular int8-friendly activation regime. In fact, the largest observed value is 4.09144 in `square_embedding.weight` (2.06 times that reference bound), and several embeddings have approximately 4–7.5% of values beyond it.[9] It supports a quantization-aware retraining hypothesis; it does not permit post-hoc recovery of the lost resolution.

### 4. Int8 activations are measured and rejected for this checkpoint; a kernel cannot repair the error

The activation study tests five whole-model quantize/dequantize schemes against the same exported weights and the two exact full-exit Python fixtures used by Rust. The project’s gate is maximum logit drift `<=5e-3`.[10]

| Whole-model int8 activation scheme | Start-position drift | 1.e4 e5 drift | Gate result |
|---|---:|---:|---|
| Per-token symmetric | 2.62e-2 | 4.09e-2 | Fail |
| Per-tensor static | 4.60e-2 | 4.84e-2 | Fail |
| Per-channel symmetric | 2.52e-2 | 3.18e-2 | Fail |
| Per-group symmetric, group 32 | 2.41e-2 | 2.76e-2 | Fail |
| Percentile, 99.9 | 7.24e-2 | 6.54e-2 | Fail |

The best whole-model outcome is still 4.8–5.5 times above the gate. This failure was determined before implementation by simulating the exact quantize/dequantize information loss that an integer kernel would see. Replacing float simulation with `maddubs`/similar intrinsics cannot recover precision absent from an 8-bit activation representation.[10]

The obvious mixed-precision escape hatch was also tested rather than assumed. A fixture-tuned assignment permits 28/50 sites and 44.357% of MACs at int8 with fixture drift 0.0042, but exceeds the gate on **80/150** unseen corpus positions, worst drift 0.0110. Adding calibration positions cuts coverage without making it safe: at 80 positions, 18 sites / 10.581% MACs remain and 2/150 holdout positions still fail; at 160, 18 sites / 13.953% MACs and 3/150 still fail. Error is diffuse through the eight residual layers, not concentrated in a handful of replaceable sites.[10]

The committed study tests protect the negative conclusion: they require each whole-model scheme to fail the actual tolerance, require every calibration-sweep result to fail generalization, and ensure holdout rows are disjoint from calibration rows. In this investigation, those artefact-level tests passed; Torch-dependent quantizer and forward tests were skipped because Torch is unavailable.[11]

### 5. The trainer has gradient clipping but no weight clipping; that is a retraining input, not a post-export transformation

Both trainers use gradient clipping, but neither constrains model values after the optimizer update. The direct student trainer performs `clip_grad_norm_` then `optimizer.step()`; its synthetic self-check has the same unconstrained update pattern.[12] The A100 oracle and student-distillation loops unscale gradients, clip norms, run autonomous safety checks, then `scaler.step(optimizer)`.[13] This establishes a concrete insertion point for a future experiment but does **not** justify applying a Stockfish-derived global clamp to the shipped weights.

Correct quantization-aware training needs a declared target representation and independently justified bounds for each relevant tensor class, particularly because the largest observed offenders are embeddings and this model’s nonlinearity/ranges differ from Stockfish NNUE. A post-hoc clamp would alter a calibrated frozen model without letting the rest of training compensate, invalidating its policy and parity expectations. A package converter that pads GAB tensors or clips values is likewise not a candidate model; it either preserves old capacity with zeros or injects random/untrained parameters.

### 6. Game-facing policy remains default-off and requires real paired-game evidence

The UCI default is explicit: `UnarchitecturedHint=false`, the selected exit defaults to `2/128`, and `UnarchitecturedMinTime` defaults to 30,000 ms. The candidate only loads its model after an explicit option; its exact-key worker includes the exit in the cache identity.[14] The supplied backlog records the required status: the runtime speed refactor is merged and parity historically passed, but `UnarchitecturedHint` is untouched and stays default-off because no configuration across four real SPRT batches trended positive.[15]

There is a ready fail-closed paired-game launcher, but it requires an engine binary, model file, opening book, output locations, and `cutechess-cli`; it tests candidate `UnarchitecturedHint=true` against baseline `false` with paired games and `elo0=0`, `elo1=5`, `alpha=beta=0.05`.[16] It currently does **not** set `UnarchitecturedHintExit`, so it silently uses the shallow `2/128` default. That is unsuitable for a future claim about the full exit: calibration shows `2/128` is materially worse than `8/256`, and the integration guidance says a future SPRT must state and test its exit.[17]

## Status by decision class

| Work item | Current disposition | Inputs now | Blocking condition | Required evidence before promotion |
|---|---|---|---|---|
| Keep retained i8 weights + dynamic i16 activations | **Implemented and retained** | Shipped package and Rust implementation | None for existing behavior | Preserve regression gates; benchmark only if changing the code. |
| Reproduce frozen GAB ablation/int8 study | **Tooling-ready, offline only** | Package, 600-FEN corpus, and labels are present | Torch absent in this sandbox | Install/use a pinned Torch environment; reproduce reports into a new, candidate-labelled path. |
| Correct stale package-loader comment | **Implemented-ready, documentation only** | Source is present | None | Ensure comment says the parser itself does not forward while `unarchitectured_metal_runtime.rs` does; no behavior change. |
| Parameterize Rust runtime for larger GAB dimensions | **Do not implement speculatively** | Only the old 8/32/32 package exists | No widened trained package/reference vectors | Candidate checkpoint first; then update schema/arrays and prove old/new intended package behavior with Python reference. |
| Widen GAB to 32/64/64 | **Retrain-gated** | Config/trainer architecture supports config-driven GAB dimensions | No Unarchitectured oracle/student checkpoint or training/calibration/validation shards | Controlled retraining, export, offline calibration, numerical parity, safety/deployment checks, real SPRT. |
| Add pure int8 activation kernels | **Rejected for current checkpoint; retrain-gated hypothesis only** | Negative study and current package exist | All five schemes miss parity; mixed split fails holdout | Quantization-aware candidate with target range design; repeat whole-model + held-out test before writing kernel. |
| Add post-step value clipping / fake-quant training | **Retrain-gated experimental code** | Exact optimizer insertion points and configs are present | No training data, checkpoint, or validated tensor bounds | Train baseline and clipping/QAT candidates from scratch with the same data/seed policy, then all evaluation gates. |
| Enable hint or change its default | **Blocked on real SPRT** | Default-off UCI path and runner exist | No candidate evidence can overturn prior negative results; local match infrastructure absent | Explicit-exit paired cutechess SPRT on real hardware, plus tactical/integrated/deployment gates. |

## Conditional implementation plan once the missing assets exist

This section is deliberately conditional. It recommends concrete code only together with the inputs that make that code testable.

### A. Candidate definition and training matrix

When trusted `UNCHD4R0` train, calibration, and final-holdout shards plus the intended oracle checkpoint are supplied, train a small factorial experiment rather than attributing every change to GAB:

| Candidate | GAB | Quantization training | Purpose |
|---|---|---|---|
| A | Current 8/32/32 | Current training | Reproducible baseline |
| B | 32/64/64 | Current training | Isolate GAB capacity |
| C | Current 8/32/32 | Target-scheme value constraints / QAT | Isolate quantization constraint cost |
| D | 32/64/64 | Target-scheme value constraints / QAT | Test the combined hypothesis |

Use the checkpoint selected by validation—not simply the final epoch—then perform regret calibration and canonical `UNARCHV1` export. Preserve manifests/hashes for training, calibration, validation, teacher/oracle, architecture configuration, and exported package. The direct trainer already accepts a configuration-driven `ElasticGeometricBias`; the A100 path has corresponding oracle and student-distillation GAB fields, so the primary code work is configuration plus a *candidate-tested* runtime compatibility update rather than an architectural rewrite.[12][13]

### B. Quantization-aware change, only with target bounds and training assets

Add an explicit configuration block such as `quantization_aware_training` with default **disabled** and named, target-scheme-derived ranges. After every successful optimizer update, clamp only the parameters covered by that declared policy under `torch.no_grad()`—after `optimizer.step()` in the direct student trainer and after `scaler.step(optimizer)` in each A100 training loop, before the next forward pass. Record maximum absolute values and fraction-at-bound in checkpoints/metrics. Do not treat Stockfish’s `±127/64` as a validated universal bound for this transformer; use it only as motivation for deriving this model’s bounds.

If int8 activations remain the delivery goal, add fake-quant/observer simulation in training or an equivalent loss-aware quantization method and validate the **complete exported candidate** over both frozen parity fixtures and a disjoint corpus holdout. Only after the candidate achieves the existing numerical criteria should an i8-activation kernel be contemplated. The unchanged runtime remains the fallback throughout.

### C. Candidate-runtime compatibility and parity contract

For a genuine widened checkpoint, update the Rust schema and GAB workspace/array dimensions to match the exported candidate or, preferably, derive validated dimensions from package sections while enforcing allowed architectural relationships. The package format’s generic shapes are not enough: the runtime must validate `gab.token_projection`, `gab.compress`, templates, coefficients, and all dependent scratch sizes before forwarding. Reject mismatched or unsupported candidate packages fail-closed.

**Do not remove, rename, weaken, or replace the following three gates.** They must continue to use the independent `tools/reference_forward_unarchitectured_metal.py` oracle and retain the full-exit `5e-3` tolerance:

1. `start_position_matches_python_reference`;
2. `midgame_position_matches_python_reference`; and
3. `position_to_input_matches_hand_built_start_position`.

The first two currently freeze logits plus value/representation checks from the independent Python forward; the third verifies the live board/movegen conversion against the hand-built start-position fixture.[5] If a deliberate new checkpoint replaces the reference package, regenerate expected values from Python, retain the test names and strict tolerances, and preserve old-package compatibility testing if the old package remains supported. Also retain `integer_matrix_path_stays_close_to_dequantized_path` (`5e-4`) and narrow-exit reference coverage; neither is a substitute for the three named full-path gates.[8]

### D. Offline and game-facing candidate sequence

With a newly exported candidate package, first re-run the package inspector, GAB ablation, int8 activation analysis (including `--mixed --calibrate 80 --validate 150` or a larger precommitted holdout), policy calibration against the existing cached labels, and all Rust/Python correctness and safety tests. Treat forward-pass timing as host-specific and benchmark the actual deployment CPU.

Only then make a copy of the SPRT launcher for the candidate, parameterize an explicit `HINT_EXIT` (defaulting to no implicit value and validating `2/128|4/192|8/256`), and pass `option.UnarchitecturedHintExit="$HINT_EXIT"` to the candidate. Record package SHA-256, exit, `MinTime`, engine SHA, opening book SHA, CPU, threads, hash, time control, and raw paired PGN. Analyze the paired PGN with the project’s pentanomial tool as well as retaining the sequential SPRT outcome.[9][16] Until a real match passes the project’s gate, leave `UnarchitecturedHint=false` unchanged.

## Verification record

| Check | Result | Meaning and limitation |
|---|---|---|
| Package inspection | **Passed.** 121 tensors; 75 int8 / 46 f32; calibrated `UNARCHV1_STUDENT_CALIBRATED_V1`; package SHA-256 `5fd9…b16d`. | Establishes package integrity/metadata and actual GAB dimensions, not forward quality or Elo. |
| Focused GAB and int8 test suites | **28 passed, 0 failed, 8 skipped.** `python3 -m unittest -v tools/test_gab_ablation.py tools/test_int8_activation_calibration.py`. | Artefact contracts, package-header dimensions, calibration-sweep logic, and tool help passed. Eight tests requiring Torch were skipped. |
| Standalone tool help | **Passed.** Both GAB and int8 analysis scripts accepted `--help` without Torch/model loading. | Confirms command discoverability, not analysis execution. |
| Inputs/assets | **Present:** shipped `.unmetal`, primary/replication calibration corpora, labels, calibration report, GAB and quantization JSON artefacts. **Absent:** any Unarchitectured `.pt/.pth/.ckpt` oracle/student checkpoint and discoverable Unarchitectured training shards. | The finished package is sufficient for offline analyses; it cannot be resumed, widened, or quantization-aware retrained. |
| Rust named parity tests | **Not executed locally.** Their declarations remain at `unarchitectured_metal_runtime.rs:2829`, `:2891`, and `:3582`. | Installed Cargo 1.75 rejects the repository’s v4 `Cargo.lock` (`requires -Znext-lockfile-bump`). The initial multi-filter cargo invocation was invalid; the corrected single-filter attempt reached this lockfile/toolchain blocker. Do not interpret the historical passing result as a fresh run. |
| Training / benchmark / SPRT | **Not run.** | Explicitly excluded by this investigation; no strength or speed claim is made. |
| Worktree | **Clean after investigation.** | This report is outside the repository; no tracked repository file changed. |

## References

[1]: [GAB capacity finding](file:///home/ubuntu/Unchessed-UCI-Engine/docs/gab-capacity-finding.md) — shipped versus paper dimensions, frozen-weight ablation, and retrain-only conclusion.

[2]: [GAB ablation tests](file:///home/ubuntu/Unchessed-UCI-Engine/tools/test_gab_ablation.py) — binary-header shape checks and artefact consistency contracts.

[3]: [Rust `UNARCHV1` loader](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/unarchitectured_metal.rs) — binary header, dtype, alignment, CRC, and section validation.

[4]: [Python `UNARCHV1` package builder/parser](file:///home/ubuntu/Unchessed-UCI-Engine/tools/unarchitectured_metal_package.py) — writer-side dtype, metadata, scale, and checksum conventions.

[5]: [Rust Unarchitectured runtime](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/unarchitectured_metal_runtime.rs) — fixed schema, i16×i8 arithmetic, GAB forward path, and named parity gates.

[6]: [GAB contribution analyser](file:///home/ubuntu/Unchessed-UCI-Engine/tools/analyse_gab_contribution.py) — reproducible frozen-weight ablation contract and candidate inputs.

[7]: [UNARCHV1 exporter](file:///home/ubuntu/Unchessed-UCI-Engine/tools/export_unarchitectured_metal.py) — calibrated checkpoint requirement and per-tensor symmetric int8 export rule.

[8]: [Runtime optimization record](file:///home/ubuntu/Unchessed-UCI-Engine/docs/unarchitectured-metal-runtime-optimization.md) — retained quantization path, numerical gates, rejected int8 activation attempt, and default-off integration decision.

[9]: [Fishtest and quantization notes](file:///home/ubuntu/Unchessed-UCI-Engine/docs/fishtest-and-quantization-notes.md) — post-training range evidence, training-time clipping context, and pentanomial analysis guidance.

[10]: [Int8 activation calibration finding](file:///home/ubuntu/Unchessed-UCI-Engine/docs/int8-activation-calibration-finding.md) — five schemes, holdout failure, diffuse error, and retrain gate.

[11]: [Int8 activation calibration tests](file:///home/ubuntu/Unchessed-UCI-Engine/tools/test_int8_activation_calibration.py) — artefact and quantizer safeguards.

[12]: [Direct compact-student trainer](file:///home/ubuntu/Unchessed-UCI-Engine/tools/train_unarchitectured_metal_student_a100.py) — config-driven student GAB, gradient clipping, optimizer step, calibration, and checkpoint selection.

[13]: [A100 oracle/student trainer](file:///home/ubuntu/Unchessed-UCI-Engine/tools/train_unarchitectured_metal_a100.py) — oracle/distillation loops and mixed-precision optimizer steps.

[14]: [UCI integration controls](file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-core/src/uci.rs) — explicit default-off hint, default shallow exit, and minimum time control.

[15]: [Supplied runtime-speed backlog](file:///home/ubuntu/Unchessed-UCI-Engine/scripts/research/arena_agent_unarchitectured_metal_runtime_speed_prompt.md) — canonical v1 scope, historical gates, benchmark context, and explicit default-off instruction.

[16]: [Paired root-hint SPRT runner](file:///home/ubuntu/Unchessed-UCI-Engine/scripts/sprt-history/sprt_unarchitectured_metal_hint.sh) — required external assets and real-match setup.

[17]: [Unarchitectured Metal calibration record](file:///home/ubuntu/Unchessed-UCI-Engine/docs/unarchitectured-metal-calibration.md) — exit-specific policy quality, corpus provenance, and remaining promotion gates.
