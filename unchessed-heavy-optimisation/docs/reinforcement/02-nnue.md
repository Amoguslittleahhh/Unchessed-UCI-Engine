# NNUE relabeling and retraining support investigation

**Repository/branch inspected:** `/home/ubuntu/Unchessed-UCI-Engine`, `manus/rustc-bootstrap-trial`  
**Scope:** `tools/nnue_relabel_existing.py`, NNUE data/training scripts and formats, tests, pipeline wrappers, and the research backlog.  
**Investigation boundary:** No tracked repository file was changed. No training, search-label generation, or SPRT was run.

## Executive assessment

The repository has a **useful proof-of-mechanics**, not yet a reproducible stronger-label experiment. `tools/nnue_relabel_existing.py` can replace the signed 16-bit score at bytes 96–97 of a single valid-looking 104-byte sample shard and report basic old/new statistics. Its four focused tests pass locally. The original 108M recipe result establishes that the current 5,000-node labels are the likely limiting variable: the recipe-best v4 net reached 47.8 cp validation MAE yet scored **−155.6 ± 47.7 Elo** against the shipped net [1]. That is sound motivation to test stronger labels; it is not evidence that a stronger-label candidate is good.

Before operating on the real corpus, the relabeling path needs three **blocking implementation changes**: (1) a streaming, collision-safe, atomic transformer; (2) a content-bound sidecar/manifest format that proves labels correspond to the exact ordered records; and (3) a way to retain or reproduce full search state (at least castling rights and en-passant) for every existing sample. The current 104-byte record deliberately retains only STM-normalized piece bitboards, score, WDL, and padding [2]; a high-node search cannot safely be reproduced from it alone because legal moves depend on state the record discarded. The present data generator has no “relabel existing records” command, has a fixed 5,000-node HCE labeler, and does not load an NNUE evaluator for a high-node teacher search [3].

These support changes, their fixtures, and all byte-level verification can be implemented and exercised on CPU without GPU or the unavailable 108M corpus. **Producing real labels, retraining the full corpus, and making any strength/default claim remain blocked or gated:** the corpus and sidecars are absent locally; PyTorch/CUDA are absent; and a real cutechess SPRT is mandatory before promotion.

## Evidence ledger

| Finding | Evidence | Consequence |
|---|---|---|
| The sample ABI is stable and precisely defined. | Datagen writes 12 little-endian `u64` STM-normalized bitboards (bytes 0–95), little-endian `i16` STM score (96–97), STM WDL byte (98), and five zero pad bytes (99–103) [2]. `train_nnue.py` uses the matching NumPy dtype [4]. | A relabeler must preserve every byte other than offsets 96–97 and must preserve the STM score convention. |
| Existing relabeling is only a positional score overwrite. | The tool reads the full source into `blobs` and old-score lists, reads the entire `.i16` file into a list, checks only record count plus source `wdl <= 2`/zero padding, then opens `out` directly with `wb` and writes modified records [5]. | It has no cryptographic evidence that a sidecar belongs to the source boards or their order; it is unsafe for large-corpus operation and unsafe on colliding paths. |
| The current transformer is not safe against output aliasing. | A scratch-only two-record probe passed the score sidecar as `apply` output. It returned 0 and changed the 4-byte sidecar into a 208-byte shard. Code review confirms direct `open(out_path, "wb")` with no `samefile`, overwrite, temporary-file, `fsync`, or rename guard [5]. | A typo can destroy the labels; using the source shard itself as output can also overwrite the baseline corpus. This is a release-blocking data-safety defect. |
| The implementation is non-streaming in multiple places. | `read_old_scores()` retains every 104-byte blob and Python integer; `load_scores()` reads/unpacks all labels; `report()` allocates diffs and two centered float vectors for Pearson [5]. At 108M records, source data alone are 11,232,000,000 bytes (10.46 GiB) and the score sidecar is 216,000,000 bytes (206.0 MiB), before Python-object/list/temporary-array overhead. | Full-corpus compare/apply lacks a bounded-memory design. It conflicts with the documented constraint that this sandbox cannot even hold the 108M training run in RAM [6]. |
| Current validation is not sufficient to establish valid NNUE positions or safe outputs. | The relabeler validates file divisibility, nonempty source, WDL range, and padding only [5]. `check_nnue_data.py` separately checks king counts and plane overlap, but is not invoked by relabeling and does not validate output equivalence [7]. | Wrong, duplicated, shuffled, or semantically invalid boards can pass relabeling; a successfully written output has no post-write proof that only score bytes changed. |
| A score-only sidecar cannot prove record identity. | The documented contract is “one i16 per record” [8]. Neither the raw `.i16` nor the CLI includes an input shard SHA-256, hash of bytes 0–95/98–103, ordered record IDs, generator version, labeler configuration, or sidecar checksum. | Equal length is not a pairing guarantee. A sidecar from a different ordering or similarly sized shard silently relabels the wrong boards. |
| Existing records cannot safely drive a deeper search by themselves. | The writer saves only bitboards, STM score, and WDL [2]. The original label search runs on a complete `Position`; its score depends on legal move generation. Castling/EP are omitted from the record, while the policy-data format explicitly carries both, illustrating the distinction [2]. | Exact higher-node HCE or NNUE-search relabeling requires an external state index/FEN/canonical position companion or deterministic regeneration from retained PGNs and an exact selection manifest. It must not silently assume “no castling/no EP.” |
| No existing stronger-teacher/relabel producer is implemented. | Datagen exposes `nnue` and `nnue-stream` generation modes; its NNUE label node count is the compile-time `NNUE_LABEL_NODES = 5000`, it instantiates `Hce::default()`, and it writes score records directly [3]. The relabel tool expressly “does not run search or inference” [5]. | There is no supported, parameterized, CPU-side command that accepts existing-state inputs, records teacher identity/configuration, and emits a bound label package. |
| Existing quality filters are important provenance, not a replacement for stronger-label validation. | The source accepts positions only after static-vs-qsearch (M1), HCE search, tactical/mate/absolute-score rejection, and static-vs-search (M2) filters [3]. The documented current defaults are 60/70 cp [9]. | A higher-depth teacher can expose new tactical/mate/out-of-range cases. The experiment must define whether to retain, reject, or separately mark them; accepting arbitrary `i16` values is not a reproducible policy. |
| The normal trainer is more mature than relabeling but lacks experimental provenance. | `train_nnue.py` validates WDL/pad, uses a seed-42 random 2% validation split, early-stops, and exports the best validation-MAE state [4]. It takes positional shard paths and emits the network; JSONL metrics are optional and do not include a source/label manifest [4]. | Same ordered inputs reproduce the split, but a run cannot later prove its shard order, hashes, labeler, sidecar lineage, environment, or exact train/validation membership from the output alone. |
| The A100 XT prototype has stronger checkpoint manifests but is not a production replacement. | `FixedRecordShards.manifest()` records path name, bytes, record count, and SHA-256; XT checkpoints save train/validation manifests [10][11]. Its output is deliberately a research checkpoint not loadable by production Rust [11]. | Reuse the manifest pattern for ordinary v4 relabel/retraining; do not substitute XT checkpoints or training for the v4 experiment. |
| Real assets required for the experiment are unavailable here. | Documentation says the 12 original 108M shards are on the reviewer’s disk, not in git; 108M × 104 B is about 11 GB, and this sandbox lacks enough RAM/GPU [6]. This inspection found no corpus shard and no `.i16` sidecar under the repository or `/home/ubuntu`; only deployed/archival 23 MB `UNCHNNUE` weight files were present. Python import reported no `torch`; CUDA cannot therefore be used. | This environment can verify code and tiny fixtures, but cannot produce the real candidate or retrain it. The shipped `unchessed-nnue.bin` is available as a teacher/evaluation weight file, not as a full original-position corpus or a stronger-label checkpoint. |
| Past 108M results do not authorize cloud retraining or promotion. | The reviewed 108M run early-stopped at epoch 6, best epoch 3 at 47.8 cp, but the SPRT result was −155.6 ± 47.7 Elo. The precommitted rule leaves 178M cloud spend **NO-GO** even at the optimistic interval end [1]. | Stronger labels are a new experiment, not approval to spend on more of the same labels. No candidate may change the default without its own real SPRT. |
| The full playing-strength gate is explicitly distinct from smoke testing. | `smoke_test_nnue.sh` calls itself a reject-only 100-game filter and states that only a full SPRT establishes an Elo gain [12]. The analysis recommends an SPRT against `unchessed-nnue.bin` at `tc=10+0.1`, `elo0=0`, `elo1=10`, with explicit Adaptive-on and Adaptive-off gates [13]. | Unit tests, MAE, Pearson, deterministic label checks, or a smoke-test pass cannot support a strength claim or default flip. |

## Status by decision class

| Decision class | What is actually available now | What it does **not** establish |
|---|---|---|
| **Implemented and ready for limited fixture use** | `nnue_relabel_existing.py compare/apply`; the thin shell wrapper; fixed record ABI; conventional v4 trainer with best-checkpoint/early-stop behavior; focused relabel tests. The focused test command passed **4/4** locally: `python3 -m pytest tools/test_nnue_relabel_existing.py -q`. `--help` also passed standalone. | It is not a safe whole-corpus campaign tool, a label generator, a corpus manifest, a training provenance system, or a strength result. |
| **Ready to implement and verify without GPU assets** | Streaming relabel/verify code; JSON manifest schema; source/sidecar binding; atomic output; strict fixture validation; a stateful CPU label-package interface; ordinary-trainer run manifests; unit/integration tests with tiny fixture data. These need only repository source, standard library/available NumPy test dependencies, and CPU CI. | Implementing them does not create a credible high-node teacher label, full-corpus candidate, or Elo evidence. |
| **Blocked on missing data/checkpoints/state** | The real 108M/178M source shards, correctly ordered stronger-label packages, and per-record full-position state or a deterministic source/selection reconstruction are not available locally. No parameterized high-node teacher producer exists. PyTorch/CUDA are absent. | A static forward-only score from the shipped net could in principle be defined from bitboards, but it is not the documented high-node search teacher and cannot be represented as such. It also cannot be run over the unavailable corpus here. |
| **Requires a real SPRT after assets exist** | Candidate full-corpus training, loader/inference acceptance, reject-only smoke screening, and a full paired cutechess SPRT on appropriate hardware. | No local unit test, label-disagreement statistic, validation MAE decrease, or smoke test is a substitute. The shipped v3 net remains default until a new candidate passes. |

## Recommended changes, ordered by dependency and risk

### P0 — make relabeling bounded, fail-closed, and non-destructive

1. **Replace list-based processing with one-pass/bounded-memory streams.** Read a fixed record chunk and a same-count score chunk; accumulate count, means, covariance, MAE, RMS, changed count, sign flips, min/max, and distribution summaries online (Welford/Pébay or two passes). `compare` must not retain board blobs at all. `apply` must modify only a chunk-local `bytearray` at offsets 96–97. This makes the implementation proportional to an explicit `--chunk-records` bound rather than to corpus size.

2. **Reject all input/output aliases before any write.** Resolve paths and use `os.path.samefile` where they exist; reject output equal to the source shard, score package, manifest, or any other declared input, including hard links/symlinks. Default to refusing an existing output; permit an explicit `--force` only for a non-input target. The observed score-sidecar destruction makes this non-optional.

3. **Use an atomic destination protocol.** Check destination-parent free space for one complete output plus the configured safety margin; write a uniquely named temporary file in the destination directory with restrictive permissions, flush and `fsync`, fully verify it, then `os.replace` only after all checks pass. Preserve the source unconditionally. On error, remove only the temporary file and leave source, sidecar, and prior destination unchanged.

4. **Add `verify` rather than treating a successful write as proof.** In a fresh streaming pass, require: expected output byte length; output record structure; exact equality of input/output for byte ranges 0–95 and 98–103; each output score equals the corresponding package score; expected records/bytes; and recorded SHA-256 values. Return nonzero on the first mismatch and emit machine-readable JSON as well as readable summary output.

5. **Validate source records to a documented level.** Retain current WDL/pad checks, and add exactly-one-king-per-side and no-plane-overlap validation in a reusable streaming validator (or invoke a corrected shared implementation). Make an explicit `--structural-check=required|off` policy; default `required` for a new experiment. Do not claim full chess legality unless it is actually checked.

### P0 — make labels provably tied to the original corpus and teacher

6. **Replace bare `.i16` as the experimental exchange format with a versioned label package.** The package can still store compact little-endian `i16` scores, but it needs a JSON manifest that records: schema/version; endianness; record size; score POV/unit; source shard names in exact order; source byte sizes, record counts, and SHA-256; SHA-256 of the ordered immutable record payload (all fields other than score, or an explicitly defined board/state digest); score-file SHA-256; label count; output hashes; and timestamp/operator information. `apply` must refuse a manifest whose hashes, count, ABI, POV, or source order do not match.

7. **Bind labels to every ordered position, not merely the total length.** Have the label producer emit either (a) a cryptographic per-record ID derived from canonical board **and full state** or (b) an ordered companion digest stream with a final Merkle/root digest. The relabeler must compare it before accepting each score. A source-shard whole-file SHA-256 is the minimum practical guard; per-record binding also detects an accidentally reordered/repartitioned source during a multi-shard campaign.

8. **Make teacher and score policy first-class metadata.** Record engine/repository commit, binary SHA-256, evaluator/`EvalFile` SHA-256, teacher type (deeper HCE versus NNUE-search versus static forward), nodes/depth/time, threads/hash/search parameters, deterministic seed policy, score clamp/mate encoding, and the M1/M2/filter policy. Refuse scores outside the predeclared policy rather than silently accepting all `i16` values. Report counts of rejected/marked mate, out-of-range, sign-flip, and WDL-inconsistent labels.

### P0 — add the missing stateful label-production path

9. **Preserve a canonical full-position companion for future corpora.** Extend the NNUE sample-generation workflow to write a lockstep, versioned record containing enough state to replay a search: canonical FEN or compact normalized board state including STM, four castling-right bits, en-passant information, and a stable record ID. Preserve the original PGN/game/ply mapping where feasible. Hash and manifest it beside each shard.

10. **For the existing corpus, require an auditable reconstruction before high-node search.** The supported options should be: (a) obtain an authoritative companion state/index from the corpus owner, or (b) regenerate the exact selection/order from immutable PGNs using the recorded generator commit, filters, seed, worker topology, and logs, then prove the reconstructed 104-byte source hash equals the supplied shard. If neither succeeds, block search-based relabeling. Never infer absent castling/EP state.

11. **Implement a CPU-capable `nnue-relabel-label` producer after state is available.** It should consume the canonical state stream, produce the bound label package in the same order, support a deliberately explicit HCE/deeper-search or NNUE-teacher configuration, write resumable per-shard progress, and run a deterministic small-fixture mode. It must be a producer only: no training, evaluation promotion, or unguarded cloud launch. A static NNUE forward-only producer, if desired, must label itself precisely as static—not “high-node self-distillation.”

### P1 — make retraining comparable and auditable

12. **Add an ordinary-v4 training run manifest.** Port the useful `FixedRecordShards.manifest()` discipline from the XT prototype: save input label-package IDs/hashes, exact ordered shard list, record counts, split seed and validation IDs/hash, trainer git commit, Python/NumPy/PyTorch versions, CPU/GPU/device, all environment overrides, epochs actually run, best epoch, val metrics, and exported `.bin` SHA-256. Write this beside both metrics and candidate weights; make `train_recipe.sh` pass a required run-manifest path for relabel experiments.

13. **Make the comparison split immutable and explicit.** The current seed-42 record-level split is deterministic only when the exact concatenation order is retained. Derive and store a content-addressed split manifest before relabeling and use the same membership for old-label and new-label training. If game IDs become available in the new companion state, prefer a game-level holdout to reduce correlated-position leakage.

14. **Add stronger label-disagreement diagnostics.** Keep MAE/RMS/Pearson, but add signed bias, score quantiles, absolute-delta quantiles, sign-flip rate, stratification by WDL and material/output bucket, counts beyond thresholds, and a reproducible sampled record-ID diff report. Pearson/MAE alone neither validates alignment nor determines whether a teacher is stronger.

15. **Protect cross-platform binary inputs.** `.gitignore` already ignores `*.bin`, and `.gitattributes` protects PGN/EPD/JSONL but not score/manifest binary sidecars [14]. If compact label packages/manifests are ever versioned or transferred through version-control tooling, mark their binary payload (for example `*.i16`, `*.nnlbl`) `-text` and keep manifests byte/hash stable. Prefer storing large real assets outside Git while versioning only schema, fixture, and run metadata.

### P1 — test the implementation, not the expensive hypothesis

16. **Expand `test_nnue_relabel_existing.py` into failure-oriented tests.** Current tests cover help, a happy overwrite, length mismatch, and standard-library imports only [15]. Add empty/odd sidecar, malformed source size, invalid WDL/padding, malformed kings/overlap, little-endian boundary values, constant-score Pearson behavior, exact output-field preservation, and known JSON statistics.

17. **Add destructive-path regression tests.** Assert `apply` refuses source-output, sidecar-output, manifest-output, hard-link-output, symlink-output, and pre-existing destination without `--force`; assert input hashes and destination contents remain unchanged after every expected failure. Add a simulated mid-write/verification failure to prove the old destination is retained and the temp file is cleaned.

18. **Add bounded-memory and multi-shard tests.** Use a large synthetic stream or mocked reader with a deliberately small chunk size; assert the tool makes no whole-file read and that output/statistics are independent of chunk size. Test a manifest whose shard order is permuted, source hash is altered, or a correct-count wrong sidecar is supplied; each must fail before output is published.

19. **Add a CPU-only end-to-end fixture.** Generate several legal canonical states, create a deterministic mock/depth-1 label package, bind it, apply it, verify byte equality except score, train only a tiny CPU test corpus when Torch is available, and validate the exported `UNCHNNUE` header/file size plus Rust loader acceptance. This verifies mechanics without pretending the mock teacher or tiny training set establishes strength.

## Verification and execution plan

### A. Implementation gates that do not require corpus, GPU, or SPRT

1. Run the expanded stdlib relabel suite, including alias, atomic-failure, malformed-input, provenance-mismatch, and chunk-size-invariance tests. A passing run must leave fixture input hashes unchanged.
2. Run `compare`, `apply`, and `verify` on a committed tiny fixture package. Check the manifest’s source, state-digest, label, output, and report hashes independently with `sha256sum`.
3. Run the CPU-only label-producer fixture twice under its declared deterministic settings. Require identical label package and manifest hashes; otherwise record and bound nondeterminism before using it as a teacher.
4. Run the existing NNUE trainer self-check and best-checkpoint tests only in an environment with Torch, using CPU/synthetic data. Validate the exported v4 file with the Rust `Nnue::load` path, not only the Python writer’s own parser.
5. Run the normal repository Python suite appropriate to the environment. The current focused relabel baseline is **4 passed in 0.14 s**; preserve and extend it. This evidence verifies support code only.

### B. Asset preflight — fail closed before any real label or training job

Require a single preflight command to reject execution unless all of the following are present and hashes match: source-shard manifest, source files, state companion/reconstruction proof, teacher binary/checkpoint, exact labeler configuration, sufficient scratch/destination space, and a writable non-colliding output location. It should report `BLOCKED` rather than start a search or a training run if any condition is absent.

For an existing-corpus deeper-search experiment, verify reconstruction by reproducing the authoritative 104-byte shard hash **before** making one new label. This is the decisive check that selection/order and lost state have been recovered. A bare bitboard shard without this proof may be used only for a clearly labelled static-forward experiment, not a legal-move search experiment.

### C. Real-experiment sequence once assets exist

1. Freeze the old corpus and source/state manifests; choose the old/new comparison split before relabeling.
2. Generate stronger labels with the teacher metadata fixed in advance. Run a small independently repeated subset first; inspect paired deltas, out-of-policy values, and source/record bindings. Do not interpret a large MAE as a win.
3. Apply and independently verify every relabeled shard; publish the aggregate provenance and label-disagreement report. Preserve originals and outputs separately.
4. Retrain with the already documented v4 recipe (15-epoch cap, early-stop/best checkpoint, exact recorded shards and split) and publish the ordinary-v4 run manifest. Full-corpus retraining is **blocked here** by absent corpus and Torch/CUDA, not by a missing code path alone.
5. Validate the output’s ABI and run the reject-only smoke screen if the real engine, book, and cutechess assets are installed. A pass means only “not obviously terrible.”
6. Run the real paired cutechess SPRT against `unchessed-nnue.bin` using the declared harness and conditions. Follow the documented two-gate recommendation: make Adaptive explicitly on for the persona-on gate and explicitly off for the eval-only gate [13]. Only a completed, recorded real SPRT can justify any strength claim, cloud decision, or default change.

## Explicit non-conclusions

- No stronger sidecar, full-corpus retrain, candidate v4 weight, or stronger-label validation-MAE result was produced in this investigation.
- The available 23 MB shipped `unchessed-nnue.bin` proves an inference weight is present; it does **not** supply the missing 108M source records, per-record search state, teacher-run configuration, or a training checkpoint.
- The existing v4-overtrained file and the XT prototype checkpoint/pointer are not evidence for the proposed experiment and should not be promoted or used as an undocumented teacher substitution.
- The historical 108M 47.8 cp / −155.6 Elo result remains evidence against more of the same labels and against the 178M cloud spend under the prior decision rule [1]. It does not predict a stronger-label result.
- No local verification can replace the final real SPRT.

## References

[1]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/nnue-v4-108m-recipe-result.md "NNUE v4: 108M-record recipe-validated SPRT"
[2]: file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-datagen/src/main.rs "NNUE 104-byte record layout and writer"
[3]: file:///home/ubuntu/Unchessed-UCI-Engine/unchessed-datagen/src/main.rs "NNUE label generation: fixed 5,000-node HCE"
[4]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/train_nnue.py "v4 trainer record dtype, split, training/export controls"
[5]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/nnue_relabel_existing.py "Existing sidecar compare/apply implementation"
[6]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/nnue-v4-training-recipe.md "108M corpus availability and host constraint"
[7]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/check_nnue_data.py "Separate NNUE data sanity checker"
[8]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/nnue-stronger-labels-existing-corpus.md "Existing-corpus sidecar experiment note"
[9]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/nnue-dataset-quiet-filters.md "Quiet filter rationale and defaults"
[10]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/a100_common.py "FixedRecordShards content manifest"
[11]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/train_nnue_xt_a100.py "XT checkpoint provenance and non-production status"
[12]: file:///home/ubuntu/Unchessed-UCI-Engine/scripts/nnue-pipeline/smoke_test_nnue.sh "Reject-only smoke-test limitation"
[13]: file:///home/ubuntu/Unchessed-UCI-Engine/docs/ieee-low-cp-val-mae-and-persona.md "Recommended relabel and dual-SPRT gates"
[14]: file:///home/ubuntu/Unchessed-UCI-Engine/.gitattributes "Cross-platform byte-integrity rules"
[15]: file:///home/ubuntu/Unchessed-UCI-Engine/tools/test_nnue_relabel_existing.py "Current relabel test coverage"
