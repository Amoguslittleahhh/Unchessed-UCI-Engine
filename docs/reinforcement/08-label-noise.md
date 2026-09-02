# 08 — NNUE label-noise cheap diagnostic

**Investigation ID:** `08-label-noise`  
**Tier:** 1 (exactly one item)  
**Repository:** `/home/ubuntu/Unchessed-UCI-Engine`  
**Branch:** `manus/rustc-bootstrap-trial`  
**Decision:** **Defer** the real label-noise conclusion until an authoritative alternative sidecar and its matching shard are supplied. Do not synthesize labels.

## Executive summary

The requested cheap diagnostic is a real paired comparison between the existing NNUE shard labels and a real alternative score sidecar, reporting mean absolute error (MAE) and Pearson correlation. I inspected the repository and `/home/ubuntu` for the required inputs and found **no real NNUE training shards and no alternative `.i16`/score sidecar** that can be paired with them. Consequently, no real `compare` run was possible and **there are no actual MAE or Pearson numbers to report**. This is an input-availability blocker, not a zero-result experiment.

The checkout does contain `tools/nnue_relabel_existing.py`, its tests, the deployed `unchessed-nnue.bin`, `nnue-shards-safe/unchessed-nnue-v4-overtrained.bin`, and documentation describing the intended sidecar experiment. These are not substitutes for an ordered source shard plus an independently produced alternative label sidecar. The shipped network is a model binary, not a sidecar of one score per original record; the archival NNUE binary is likewise not training data. I therefore stopped after the cheap preflight and did not generate, infer, or fabricate labels.

## Question and decision boundary

The narrow question was: **does a real alternative labelling pass differ materially from the existing 5,000-node HCE labels?** The intended diagnostic is a score-wise comparison over the same ordered records:

```text
MAE = mean(abs(new_score - old_score))
Pearson = corr(new_score, old_score)
```

MAE is the standard non-negative regression error whose best value is zero, as documented by scikit-learn’s authoritative API reference [1]. Pearson is useful here only as a paired-association diagnostic; neither metric establishes that a retrained net will improve validation error or playing strength. The repository’s existing note explicitly says that `nnue_relabel_existing.py` replaces only the i16 score field and prints MAE/Pearson against the original labels [2].

A real comparison would have required all of the following: an actual source shard in the repository or supplied storage, an alternative sidecar with exactly one score per record, verified ordering/identity binding, and a declared teacher/configuration. Equal file length alone would not prove pairing. The documents also note that the existing 104-byte record discards full-position state such as castling and en-passant, so a deeper-search relabeler cannot safely reconstruct legal search from the old record alone without an authoritative state companion or auditable regeneration [3].

## What I read and inspected

I read the required synthesis and oracle documents, `docs/reinforcement/00-synthesis.md` through `docs/reinforcement/05-oracle.md` (the repository names the intermediate files `01-search.md`, `02-nnue.md`, `03-persona.md`, and `04-unarch.md`), plus `docs/nnue-stronger-labels-existing-corpus.md` and the relevant NNUE tooling references. The synthesis identifies existing-corpus high-node labelling as blocked by absent 108M shards, sidecars, and complete per-record state; it separately identifies full training as blocked by missing Torch/CUDA [3]. The NNUE investigation records the intended cheap compare, the 104-byte ABI, and the provenance/safety limitations of an unbound `.i16` file [4].

I inspected the branch state and relevant files without changing source code. At inspection time the branch was `manus/rustc-bootstrap-trial`; pre-existing untracked reports were present, but `08-label-noise.md` did not exist before this task. No commit or push was made.

## Asset inventory

| Required/related asset | Inventory result | Interpretation |
|---|---|---|
| Existing NNUE training shards | **Not found** under the repository or `/home/ubuntu` in the shard/data inventory | There is no source record stream against which to align scores. |
| Alternative score sidecar (`.i16`, score/label sidecar) | **Not found** under the repository or `/home/ubuntu` (excluding irrelevant package/cache files) | No real alternative labels are available for comparison. |
| `tools/nnue_relabel_existing.py` | Present | Comparison/apply tooling exists, but it requires the missing real inputs. |
| Relabel focused tests | Present | These exercise synthetic fixtures and tool mechanics, not the real corpus. |
| `unchessed-nnue.bin` | Present, model binary | Not a per-record sidecar and must not be treated as one. |
| `nnue-shards-safe/unchessed-nnue-v4-overtrained.bin` | Present, model binary | Also not a per-record sidecar or source shard. |
| Full-position/state companion | Not found | A deeper legal-search relabel cannot be justified from the old 104-byte records alone. |

The repository also contains unrelated JSONL corpora and calibration artifacts, but none is the authoritative NNUE 104-byte shard plus matching alternative score stream required by this item. No file was reinterpreted as a label sidecar merely because its name contained `label`, `score`, or `shard`.

## What I ran

I ran the following cheap, non-destructive checks:

1. Repository status/branch and reinforcement-file inventory.
2. File inventories for NNUE, shard, sidecar, label, score, and `.i16` candidates under the checkout and `/home/ubuntu`.
3. `python3 tools/nnue_relabel_existing.py --help` to confirm the compare/apply interface is available.
4. `python3 -m pytest tools/test_nnue_relabel_existing.py -q` to exercise the existing synthetic/tooling tests.
5. Source/document inspection of the relabeler, tests, NNUE data format, stronger-label note, and synthesis gates.

The focused tests are **mechanics-only evidence**. Their fixtures demonstrate validation, chunk behavior, and score replacement, but they do not produce a real label-noise estimate. The compare command was **not run on a real corpus**, because no valid shard/sidecar pair existed. Therefore:

| Quantity | Result |
|---|---:|
| Real records compared | **0** |
| Real alternative sidecar records compared | **0** |
| Real MAE | **Not available — blocked before comparison** |
| Real Pearson correlation | **Not available — blocked before comparison** |
| Labels synthesized | **0** |
| Training/retraining | **Not attempted** |
| Search-label generation | **Not attempted** |
| Cloud/expensive dependency installation | **Not attempted** |
| Match/SPRT | **Not attempted** |

## Design-only versus verified evidence

### Verified in this task

The relabel tool and its interface are present. The repository documentation defines a 104-byte record with a signed little-endian i16 score at bytes 96–97 and identifies the intended compare statistics. The asset search verified the absence of a usable real shard/alternative sidecar in the inspected locations. The deployed NNUE files are model binaries, not label streams. No real comparison was silently substituted with a model forward pass or a synthetic fixture.

The repository’s prior evidence also records that a focused relabel test suite passed in the earlier investigation and that the tool has synthetic tests for malformed records, aliasing, chunk-size invariance, and verification [4]. Those are support-tool facts, not a real label-noise result. The current task did not treat them as MAE/Pearson evidence.

### Design-only or assumed, not measured here

The following remain hypotheses/design guidance, not results from this task:

* A **small** new-vs-old MAE could indicate that the 5,000-node labels are already near the tested teacher’s noise floor; a large MAE would motivate retraining on the alternative labels [2].
* A deeper HCE or high-node self-distillation sidecar could be an alternative teacher, but producing it requires a real searcher, exact source positions/state, and provenance metadata. The current old records alone are insufficient for legal-state reconstruction [4].
* A lower validation MAE after retraining would still not establish playing strength. The synthesis requires a reject-only screen followed by a real paired-game SPRT for any candidate/default conclusion [3].
* Pearson can summarize association, but it cannot prove label correctness, causal improvement, or deployment benefit.

## Literature context

The literature supports treating noisy labels as a statistical measurement problem rather than assuming that a different training recipe fixes it. Natarajan et al. formulate learning with random, including class-conditional, label noise and derive loss/risk procedures under explicit noise assumptions [5]. That result concerns classification and does **not** validate any particular chess relabelling strategy; it is relevant only as a caution that noise assumptions and provenance must be explicit. For this diagnostic, the direct evidence must therefore come from a real paired score stream, not from a simulated noise distribution.

The metric terminology is also deliberately conservative: scikit-learn defines MAE as a non-negative regression loss with optimum 0.0 [1]. No literature claim is being made that a lower MAE or higher Pearson necessarily predicts Elo.

## Exact blocker and required unblocking evidence

**Blocker:** there is no authoritative source-shard/alternative-sidecar pair available in the inspected repository or `/home/ubuntu`. Without both files, there is no record-wise comparison to run. The shipped NNUE binaries cannot supply the missing ordered labels, and synthesizing or forward-scoring labels would answer a different question and violate the task boundary.

To resume, supply an immutable, ordered source shard (or complete shard set) and a real alternative sidecar with one score per record, together with a manifest containing at least source names/order, sizes/counts, SHA-256 hashes, sidecar hash/count, score POV/units, teacher binary/checkpoint identity, search nodes/depth/time, and state/reconstruction provenance. The sidecar must be generated from the same positions—not merely from an equal-length file. Then run `compare` in a fresh output/report location, preserve the raw JSON, and independently verify the source/sidecar hashes and counts before interpreting MAE/Pearson.

## Recommendation

**Defer.** Do not pursue a label-noise conclusion or retraining spend from this checkout because the cheap real diagnostic cannot start without real paired assets. Do not drop the research question: the prior 108M result and repository notes provide a legitimate motivation to revisit label quality, but they do not justify inventing labels or using an incompatible model binary as a sidecar. Once the exact shard, sidecar, and provenance manifest are available, rerun this same cheap comparison first; only a materially informative real result should gate any subsequent retraining design. No Tier 2/3 work is recommended by this report.

## References

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html "scikit-learn mean_absolute_error API"

[2]: `docs/nnue-stronger-labels-existing-corpus.md` (repository note: stronger labels on existing NNUE corpus)

[3]: `docs/reinforcement/00-synthesis.md` and `docs/reinforcement/02-nnue.md` (repository synthesis and NNUE investigation)

[4]: `docs/reinforcement/02-nnue.md` (NNUE relabeling/provenance investigation and evidence ledger)

[5]: https://papers.nips.cc/paper/5073-learning-with-noisy-labels "Natarajan et al., Learning with Noisy Labels, NeurIPS 2013"

## Reproducibility note

This report intentionally records **no real MAE/Pearson numbers**, because the measured sample count is zero. Any future report that contains those numbers must name the exact source/sidecar hashes and record count. Synthetic fixture statistics, model outputs, prior documents’ simulations, and prior validation/playing-strength results must remain clearly separated from that real comparison.

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/08-label-noise.md`
