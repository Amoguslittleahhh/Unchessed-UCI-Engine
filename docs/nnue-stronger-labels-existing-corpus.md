# Stronger labels on the existing NNUE corpus

This commit does **not** train a net, does **not** spend cloud, and does
**not** flip any UCI default.

The 104-byte shard format is unchanged. Only the i16 STM search-score
field is a candidate for replacement. `tools/nnue_relabel_existing.py`
applies a sidecar of new scores (one i16 per record) and prints MAE /
Pearson vs the original 5000-node HCE labels.

How to produce the sidecar is out of this tool (needs a searcher):

  * deeper HCE on the same boards, or
  * self-distillation: shipped net at high node count, same positions

If new-vs-old MAE is small, the 5000-node labels were already near the
searcher's own noise floor and architecture is not the bottleneck. If
MAE is large and a retrain on the new shards drops val-MAE below ~48cp,
the cap was label noise.

PersonaSmooth / EngineDetectV2 stay default-off (product call).
UnarchitecturedHint stays default-off.
