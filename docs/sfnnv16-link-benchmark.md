# SFNNv16-compatible evaluator/link benchmark

Branch: `manus/research-facilities`.

The implementation uses the existing incremental NNUE state seam and exposes a read-only `EvalBarLink` carrying the bitboard snapshot and Elo-detector telemetry. The UCI path is on-demand, so it does not execute on every search node.

| Test | Result |
|---|---:|
| Full core library tests | 144 passed, 0 failed, 6 ignored |
| Eval-bar focused tests | 8 passed, 0 failed |
| Matetrack EPD positions | 7 |
| WDL benchmark rounds | 10,000 |
| Exact logistic reference | 4,974,389 ns |
| Tensorized fast path | 1,643,798 ns |
| WDL projection speedup | 3.03x |
| Maximum WDL delta | 0 per mille |
| Exact checksum | 62,020,000 |
| Fast checksum | 62,020,000 |

The final reviewer UCI smoke test produced:

```text
info string [Unchessed] evalbar cp 14 wdl 32 957 11 bar 0.510 smoothed_cp 14 source=hce-proxy provenance=proxy
info string [Unchessed] evalbar-link hash 1780828403df6413 occ ffff00000000ffff material 78 elo 1500 +/-450 suspect=false reason=none
```

The `EvalBarLink` API is documented in `docs/evalbar-elo-bitboard-link.md`. The link is read-only and carries `position_hash`, total occupancy, White occupancy, Black occupancy, material index, display score, bar fraction, Elo estimate, confidence, suspect flag, and detector reason. The `sample_with_state` seam accepts an existing `EvalState`, allowing an accumulator owner to avoid board rescans.

The benchmark measures the WDL projection, not whole-engine playing strength or whole-search nodes per second. Exact SFNNv16 quality still requires the trained network weights and matching training pipeline; the current loaded NNUE route is explicitly labeled `nnue-proxy` unless those exact artifacts are present.
