# Regression Audit: Main versus Heavy Optimization

**Date:** 5 September 2026

## Supplied real SPRT result

A real `cutechess-cli` run compared `main` with `unchessed-heavy-optimisation` under the predeclared fast screening gate. Both engines used the same SHA-256-verified NNUE file, one thread, 256 MiB Hash, `Adaptive=true`, `OwnBook=false`, paired openings with colors reversed, `elo0=0`, `elo1=5`, and `alpha=beta=0.05`.

The supplied result was:

- SPRT: H0 accepted.
- Estimated difference: **−244.4 ± 48.0 Elo** for heavy optimization relative to main.
- LOS: **0.0%**.
- Games: **223**.
- Illegal moves: **0**.
- Disconnects/crashes: **0**.
- Time forfeits: **0**.

Twelve games per side were aborted when the SPRT boundary was reached. They were not engine failures.

This is decisive evidence that the tested heavy-optimization configuration was weaker in that experiment. It is not evidence that the optimization concept is impossible, but it does invalidate any claim that the current copy is stronger than `main`.

## PersonaSmooth default divergence

The supplied audit identified a real default mismatch:

| Branch | `Options::default()` | UCI advertisement |
|---|---:|---|
| `main` | `false` | `default false` |
| Heavy optimization before correction | `true` | `default true` |
| Heavy optimization after correction | `false` | `default false` |

The repository’s persona design documents explicitly state that `PersonaSmooth` and `EngineDetectV2` are default-off until a relevant strength test supports a product decision. The heavy branch’s `true` default was therefore inconsistent with the documented contract and was treated as an accidental side effect of the persona-stability work.

The default has been restored to `false`. The existing regression test now asserts the documented default, and the rebuilt UCI adapter advertises `PersonaSmooth type check default false`.

The reported SPRT had both options explicitly pinned false, so this correction does not retroactively change that particular result. It does remove an unintended behavioral divergence from future default-option comparisons.

## Cargo release-profile experiment

The remaining proposed explanation was the heavy branch’s release profile:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
incremental = false
```

The main branch uses `lto = true` and does not specify `panic`, `strip`, or `incremental`. To isolate this variable, the exact heavy source tree was copied to a temporary directory and rebuilt twice:

1. Heavy profile: the branch’s exact release profile.
2. Plain profile: `lto = true`, with the extra `panic`, `strip`, and `incremental` settings removed.

The controlled start-position `go nodes 3000000` run used one thread, Hash 256, the same explicit NNUE file, `Adaptive=false`, and `OwnBook=false`.

| Variant | Depth | Nodes reported | NPS | Hashfull |
|---|---:|---:|---:|---:|
| Heavy profile | 14 | 2,184,042 | 908,881 | 8 |
| Exact heavy source, plain profile | 14 | 2,184,042 | 912,679 | 8 |

The profile change altered measured NPS by approximately **0.4%** in this controlled run and did not alter the reported depth, node count, or hashfull. Therefore, the Cargo profile is **not supported as the cause** of the large TT/hashfull anomaly reported in the external 223-game investigation.

A three-way run also produced the following local measurements:

| Variant | Depth | Nodes | NPS | Hashfull |
|---|---:|---:|---:|---:|
| Main binary | 14 | 2,184,042 | 821,995 | 8 |
| Heavy profile | 14 | 2,184,042 | 908,881 | 8 |
| Exact heavy source, plain profile | 14 | 2,184,042 | 912,679 | 8 |

These local results did not reproduce the reported 7–9× hashfull difference. That discrepancy remains open because benchmark command timing, engine build identity, UCI option state, hash implementation state, or result-capture methodology may differ. The local experiment only establishes that the extra Cargo profile settings are not sufficient to explain the observed branch loss.

## Source-level scope

The direct source comparison found that the heavy branch differs from `main` in `adapt.rs`, `uci.rs`, `search.rs`, and `aegis_v4_runtime.rs`. The `search.rs` functional change is fail-closed handling for illegal `searchmoves`; it is not active in an unrestricted start-position search. The heavy UCI path also contains explicit low-time/node/depth gates, book restriction handling, corrected observation timestamps, and known-full MultiPV narrowing. These differences remain candidates for separate controlled measurement even though the reported audit found that `engine_suspect()` was false with zero observations.

The present evidence does not justify claiming a single root cause for the −244 Elo result. It does justify three conclusions:

1. The current heavy branch is decisively weaker in the supplied SPRT configuration.
2. `PersonaSmooth=true` was an unintended default divergence and is now corrected.
3. The extra Cargo release-profile settings do not explain the regression in the controlled local experiment.

## Validation after correction

The corrected branch passed:

- Focused Rust regression test for default-off persona experiments.
- Release compilation of `unchessed-adapter`.
- UCI smoke verification showing `PersonaSmooth ... default false`.
- Git whitespace validation.

The next investigation should isolate one source-level behavior at a time, beginning with an exact reproducibility package containing both binaries, compiler version, NNUE hash, UCI command transcript, per-search `info` lines, and the complete cutechess configuration.

## Adaptive-path reproduction

The follow-up hypothesis was tested by rerunning the exact three-way profile benchmark with `Adaptive=true` explicitly set on all variants, while also pinning `PersonaSmooth=false`, `EngineDetectV2=false`, `OwnBook=false`, one thread, Hash 256 MiB, and the same explicit NNUE file.

| Variant | Depth | Nodes | NPS | Hashfull |
|---|---:|---:|---:|---:|
| Main binary | 13 | 2,545,559 | 899,490 | 10 |
| Heavy profile | 13 | 2,545,559 | 938,627 | 10 |
| Exact heavy source, plain profile | 13 | 2,545,559 | 971,216 | 10 |

The reported 7--9x hashfull anomaly did **not** reappear under `Adaptive=true`. Hashfull remained identical across all three variants, and the exact-source plain-profile build was faster than the heavy-profile build. This narrows the issue but does not fully explain the externally supplied 223-game result.

The independent 223-game SPRT itself was not rerun in this sandbox because `cutechess-cli` is not installed here and no exact launcher or machine-readable game package was included with the attached report. The branch-level conclusion therefore remains based on the supplied SPRT, while the local controlled evidence now rules out both the extra Cargo profile settings and the tested isolated adaptive start-position path as sufficient explanations for the hashfull anomaly.

The next reproducibility requirement is an exact package containing the two binaries, compiler/toolchain information, NNUE hash, UCI transcript, opening suite, cutechess command line, per-game PGNs, and SPRT log. Without that package, a local rerun cannot be claimed as an independent reproduction of the -244 Elo result.
