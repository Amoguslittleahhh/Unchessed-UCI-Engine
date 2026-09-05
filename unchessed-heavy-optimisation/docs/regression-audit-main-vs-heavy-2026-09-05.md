# Regression Audit: Main versus Heavy Optimization

**Date:** 5 September 2026

## Corrected supplied SPRT interpretation

Two externally supplied real `cutechess-cli` results compared `main` with `unchessed-heavy-optimisation` under matched conditions. The original prose interpreted the sign backwards. `cutechess-cli` reports score and Elo difference relative to the first-named engine, which was `main`. The raw summaries show that `main` scored below 0.5 and therefore lost to the heavy branch.

The 223-game fast gate used the same SHA-256-verified NNUE file, one thread, 256 MiB Hash, `Adaptive=true`, `OwnBook=false`, paired openings with colors reversed, `elo0=0`, `elo1=5`, and `alpha=beta=0.05`. The supplied result was approximately **−244.4 ± 48.0 Elo for main relative to heavy optimization**, with LOS 0.0%. The accompanying termination summary reported 146 losses and 18 wins for main. Both engines completed without illegal moves, disconnects, crashes, or time forfeits. Twelve games per side were aborted when the SPRT boundary was reached; these were test termination events rather than failures.

A second supplied 200-game low-concurrency control reported `Score of Main vs HeavyOpt: 21 - 140 - 39 [0.203]` and `Elo difference: -238.1 +/- 50.7`, again relative to main. This independently reproduced the direction and approximate magnitude: **heavy optimization outperformed main by roughly 240 Elo** under the supplied Adaptive=true configuration.

The corrected conclusion is therefore that the tested heavy branch is materially stronger than `main` in these supplied matches. The result does not prove that every heavy-branch change is beneficial, because several changes are coupled. It does establish that the earlier “heavy optimization is weaker” interpretation was incorrect.

## PersonaSmooth default divergence

The supplied audit also identified a real default mismatch:

| Branch | `Options::default()` | UCI advertisement |
|---|---:|---|
| `main` | `false` | `default false` |
| Heavy optimization before correction | `true` | `default true` |
| Heavy optimization after correction | `false` | `default false` |

The repository’s persona design documents explicitly state that `PersonaSmooth` and `EngineDetectV2` are default-off until a relevant strength test supports a product decision. The heavy branch’s `true` default was inconsistent with that documented contract and was treated as an accidental side effect of the persona-stability work.

The default has been restored to `false`. The existing regression test now asserts the documented default, and the rebuilt UCI adapter advertises `PersonaSmooth type check default false`.

The supplied SPRT pinned both options false, so this correction does not change those match results. It removes an unintended default divergence from future comparisons.

## Cargo release-profile experiment

The proposed Cargo explanation was tested independently. The heavy profile is:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
incremental = false
```

The main branch uses `lto = true` without the extra `panic`, `strip`, and `incremental` settings. The exact heavy source tree was rebuilt with both profiles. On a controlled start-position `go nodes 3000000` run with one thread, Hash 256, the same explicit NNUE file, and `Adaptive=false`, both variants reached depth 14, reported 2,184,042 nodes, and hashfull 8.

| Variant | NPS | Hashfull |
|---|---:|---:|
| Heavy profile | 908,881 | 8 |
| Exact heavy source, main-style plain profile | 912,679 | 8 |

The profile difference was approximately 0.4%. It is not the source of the roughly 240 Elo advantage, and it does not explain the externally reported 7--9x hashfull difference.

A three-way local run also failed to reproduce the reported hashfull anomaly. With the same isolated command, main measured 821,995 NPS and hashfull 8, while heavy and exact-source plain-profile builds measured 908,881 and 912,679 NPS with hashfull 8.

## Adaptive-path reproduction

The exact three-way profile benchmark was then rerun with `Adaptive=true` explicitly set on all variants, while pinning `PersonaSmooth=false`, `EngineDetectV2=false`, `OwnBook=false`, one thread, Hash 256 MiB, and the same NNUE file.

| Variant | Depth | Nodes | NPS | Hashfull |
|---|---:|---:|---:|---:|
| Main binary | 13 | 2,545,559 | 899,490 | 10 |
| Heavy profile | 13 | 2,545,559 | 938,627 | 10 |
| Exact heavy source, plain profile | 13 | 2,545,559 | 971,216 | 10 |

The reported 7--9x hashfull anomaly did not reappear under `Adaptive=true`. This local isolated benchmark does not reproduce the supplied hashfull observation, but it also does not challenge the supplied paired-game strength results. Those results are the higher-value evidence because they measured complete games rather than one start-position search.

## Leading mechanism: known_full MultiPV narrowing

The strongest source-level strength hypothesis is the heavy branch’s `known_full` change in `uci.rs`:

```rust
let known_full = adaptive_now
    && !job.opt.limit_strength
    && model.lock().unwrap().engine_suspect();
let multipv_search = if adaptive_now && !known_full {
    multipv_shown.max(5)
} else {
    multipv_shown
};
```

The main branch widens MultiPV to at least five whenever adaptive mode is active. The heavy branch returns to the configured MultiPV, normally one, after the opponent model identifies a strong or computer-like opponent. Tracking five principal variations costs search efficiency and can reduce pruning effectiveness. If the persona system is already in `FULL` mode and does not use alternative lines for humanization, that extra MultiPV work is an unnecessary strength tax.

The reported telemetry showed 100% `FULL` decisions and zero persona transitions in the strong-engine match. This makes the change a credible explanation for a large portion of the observed gap. It remains a hypothesis until isolated in a dedicated match.

## Required isolated experiment

The next experiment should build a candidate from untouched `main` with exactly one behavioral change: the `known_full` MultiPV narrowing shown above. It should not include the heavy branch’s other persona, UCI, search-restriction, cache, or documentation changes.

The candidate should be tested against unmodified main with the same NNUE, options, openings, threads, Hash, time control, and SPRT parameters. The expected interpretation is:

| Result | Interpretation |
|---|---|
| Approximately +240 Elo for known_full-only | Strong evidence that the MultiPV narrowing is the primary source of the supplied gain. |
| Small positive result | The change helps, but other heavy-branch changes contribute materially. |
| Near-even result | The full-branch gain depends on interaction with another change or on an environment/configuration difference. |
| Negative result | The original result depends on coupled behavior, benchmark state, or an unisolated factor. |

The isolated candidate must pass the usual UCI, legality, crash, and SPRT gates before any default or main-branch port is considered.

## Independent reproduction limitation

The supplied 223-game and 200-game results are real reports, but they were not independently rerun in this sandbox because `cutechess-cli` is unavailable and the attached reports did not include complete machine-readable game packages. The corrected direction is based on the raw score and termination summaries supplied in the reports. A portable reproduction package should contain both binaries, compiler/toolchain information, NNUE hash, UCI transcript, opening suite, exact cutechess command, per-game PGNs, and SPRT logs.

## Validation after default correction

The corrected branch passed:

- Focused Rust regression test for default-off persona experiments.
- Release compilation of `unchessed-adapter`.
- UCI smoke verification showing `PersonaSmooth ... default false`.
- Git whitespace validation.

## Known_full-only isolation candidate

A clean temporary copy of the untouched `main` clone was created at `/tmp/unchessed-known-full-isolation`. The only source change was the heavy branch’s `known_full` MultiPV narrowing block. The isolated candidate built successfully with the main release profile and produced adapter SHA-256:

```text
ae32c419c39a338de6fd15971160f32b46851f3bb90ec4fe8843b7d5998f4dcb
```

The candidate passed UCI handshake, `readyok`, explicit NNUE loading, and a 100,000-node start-position smoke search. The main clone remained clean and unchanged. A full paired-game SPRT was not run because `cutechess-cli` is unavailable in this sandbox.

The reproducible builder is committed at `scripts/research/build-known-full-isolation.sh`.
