# Unchessed Heavy Optimisation

This directory is an isolated copy of `main` at commit `55da896e1870a51cbbd9a31518fe4e5a7c505d38`. The parent branch remains untouched. The optimisation is intentionally concentrated in build/code-generation settings so the engine's chess behavior and adaptive policy remain compatible with the source snapshot.

## Preserved core contract

The following behavior is inherited unchanged from `main` and must not be removed or flattened:

| Core function | Preservation requirement |
|---|---|
| Persona modes | `Full`, `Match`, `Clinch`, `Punish`, and `Defend`, each with its distinct move-selection policy |
| Match mode | Below 2200 Elo, candidate sampling includes all legal moves and shallow probes rather than only engine top-K lines |
| Clinch mode | Trap selection uses the opponent's best-vs-second-best reply gap and favors keeping both queens in drawish positions |
| Punish mode | Found mates are forced; otherwise forcing captures/checks are preferred when ahead |
| Defend mode | Maximum-resistance engine best move |
| Opponent modeling | Live Elo estimate from choice-weighted cp-loss plus time-usage engine-detection signal, feeding `EngineDetectV2` |
| Troll book | Risk-tiered tricky/dubious/meme lines with shallow safety recheck and fallback when the line evaluates below -60cp |
| Contempt | Draw scoring remains tied to the active persona rather than a flat constant |
| Low-time gating | Luxury observation, Clinch, and troll-refutation probes remain gated by the clock and current move budget |

## Build optimisations

The isolated workspace uses `lto = "fat"`, one codegen unit, `panic = "abort"`, symbol stripping, disabled incremental compilation, and `target-cpu=native`. These settings reduce cross-crate overhead and let LLVM use the host CPU's instruction set without changing the search tree or policy logic. The native target setting means release binaries should be benchmarked and deployed on compatible CPUs.

Build and test with:

```bash
cd unchessed-heavy-optimisation
bash scripts/build-and-test.sh --release
```

The current sandbox does not have `rustc`/`cargo`, so compilation must be run in CI or on a Rust-enabled host. The copied source was created from `main`; no files in `main` were edited.
