# Reproduction gap in the profile-regression experiment: Adaptive state, not just Cargo profile

Good audit overall -- the PersonaSmooth default correction matches
independent verification here exactly, and the release-profile
experiment (fat LTO / abort / stripped vs plain, 0.4% NPS delta,
identical depth/nodes/hashfull) is a real, well-controlled result: the
Cargo profile is ruled out.

The non-reproduction of the reported 7-9x hashfull anomaly is most
likely explained by a configuration mismatch, not a measurement error
on either side.

## The gate that matters

`unchessed-core/src/uci.rs`:

```rust
let adaptive_now =
    job.ident_adaptive && (job.opt.adaptive || job.opt.limit_strength) && game_mode;
```

`game_mode` is `Limits::is_game_mode()` (`search.rs`), which is `true`
whenever `nodes.is_some()` -- i.e. **a plain `go nodes N` command already
counts as game mode**, and `Options::default().adaptive` is `true` on
both branches. So a `go nodes 3000000` bench with `Adaptive` left
unset (the setup used for the original 223-game-adjacent isolated
bench here) runs with `adaptive_now = true` on both engines by default.

The `profile-regression-bench.sh` control explicitly sends
`setoption name Adaptive value false` on all three variants. That is a
real, deliberate control for isolating the Cargo-profile variable, but
it also switches off the one documented source-level behavior that
differs between the branches (the `known_full` MultiPV-narrowing added
in `uci.rs`, and whatever else is gated on `adaptive_now`). A
null result under `Adaptive=false` does not rule out something in the
adaptive path -- it rules out the profile while testing a config that
cannot exercise that path at all.

Note: `engine_suspect()` is `false` with zero observations
(`OpponentModel::is_computer` defaults `false`), so on a truly fresh
`go nodes` call with no game history, `known_full` should also be
`false` on both sides, which would make the MultiPV width identical
too. That was already checked and doesn't obviously explain the
anomaly either -- which is exactly why this remains open rather than
resolved by source inspection alone.

## What would actually settle it

Two things, in order of priority:

1. **Re-run the actual 223-game SPRT gate itself, independently, on
   your own hardware.** Everything so far past the first result has
   been isolated single-position benches on one machine (this
   reviewer's WSL box). The one thing that hasn't been independently
   reproduced is the result that actually matters -- the -244 Elo
   paired-game outcome. If your own cutechess run of the identical gate
   (same options: `elo0=0 elo1=5 alpha=beta=0.05`, `Adaptive=true`,
   `OwnBook=false`, `PersonaSmooth=false EngineDetectV2=false` pinned
   on both, matched `EvalFile` SHA-256, `tc=5+0.05`, paired openings
   reversed -- see `scripts/research/wsl_sprt_main_vs_heavyopt_fast.sh`
   for the exact cutechess-cli invocation) lands anywhere near -244 Elo
   on different hardware, that rules out a single-machine artifact
   entirely and confirms this is a real, portable regression rather
   than something specific to one reviewer's environment. If it comes
   back materially different (e.g. roughly even), that itself is a
   critical and separate finding worth chasing.

2. Re-run `profile-regression-bench.sh`'s exact three-way comparison,
   but with `Adaptive` left at its default (`true`, or set explicitly
   to `true` for clarity) instead of `false`, on the same `go nodes
   3000000` startpos position. If the hashfull/NPS gap reappears under
   `Adaptive=true` and stays absent under `Adaptive=false`, that pins
   the anomaly to the adaptive code path specifically (not the Cargo
   profile, already ruled out, and not raw search/eval, already
   byte-identical) -- and narrows the next step to instrumenting
   whatever runs per-node or per-`go`-call under `adaptive_now` on the
   heavy branch that doesn't run under it on main.

(1) is the higher-value ask: it's the actual gate result, not a proxy
for it, and an independent same-result on different hardware is worth
more than any number of single-machine isolated benches from either
side.
