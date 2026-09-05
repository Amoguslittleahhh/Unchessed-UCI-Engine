# Real SPRT result: main vs `unchessed-heavy-optimisation`, and an open regression

Ran the predeclared gate you specified (relayed 2026-09-05): main vs
`unchessed-heavy-optimisation`, same explicit `EvalFile`, `Adaptive=true`,
`OwnBook=false`, paired openings with colors reversed, `elo0=0 elo1=5
alpha=beta=0.05`, fast screening control first. Real cutechess-cli run,
not a simulation.

## Result

**SPRT: H0 accepted. Elo difference -244.4 +/- 48.0, LOS 0.0%, 223 games**
(tc=5+0.05, single-threaded, Hash=256, real NNUE both sides, SHA-256
verified identical file on both engines).

This is not "unproven improvement" -- it is a large, statistically
decisive loss for `unchessed-heavy-optimisation` against main under
matched conditions. Stability was otherwise clean: zero illegal moves,
zero disconnects/crashes, zero time forfeits on either side across all
223 games (12 games per side were aborted in-flight when the SPRT bound
was hit, not failures).

## A confound caught and fixed before it produced a false result

A first attempt at this gate, run without pinning `PersonaSmooth`
explicitly, scored main at only 0.222 after 36 games. Investigating why
surfaced a real default divergence: `unchessed-heavy-optimisation`'s
`Options::default()` ships `persona_smooth: true`
(`unchessed-core/src/uci.rs`), while main's is `false`. Per
`docs/persona-sprt-gate.md` neither PersonaSmooth nor EngineDetectV2 is
a promoted default anywhere, so this divergence looks unintentional --
worth checking whether it was an accidental side effect of the persona
research commits on this branch (`e6bac0b`, `6b0d84d`) rather than a
deliberate change. The reported SPRT above has both options pinned
`false` explicitly on both engines to remove this confound.

## Ruled out: target-cpu / build variant

Your own `scripts/benchmark-portable-v3.sh` builds two separate
binaries (`portable`: plain `cargo build --release`; `v3`:
`RUSTFLAGS='-C target-cpu=x86-64-v3'`). The first pass at this
investigation used the plain portable build by mistake. Rebuilt the
real x86-64-v3-optimized binary (host CPU confirmed to support
avx2/bmi2/fma) and re-measured -- the anomaly described below persists
identically on both build variants, so it is not a target-cpu effect.

## An open, unexplained regression: TT fills far faster per node on your branch

Isolated, single-position benchmark (`position startpos`, `go nodes
3000000`, no game history, no persona state involved, same NNUE file,
`Hash=256` explicit on both):

| Engine | depth reached (~12s) | nodes | nps | hashfull at that node count |
|---|---:|---:|---:|---:|
| main | 13 | 2,089,006 | ~1.7-2.2M | 133 (13.3%) |
| heavy-optimisation (v3) | 12 | 1,802,335 | ~1.7-1.8M | 942 (94.2%) |

Your table is filling roughly 7-9x faster per node processed than
main's, on an identical position with zero game/persona state -- and
NPS is consistently 15-25% lower. This reproduces on both the portable
and x86-64-v3 builds, so it is very likely the actual cause of the -244
Elo gap, not a coincidence.

What has been ruled out by direct diff:
- `search.rs`, `tt.rs`, `nnue.rs`, `hce.rs`, `movegen.rs`, `board.rs` are
  **byte-identical** between main and your branch. No hot-path logic
  bug.
- `target-cpu=x86-64-v3` vs generic: ruled out, anomaly present on both.
- The `known_full` MultiPV-narrowing you added (drop to `MultiPV=1`
  once `engine_suspect()` is true, vs main's unconditional
  `MultiPV>=5` under `Adaptive=true`) is real and worth documenting on
  its own, but `OpponentModel::is_computer` defaults `false`, so
  `engine_suspect()` is false with zero observations -- it cannot
  explain the anomaly on an isolated first move with no game history.

What has **not** been ruled out: the differing Cargo release profile
(`panic = "abort"`, `lto = "fat"`, `strip = "symbols"` in your
`unchessed-heavy-optimisation/Cargo.toml`, vs main's plain default
release profile). Since every hot-path source file is identical, this
is the last remaining variable. The clean next experiment is building
your exact source tree under main's plain profile (same target-cpu) and
re-running this same isolated NPS/hashfull bench -- if the anomaly
disappears, the release profile itself is the regression, not any
algorithmic change.

## One honest limitation of this gate

`Adaptive=true` was requested so persona/adaptive behavior would be
exercised, but the bucket-by-mode telemetry (via a tee-wrapper around
each engine's stdout, since this cutechess-cli 1.5.1 build's `-debug`
flag fails outright -- reproduced with a minimal 2-engine repro,
unrelated to this match) shows **100% of decisions on both sides landed
in FULL mode, zero persona transitions**. Against another strong UCI
engine the persona system correctly and immediately locks into full
strength, so Match/Clinch/Punish/Defend never activate. This SPRT is
effectively a pure search/eval-strength comparison despite
`Adaptive=true` -- worth knowing before reading anything into the
requested mode-bucket breakdown, which is trivial by construction here.

**Ask, concretely:** given a -244 Elo result this large and this clean
(no crashes, no confound left unaccounted for), please check whether
`unchessed-heavy-optimisation`'s Cargo release profile is the actual
cause before any more hardware/speed claims are made from that copy --
and separately, confirm whether the PersonaSmooth default flip was
intentional.
