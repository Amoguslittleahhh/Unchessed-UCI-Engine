# 34 — Classic pawn-structure terms

## Scope and decision question

This investigation covers only the four classic terms named in item 26 of `/home/ubuntu/upload/pasted_content_7.txt`: **isolated pawns, doubled pawns, backward pawns, and pawn islands**. The question is whether any are absent from the current hand-crafted evaluator, and, if absent, which additions are most plausibly additive given the existing passed-pawn, mobility, rook-file, knight-outpost, piece-square, and material evaluation. This is research only: no implementation, tuning, default change, commit, push, or Tier 2/3 work was performed.

The standing rule required a real check wherever feasible. I therefore inspected the evaluator source, ran the existing release engine on concrete FENs, and attempted the native core test command. The latter was blocked by the installed Cargo version and is reported as a negative result rather than silently treated as a pass.

## Source and code audit

The targeted searches were run from `/home/ubuntu/Unchessed-UCI-Engine`:

```text
for t in isolated doubled backward island; do
  rg -n -i "$t" unchessed-core/src unchessed-adapter/src || true
done
```

The result was empty for all four terms. A broader `rg -n -i "isolat|doubl|backward|island|passed pawn|passed_pawn|pawn" unchessed-core/src` found passed-pawn comments and code, but no implementations for the four requested terms. This verifies **absence by source inspection** in the core and adapter source paths searched; it is not a claim about every historical artifact or external binary.

The evaluator in [`unchessed-core/src/eval.rs`](../../unchessed-core/src/eval.rs) is a direct material plus piece-square evaluation followed by optional feature terms. At lines 630–653, `evaluate` loops over each piece and adds material and PST values. The only explicit pawn-structure feature in the inspected scoring path is passed-pawn evaluation (lines 665–736). It detects enemy pawns in the three-file forward mask and an own-pawn forward-file condition, then applies rank-scaled middlegame/endgame values. It also discounts a technically passed pawn when its next square is occupied or attacked. Mobility follows at lines 738–746, rook-file/7th-rank terms at 748–754, and knight outposts at 756 onward. There is no isolated, doubled, backward, or island count, mask, penalty, UCI parameter, or test.

This distinction matters: a pawn PST is not an isolated/doubled/backward/island term, and the passed-pawn test is not a general pawn-structure evaluation. The existing passed-pawn logic can indirectly react to some pawn placements, but it does not identify these four properties. The comments and existing tests explicitly describe passed-pawn scaling and blockades, not the requested structural weaknesses.

## Real-world Tier 1 check

The repository contained a prebuilt `target/release/unchessed-adapter`, so I used it without changing source or defaults. The exact command was:

```text
cd /home/ubuntu/Unchessed-UCI-Engine
{ echo uci; echo 'setoption name Threads value 1';
  echo 'setoption name OwnBook value false';
  echo 'position fen 4k3/8/8/8/8/8/P7/4K3 w - - 0 1'; echo 'go depth 1';
  echo 'position fen 4k3/8/8/8/8/8/PP6/4K3 w - - 0 1'; echo 'go depth 1';
  echo 'position fen 4k3/8/8/8/8/8/1P6/4K3 w - - 0 1'; echo 'go depth 1';
  echo 'position fen 4k3/8/8/8/8/8/P1P5/4K3 w - - 0 1'; echo 'go depth 1'; echo quit; }
| timeout 15s target/release/unchessed-adapter 2>&1
| grep -E 'position|info depth|info string \\[Unchessed\\] eval|bestmove'
```

Relevant output was:

```text
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info depth 1 multipv 1 score cp 142 nodes 8 nps 8000 hashfull 0 time 0 pv a2a4
bestmove e1f1
info depth 1 multipv 1 score cp 250 nodes 11 nps 11000 hashfull 0 time 0 pv a2a4
bestmove a2a4
info depth 1 multipv 1 score cp 136 nodes 10 nps 10000 hashfull 0 time 0 pv e1e2
bestmove e1d1
info depth 1 multipv 1 score cp 252 nodes 10 nps 10000 hashfull 0 time 0 pv a2a4
bestmove a2a4
```

These are real engine outputs, not a synthetic evaluator claim. They show that the release binary used the hand-crafted path and that different pawn placements materially changed the shallow score. However, they **do not isolate** a structural penalty: the positions also change material, PST squares, passed-pawn status, king distance, and legal move choices. Therefore the correct conclusion is negative: this smoke test verifies a runnable code path, but cannot attribute the score differences to isolated, doubled, backward, or island logic. No strength or Elo conclusion is warranted.

I also ran:

```text
cargo test -p unchessed-core --lib
```

It failed before compilation:

```text
error: failed to parse lock file at: /home/ubuntu/Unchessed-UCI-Engine/Cargo.lock
Caused by:  lock file version 4 requires -Znext-lockfile-bump
```

The installed toolchain is `cargo 1.75.0`; this is a toolchain/lockfile compatibility blocker, not a test failure in the evaluator. Installing or selecting another toolchain, adding diagnostic code, and running a controlled term-isolation suite were outside this no-implementation investigation.

## What the terms mean

The [Chess Programming Wiki pawn-structure overview](https://www.chessprogramming.org/Pawn_Structure) lists backward, doubled, isolated, and pawn islands among standard pawn-structure concepts. It also notes that pawn evaluation is commonly cached in a pawn hash because pawn structure changes slowly and that strictly pawn-related information can be stored there. That is relevant architectural guidance, not evidence that this repository has such a cache: the inspected repository path has no separate pawn hash term for these features.

The [isolated-pawn reference](https://www.chessprogramming.org/Isolated_Pawn) defines an isolated pawn as one with no same-colour pawn on either neighboring file. It is usually treated similarly to a backward pawn, often more severely because it is open to attack; central isolated pawns can be balanced by increased mobility. Thus a flat isolated-pawn penalty risks charging a weakness that existing mobility already partly prices, especially for an isolated queen's pawn.

The [doubled-pawn reference](https://www.chessprogramming.org/Doubled_Pawn) defines doubled pawns as multiple same-colour pawns on one file and warns that their value depends on context. It specifically points to Larry Kaufman's *All About Doubled Pawns* and describes file-dependent exchange potential and “crippled majority” effects. This argues against a universal large penalty. A small, context-sensitive term is more defensible than treating every doubled pair as equally bad; some doubled pawns create open files, control useful squares, or support a passed pawn.

The [backward-pawn reference](https://www.chessprogramming.org/Backward_Pawn) defines backwardness in terms of a pawn that cannot be defended by own pawns, whose stop square lacks pawn protection, and whose advance is controlled by an enemy sentry. It also warns that a prospective backward pawn can retain a useful tempo. This is substantially more than a purely geometric “pawn has no supporting neighbour” test and requires enemy pawn attacks, the stop square, advance legality, and ideally tactical/context checks. A naive implementation would be especially prone to double-counting isolatedness and weak squares.

The [pawn-island reference](https://www.chessprogramming.org/Pawn_Islands) defines an island as a group of one or more same-colour pawns separated from another group by files without own pawns. Fewer islands are advantageous all else equal because the groups are easier to defend. Island count is a coarse global proxy for defensive workload, not necessarily an independent weakness in every position. It overlaps with isolated and doubled structure and should not be stacked at full strength with both.

As a reference-engine contrast, current [Stockfish `evaluate.cpp`](https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/evaluate.cpp) shows that modern Stockfish’s outer evaluation is NNUE-driven rather than a directly comparable classical term list. It therefore supports the general caution that importing old hand-tuned constants from another engine is not evidence for this engine. The source is useful context, but no Stockfish constant was used as a proposed value here.

## Additive-priority assessment

The priority ranking below is a research hypothesis, not a measured Elo result. “Additive” means likely to add information not already represented by the current evaluator, not guaranteed to improve playing strength.

| Priority | Candidate | Why it may add information | Main overlap/risk | Tier 1 recommendation |
|---|---|---|---|---|
| 1 | **Isolated pawn**, especially with half-open-file and central/advanced context | Very cheap bitboard test; absent from current code; identifies a common weakness that passed-pawn logic does not identify | Overlaps with backwardness, rook-file pressure, weak squares, and mobility; IQP can buy activity | Best first diagnostic candidate, but only as a pawn-hash-style feature with a small phase/context table and an isolated A/B test |
| 2 | **Doubled pawns**, context-sensitive | Also cheap to detect by per-file counts; captures structural information not represented by passed-pawn scoring | Many doubled pawns are useful; a flat penalty double-counts lost material/weakness and can punish active majorities | Worth a second isolated diagnostic, conditioned on support, front/back pawn, file, and passed-pawn status; do not begin with a universal half-pawn penalty |
| 3 | **Pawn islands** | Extremely cheap global count and a recognized standard concept | Strongly correlated with isolated/doubled structure; coarse and likely redundant once local pawn features exist | Defer until isolated/doubled diagnostics establish residual value; if tested, use a very small endgame-weighted term and test alone |
| 4 | **Backward pawns** | Can model a distinct target when the stop square is controlled and the pawn cannot safely advance | Definition is expensive/context-sensitive and overlaps most with isolated, weak-square, half-open-file, and rook terms; false positives likely | Defer first; only investigate after a precise bitboard definition and a labelled diagnostic suite, not a naive geometric penalty |

The ranking is intentionally conservative. A **single isolated-pawn feature** is the clearest low-cost candidate because it has a compact definition and a plausible residual signal. Doubled pawns are nearly as cheap but need more context. Islands are best viewed as an aggregate regularizer after local features, while backwardness has the highest semantic value but the weakest implementation-to-evidence ratio for this evaluator.

## Proposed evidence gate (not executed)

If this topic is approved later, the next Tier 1 work should remain diagnostic and default-off. First add no strength-changing code: build a pure bitboard analyser or offline script over the repository’s real PGN/FEN data that labels each side’s four structures under an explicitly written definition. Measure prevalence, co-occurrence, phase/piece-count distribution, half-open-file frequency, and correlation with existing terms. This would reveal whether an island feature adds anything after local structure labels.

Second, construct paired FENs that hold material and piece-square placement as constant as possible while changing one structural property, and run both the hand evaluator and a reference engine. The existing release probe demonstrates that engine output is available, but it also demonstrates why such pairs are necessary: raw score changes are confounded. Third, if a candidate survives the diagnostic, add a percentage-gated, default-off term and run unit tests plus a bounded paired-game test; no default change or Elo claim should follow from static eval examples.

A useful implementation design would compute all pawn-only features once and cache them by a pawn hash, consistent with the architecture described by the Chess Programming Wiki. That is a future design recommendation, not an assertion that the current engine already has a pawn hash. Any term must also avoid charging the same weakness repeatedly: for example, an isolated doubled pawn can trigger several labels, but the combined penalty needs explicit caps or mutually exclusive components.

## Verified versus assumed

| Statement | Status |
|---|---|
| Passed-pawn evaluation exists in `unchessed-core/src/eval.rs` and is tunable through `PassedPawnMgPct`/`PassedPawnEgPct` | **Verified** by source inspection and UCI option output |
| Isolated, doubled, backward, and island terms are absent from the searched core/adapter eval source | **Verified** by targeted `rg` and evaluator inspection; scope is the searched source paths |
| Existing evaluator also includes material/PST, mobility, rook-file/7th, and knight-outpost terms | **Verified** by `eval.rs` inspection |
| The release binary ran the hand-crafted evaluator on the supplied FEN probes | **Verified** by exact UCI output |
| The FEN probe isolates any one of the four terms | **Not verified; explicitly false as a conclusion** because material/PST/passed-pawn/king effects are confounded |
| Native `cargo test -p unchessed-core --lib` passes | **Not verified; blocked** by Cargo 1.75.0 parsing lockfile version 4 |
| Isolated pawn is the most promising first addition | **Research recommendation/hypothesis**, based on compact definition and likely residual signal, not an Elo measurement |
| Doubled-pawn context is more important than a flat penalty | **Authoritative design rationale**, supported by CPW’s file/context discussion; not engine-specific measurement |
| Backwardness will improve strength if implemented | **Unknown**; no implementation or game test was performed |
| Pawn islands are additive after local terms | **Unknown and likely redundant**, requiring data/paired testing |

## Recommendation

**Do not implement any of the four terms yet.** The source audit confirms a real gap, but the required real evaluator test cannot identify an additive strength signal, and the native test suite is blocked by the old Cargo toolchain. Preserve the current defaults and defer Tier 2/3 work.

If one follow-up is funded within Tier 1, prioritize an offline prevalence/co-occurrence and paired-FEN diagnostic for a **small, context-sensitive isolated-pawn term**, followed by doubled pawns. Treat pawn islands as a later, low-weight aggregate candidate and backward pawns as the last candidate unless a precise definition and labelled evidence justify their complexity. Any eventual strength claim requires an isolated default-off A/B and a real paired-game/SPRT gate; literature definitions and shallow static scores alone are insufficient.

## References

1. [Chess Programming Wiki — Pawn Structure](https://www.chessprogramming.org/Pawn_Structure)
2. [Chess Programming Wiki — Isolated Pawn](https://www.chessprogramming.org/Isolated_Pawn)
3. [Chess Programming Wiki — Doubled Pawn](https://www.chessprogramming.org/Doubled_Pawn)
4. [Chess Programming Wiki — Backward Pawn](https://www.chessprogramming.org/Backward_Pawn)
5. [Chess Programming Wiki — Pawn Islands](https://www.chessprogramming.org/Pawn_Islands)
6. [Stockfish — `src/evaluate.cpp`](https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/evaluate.cpp)
7. Repository task notes: [`/home/ubuntu/upload/pasted_content_7.txt`](file:///home/ubuntu/upload/pasted_content_7.txt)
8. Repository evaluator: [`unchessed-core/src/eval.rs`](../../unchessed-core/src/eval.rs)

**Report status:** complete; research only; no implementation, default change, commit, push, expensive training, or cloud spend.
