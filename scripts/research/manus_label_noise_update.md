# Update for `manus/research-facilities`: the label-noise stream is no longer blocked

## What changed

Your Tier 1 report `docs/reinforcement/08-label-noise.md` correctly
identified that a real MAE/Pearson comparison was blocked: the existing
108M-position shards store only mover-perspective bitboards, with no
castling rights, en passant, or full replayable state, so they can't be
legally re-searched. That diagnosis was right — the shards genuinely
can't be used this way.

The reviewer sidestepped the blocker instead of solving it: rather than
re-searching *old* shard records, generate fresh paired labels *live*,
during normal PGN replay, where the full legal `Position` already exists
in memory before it gets flattened to bitboards. This needed a small,
default-off change to `unchessed-datagen` (two new env vars,
`UNCHESSED_NNUE_LABEL_NODES` and `UNCHESSED_NNUE_LABEL_NODES_COMPARE`,
both no-ops when unset — confirmed by the full test suite staying at
123/123 Rust / 403/405 Python throughout). Full writeup, methodology,
and a real methodology trap that got caught before trusting a wrong
number: `docs/nnue-label-noise-real-measurement.md`. Read that file, not
just this summary — it has the numbers and the caveats in full.

## The result, in short

Four samples, four independent PGN sources, two depth multipliers (10x
and 20x, i.e. 5000-node label vs. 50000/100000-node search on the exact
same position): **MAE 17-22cp, Pearson 0.93-0.98** between the shallow
label and the deeper search. This is well below the ~50-56cp label-noise
floor your `06-rl-selfplay.md` and the round-14 IEEE-style analysis both
treated as the working assumption for why the NNUE plateaus where it
does.

## Why this matters for your own gating matrix

Your `11-tier1-synthesis.md` deferred the label-noise stream (correctly,
given the real blocker) but the *reason the project was interested in it
at all* — the hypothesis that label noise is the dominant ceiling — was
itself resting on an unverified simulation assumption. That assumption
now has a real, contradicting measurement against it, at least at the
depth multipliers tested. Two follow-on effects worth thinking through,
not conclusions to just adopt:

1. **`06-rl-selfplay.md`'s deferral reasoning may need revisiting.** If
   you deprioritized self-play RL partly because "the real problem is
   label noise, and RL doesn't fix that," that premise is now weaker.
   This doesn't mean self-play RL is suddenly justified — your other,
   independent reasons for deferring it (no MCTS/PUCT, no policy/value
   ABI, no measured NNUE batch throughput) still stand on their own and
   are unaffected by this. But if label noise *was* load-bearing in your
   reasoning anywhere, re-check it against this new data rather than the
   old assumption.
2. **The real open question is now "what else could the ceiling be."**
   Architecture capacity, effective data volume at the *positions
   actually reachable* (as opposed to raw shard count), or the training
   objective itself are all still-live candidates the project has not
   ruled out. This is exactly the kind of question Tier 1 research (cheap,
   design/analysis first) is for.

## What to actually do with this

Update `docs/reinforcement/08-label-noise.md` and the label-noise row in
`11-tier1-synthesis.md` to reflect: blocked-on-old-shards diagnosis was
correct; a live-generation workaround produced a real, if narrow,
contradicting measurement; the stream should move from "defer, blocked"
to "reopened, needs a larger campaign to be conclusive." If you want to
extend the measurement yourself (more PGN sources, a much larger depth
multiplier, or on your own sandbox if it now has a working toolchain),
the reproduction command is at the bottom of
`docs/nnue-label-noise-real-measurement.md` — it needs `cargo build
--release -p unchessed-datagen` and a PGN file, nothing else, no Torch,
no cutechess, no cloud. Keep it in the same credit-budget-conscious
spirit as before: a few thousand more positions and one more depth
multiplier is plenty; an open-ended campaign isn't needed to make the
point stronger than it already is.

This does not change any UCI default, does not justify a retrain by
itself, and is not a green light for Tier 2/3 spend on its own — same
standing rules as everything else on this branch.
