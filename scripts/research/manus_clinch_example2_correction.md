# Correction needed: Example 2 in the CLINCH/MATCH walkthrough (commit 39083d5)

Example 2 ("CLINCH avoids sterile conversion") describes the engagement
bonus as rewarding a move that "leaves the opponent between roughly 8
and 20 legal choices." That range belongs to the standalone
`engagement_score()` function, which is only wired into **Mode::Match**
(`select_move`'s blunder-sampling weight loop), not Clinch.

The function actually used by `ClinchAdapter` is `engagement_bonus`,
and it works on a different signal -- the probe's *reply count*, not
the position's raw legal-move count:

```rust
fn engagement_bonus(pos: &Position, after: &Position, reply_count: usize) -> f64 {
    let choice = if (3..=20).contains(&reply_count) {
        8.0
    } else {
        0.0
    };
    let queens = pos.bb[0][QUEEN] != 0
        && pos.bb[1][QUEEN] != 0
        && after.bb[0][QUEEN] != 0
        && after.bb[1][QUEEN] != 0;
    choice + if queens { 6.0 } else { 0.0 }
}
```

So for Example 2 specifically (it's describing Clinch), the correct
range is **3 to 20 probe replies** (not 8-35 legal moves), and the
bonus structure is +8.0 for that reply-count band plus +6.0 if both
queens survive -- not the +1.0/+1.0 weighting `engagement_score` uses
for Match. Everything else in the walkthrough (the CLINCH score
formula, the 40cp budget, the phase-aware blunder multipliers and move
thresholds in `natural_blunder_probability`) matched the code exactly
on cross-check -- just this one example mixed up the two engagement
functions.
